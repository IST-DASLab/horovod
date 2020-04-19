#include "cuda_functions.h"
#include <cstdlib>
#include <iostream>
#include <string>

const int MAX_THREADS_PER_BLOCK = 1024;
const float EPS = 1e-6;
const int PACK_SIZE = 8;
constexpr int BLOCKS_PER_GRID(int num_elems) { return 512; }
// constexpr int BLOCKS_PER_GRID(int num_elems) {
//  return num_elems + (MAX_THREADS_PER_BLOCK - 1);
//}

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t cuda_result = condition;                                       \
    if (cuda_result != cudaSuccess) {                                          \
      throw std::runtime_error(                                                \
          std::string(#condition) + " on line " + std::to_string(__LINE__) +   \
          " returned: " + cudaGetErrorString(cuda_result));                    \
    }                                                                          \
  } while (0)

/*
==== Utilite functions. ===
*/
__device__ int toInt(unsigned char* z) {
  return ((unsigned int)z[0] & 0xFF) << 24 | ((unsigned int)z[1] & 0xFF) << 16 |
         ((unsigned int)z[2] & 0xFF) << 8 | ((unsigned int)z[3] & 0xFF);
}

__global__ void _init_curand(unsigned int seed, CurandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
/* we have to initialize the state */
  curand_init(seed,  /* the seed can be the same for each core, here we
  pass the
                        time in from the CPU */
              index, /* the sequence number should be different for each
              core
                        (unless you want all cores to get the same
                        sequence of numbers for some reason - use thread
                        id! */
              0, /* the offset is how much extra we advance in the
              sequence for
                    each call, can be 0 */
              &states[index]);

//  unsigned char z[4];
//  for (int i = 0; i < 4; i++)
//    z[i] = 128 + index % 128;
//  states[index] = toInt(z);
}

inline __device__ void TausStep(unsigned char& z, unsigned int S1,
                                unsigned int S2, int S3, unsigned M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  z = (((z & M) << S3) ^ b);
}

inline __device__ void LCGStep(unsigned char& z, unsigned A, unsigned C) {
  z = (A * z + C);
}

inline __device__ float HybridTaus(int* state_p) {
  unsigned char z[4];
  int state = *state_p;
  // present int as char array.
  z[0] = (state >> 24) & 0xFF;
  z[1] = (state >> 16) & 0xFF;
  z[2] = (state >> 8) & 0xFF;
  z[3] = state & 0xFF;
  TausStep(z[0], 13, 19, 12, 4294967294UL);
  TausStep(z[1], 2, 25, 4, 4294967288UL);
  TausStep(z[2], 3, 11, 17, 4294967280UL);
  LCGStep(z[3], 1664525, 1013904223UL);
  *state_p = toInt(z);
  return (z[0] ^ z[1] ^ z[2] ^ z[3]) / 256.0;
}

__device__ float GetRand(CurandState* state_p) {
  return curand_uniform(state_p);
//  return HybridTaus(state_p);
}

__global__ void _add(int n, const float* x, const float* y, float* sum) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    sum[i] = x[i] + y[i];
  }
}

void CUDA_init_curand(CurandState* states, int num_elems, unsigned int seed,
                      cudaStream_t stream) {
  _init_curand<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
                 stream>>>(seed, states);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

int CUDA_get_curand_array_size(int num_elems) {
  return BLOCKS_PER_GRID(num_elems) * MAX_THREADS_PER_BLOCK *
         sizeof(CurandState);
}

void CUDA_add(int n, const float* x, float* y, float* sum,
              cudaStream_t stream) {
  int blocks = BLOCKS_PER_GRID(n);
  int num_threads = MAX_THREADS_PER_BLOCK;
  _add<<<blocks, num_threads, 0, stream>>>(n, x, y, sum);
}

/*
==== Functions for quantization preparation. ===
*/
__global__ void MaxMin_find_meta(const float* x, unsigned char* meta, int n,
                                 int bucket_size) {
  float* maxandmin = (float*)meta;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
    float mmin = x[i * bucket_size];
    float mmax = x[i * bucket_size];
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      mmin = fminf(mmin, x[j]);
      mmax = fmaxf(mmax, x[j]);
    }
    maxandmin[2 * i] = mmax;
    maxandmin[2 * i + 1] = mmin;
  }
}

__global__ void LinfNorm_find_meta(const float* x, unsigned char* meta, int n,
                                   const int bucket_size) {
  float* norms = (float*)meta;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
    float bmax = fabsf(x[i * bucket_size]);
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      bmax = fmaxf(bmax, fabsf(x[j]));
    }
    if (fabsf(bmax) < EPS)
      bmax += EPS;
    norms[i] = bmax;
  }
}

__global__ void L2Norm_find_meta(const float* x, unsigned char* meta, int n,
                                 const int bucket_size) {
  float* norm = (float*)meta;
  int num_buckets = (n + bucket_size - 1) / bucket_size;
  unsigned char* max_logs = meta + num_buckets * sizeof(float);
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (int i = index; i < num_buckets; i += stride) {
    float bnorm = 0.0;
    float bmax = fabsf(x[i * bucket_size]);
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      bnorm += x[j] * x[j];
      bmax = fmaxf(bmax, fabsf(x[j]));
    }
    bnorm = sqrt(bnorm);
    if (fabsf(bnorm) < EPS)
      bnorm += EPS;
    norm[i] = bnorm;
    bmax /= bnorm;
    max_logs[i] = (unsigned char)(floor(-log2(bmax)));
  }
}

// Single value quantization functions
inline __device__ unsigned char MaxMinEncodeValue(float input, float* feedback,
                                                  unsigned char* meta_info,
                                                  unsigned int idx,
                                                  int bucket_size, int bits,
                                                  float rand, void* ctx) {
  int bucket_no = idx / bucket_size;
  float* maxmin = ((float*)meta_info) + 2 * bucket_no;
  if (maxmin[0] - maxmin[1] < EPS) {
    return 0;
  }
  int divisor = (1 << bits) - 1;
  float min = maxmin[1];
  float unit = (maxmin[0] - min) / divisor;
  float d = ((input - min) / unit) + rand;
  unsigned char level = (unsigned char)floor(d);
//  if (idx < 8)
//    printf("Encode unit %f min %f idx: %i, d: %f, rand: %f decoded: %f\n", unit,
//           maxmin[1], idx, d, rand, (min + level * unit));
  if (feedback)
    *feedback = input - (min + level * unit);
  return level;
}

inline __device__ float MaxMinDecodeValue(unsigned char input,
                                          unsigned char* meta_info,
                                          unsigned int idx, int bucket_size,
                                          int bits, void* ctx) {
  int bucket_no = idx / bucket_size;
  float* maxmin = ((float*)meta_info) + 2 * bucket_no;
  int divisor = (1 << bits) - 1;
  float min = maxmin[1];
  float unit = (maxmin[0] - min) / divisor;
//  if (idx < 8)
//    printf("Decode unit %f min %f idx: %i, decoded: %f\n", unit, maxmin[1], idx,
//           (min + input * unit));
  return min + unit * input;
}

inline __device__ unsigned char
LinfNormEncodeValue(float input, float* feedback, unsigned char* meta_info,
                    unsigned int idx, int bucket_size, int bits, float rand,
                    void* ctx) {
  int bucket_no = idx / bucket_size;
  float norm = ((float*)meta_info)[bucket_no];
  char sign = (input < -EPS);
  float* levels = (float*)ctx;
  int num_levels = 1 << (bits - 1);

  float d = fabs(input / norm);
  unsigned char level_idx = 0;

  // levels are going 1.0 q_n q_{n-1} ... 0.0
  while (level_idx + 1 < num_levels) {
    if (d - levels[level_idx + 1] > EPS) {
      if (d + (levels[level_idx] - levels[level_idx + 1]) * rand -
              levels[level_idx] <
          -EPS) {
        level_idx++;
      }
      break;
    }
    level_idx++;
  }
  // update error feedback
  if (feedback) {
    float recovered_v = norm * (sign ? -1.0 : 1.0);
    if (bits > 1)
      recovered_v *= (level_idx < num_levels - 1) ? levels[level_idx] : 0.0;
    *feedback = input - recovered_v;
  }
  level_idx |= (sign << (bits - 1));
  return level_idx;
}

inline __device__ float LinfNormDecodeValue(unsigned char input,
                                            unsigned char* meta_info,
                                            unsigned int idx, int bucket_size,
                                            int bits, void* ctx) {
  int bucket_no = idx / bucket_size;
  float norm = ((float*)meta_info)[bucket_no];
  float* levels = (float*)ctx;
  int num_levels = 1 << (bits - 1);
  char sign = (input & num_levels) ? -1 : 1;
  float decode_value = norm * sign;

  if (bits > 1) {
    input &= num_levels - 1;
    decode_value *= (input < num_levels - 1) ? levels[input] : 0.0;
  }
  return decode_value;
}

struct QuanCtx {
  void* host_ctx;
  int num_buckets;
};

inline __device__ unsigned char L2NormEncodeValue(float input, float* feedback,
                                                  unsigned char* meta_info,
                                                  unsigned int idx,
                                                  int bucket_size, int bits,
                                                  float rand, void* ctx) {
  QuanCtx* qctx = (QuanCtx*)ctx;
  float* levels = (float*)(qctx->host_ctx);
  char sign = (input < -EPS);
  int bucket_no = idx / bucket_size;
  float norm = ((float*)meta_info)[bucket_no];
  unsigned char max_log =
      (meta_info + sizeof(float) * qctx->num_buckets)[bucket_no];
  unsigned int num_levels = 1 << (bits - 1);

  float d = fabs(input / norm);
  unsigned char level_idx;
  if (d < EPS) {
    level_idx = num_levels - 1;
  } else {
    float level_f = -log2(d);
    int c = (int)(ceil(level_f));
    int f = (int)(floor(level_f));
    level_idx = f - max_log;
    if (level_idx < num_levels - 1 &&
        d + (levels[f] - levels[c]) * rand - levels[f] < -EPS) {
      level_idx++;
    }
    level_idx = fminf(level_idx, (unsigned char)(num_levels - 1));
  }

  if (feedback) {
    float decode_value = norm * (sign ? -1.0 : 1.0);
    if (bits > 1)
      decode_value *=
          (level_idx < num_levels - 1) ? levels[max_log + level_idx] : 0.0;
    *feedback = input - decode_value;
  }
  level_idx |= (sign << (bits - 1));
  return level_idx;
}

inline __device__ float L2NormDecodeValue(unsigned char input,
                                          unsigned char* meta_info,
                                          unsigned int idx, int bucket_size,
                                          int bits, void* ctx) {
  QuanCtx* qctx = (QuanCtx*)ctx;
  float* levels = (float*)(qctx->host_ctx);
  int bucket_no = idx / bucket_size;
  float norm = ((float*)meta_info)[bucket_no];
  unsigned char max_log =
      (meta_info + sizeof(float) * qctx->num_buckets)[bucket_no];

  unsigned int num_levels = 1 << (bits - 1);
  char sign = (input & num_levels) ? -1 : 1;
  float decode_value = norm * sign;

  if (bits > 1) {
    input &= num_levels - 1;
    decode_value *= (input < num_levels - 1) ? levels[max_log + input] : 0.0;
  }
  return decode_value;
}

// Packaging functions
#define PACK_ARRAY_FUNC(type)                                                  \
  __global__ void PackArray##type(float* input, unsigned char* meta_info,      \
                                  unsigned char* output, float* feedback,      \
                                  int num_elems, int bucket_size, int bits,    \
                                  void* ctx, CurandState* states) {            \
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;                  \
    unsigned int stride = gridDim.x * blockDim.x;                              \
    int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;             \
    float* feedback_ = nullptr;                                                \
    CurandState local_state = states[tid];                                     \
    float rand;                                                                \
    for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;    \
         i += stride) {                                                        \
      uint64_t value = 0;                                                      \
      for (unsigned int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; \
           j++) {                                                              \
        int idx = i * PACK_SIZE + j;                                           \
        if (feedback)                                                          \
          feedback_ = feedback + idx;                                          \
        rand = GetRand(&local_state);                                          \
        uint64_t encoded =                                                     \
            type##EncodeValue(input[idx], feedback_, meta_info, idx,           \
                              bucket_size, bits, rand, ctx);                   \
        value += (encoded << (j * bits));                                      \
      }                                                                        \
      for (unsigned int j = 0; j < bits && i * bits + j < num_char; j++) {     \
        output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;                \
      }                                                                        \
    }                                                                          \
    states[tid] = local_state;                                                 \
  }

#define UNPACK_ARRAY_FUNC(type)                                                \
  __global__ void UnpackArray##type(                                           \
      unsigned char* input, unsigned char* meta_info, float* output,           \
      int num_elems, int bucket_size, int bits, void* ctx) {                   \
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;                  \
    unsigned int stride = gridDim.x * blockDim.x;                              \
    int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;             \
    unsigned int divisor = 1 << bits;                                          \
    for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;    \
         i += stride) {                                                        \
      uint64_t value = 0;                                                      \
      for (int j = 0; j < bits && i * bits + j < num_char; j++) {              \
        value |= ((uint64_t)input[i * bits + j]) << (j * PACK_SIZE);           \
      }                                                                        \
      _Pragma("unroll 4") for (int j = 0;                                      \
                               j < PACK_SIZE && i * PACK_SIZE + j < num_elems; \
                               j++) {                                          \
        unsigned char encoded_value = (value >> (j * bits)) & (divisor - 1);   \
        output[i * PACK_SIZE + j] =                                            \
            type##DecodeValue(encoded_value, meta_info, i * PACK_SIZE + j,     \
                              bucket_size, bits, ctx);                         \
      }                                                                        \
    }                                                                          \
  }

PACK_ARRAY_FUNC(MaxMin)
UNPACK_ARRAY_FUNC(MaxMin)
PACK_ARRAY_FUNC(LinfNorm)
UNPACK_ARRAY_FUNC(LinfNorm)
// PACK_ARRAY_FUNC(L2Norm)
// UNPACK_ARRAY_FUNC(L2Norm)

__global__ void PackArrayL2Norm(float* input, unsigned char* meta_info,
                                unsigned char* output, float* feedback,
                                int num_elems, int bucket_size, int bits,
                                void* ctx, CurandState* states) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  float* feedback_ = nullptr;
  CurandState local_state = states[tid];
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  QuanCtx qctx = {ctx, num_buckets};
  float rand;
  uint64_t encoded;
  uint64_t value = 0;
  int idx;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    value = 0;
    for (unsigned int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems;
         j++) {
      idx = i * PACK_SIZE + j;
      if (feedback)
        feedback_ = feedback + idx;
      rand = GetRand(&local_state);
      encoded = L2NormEncodeValue(input[idx], feedback_, meta_info, idx,
                                  bucket_size, bits, rand, &qctx);
      value += (encoded << (j * bits));
    }
    for (unsigned int j = 0; j < bits && i * bits + j < num_char; j++) {
      output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;
    }
  }
  states[tid] = local_state;
}

__global__ void UnpackArrayL2Norm(unsigned char* input,
                                  unsigned char* meta_info, float* output,
                                  int num_elems, int bucket_size, int bits,
                                  void* ctx) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  unsigned int divisor = 1 << bits;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  QuanCtx qctx = {ctx, num_buckets};
  unsigned char encoded_value;
  uint64_t value;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    value = 0;
    for (int j = 0; j < bits && i * bits + j < num_char; j++) {
      value |= ((uint64_t)input[i * bits + j]) << (j * PACK_SIZE);
    }
    _Pragma("unroll 4") for (int j = 0;
                             j < PACK_SIZE && i * PACK_SIZE + j < num_elems;
                             j++) {
      encoded_value = (value >> (j * bits)) & (divisor - 1);
      output[i * PACK_SIZE + j] =
          L2NormDecodeValue(encoded_value, meta_info, i * PACK_SIZE + j,
                            bucket_size, bits, (void*)&qctx);
    }
  }
}

/*
 * Quantization handles.
 */
__global__ void _quantize_maxmin(unsigned char* y, const float* x,
                                 const float* maxandmin, float* feedback,
                                 const int n, int bits, int bucket_size,
                                 CurandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  CurandState local_state;
  local_state = states[index];

  int parts = 8 / bits;
  int divisor = 1 << bits;

  for (int i = index; i < (n + parts - 1) / parts; i += stride) {
    int a = 0;
    for (int j = 0; j < parts && i * parts + j < n; j++) {
      int my_bucket = (i * parts + j) / bucket_size;
      float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) /
                   (divisor - 1);
      float d =
          (x[i * parts + j] - maxandmin[my_bucket * 2 + 1]) / unit
          //               + (curand(&local_state) % 100001) / 100000.0;
                                           + curand_uniform(&local_state);
//          + HybridTaus(&local_state);
      int level = (int)floor(d);
      a += level << (j * bits);
      if (feedback) {
        feedback[i * parts + j] =
            x[i * parts + j] - (maxandmin[2 * my_bucket + 1] + level * unit);
      }
    }
    y[i] = (unsigned char)a;
  }
  states[index] = local_state;
}

__global__ void _dequantize_maxmin(const unsigned char* y,
                                   const float* maxandmin, float* x, int n,
                                   int bits, int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  int parts = 8 / bits;
  int divisor = 1 << bits;

  for (int i = index; i < n; i += stride) {
    int my_bucket = i / bucket_size;
    float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) /
                 (divisor - 1);
    x[i] = maxandmin[my_bucket * 2 + 1] +
           ((y[i / parts] >> ((i % parts) * bits)) & (divisor - 1)) * unit;
  }
}

// Normalize each value and encode it with quantization points.
// Encoding is performed with bits - 1, levels must be prepared in advance.
// One bit per encoded value is used for signs.
__global__ void _quantize_Linf_normalized(unsigned char* y, const float* x,
                                          const float* norm, float* feedback,
                                          const float* levels, int n, int bits,
                                          int bucket_size,
                                          CurandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  CurandState local_state;
  local_state = states[index];
  int parts = 8 / bits;
  unsigned char num_levels = 1 << (bits - 1);
  for (int i = index; i < (n + parts - 1) / parts; i += stride) {
    int a = 0;
    for (int j = 0; j < parts && i * parts + j < n; j++) {
      int my_bucket = (i * parts + j) / bucket_size;
      float rand =
                    curand_uniform(&local_state);
//          HybridTaus(&local_state);

      float d = x[i * parts + j] / norm[my_bucket];
      char sign = (d < -EPS);
      d = fabsf(d);
      unsigned char level_idx = 0;
      while (level_idx + 1 < num_levels) {
        if (d - levels[level_idx + 1] < EPS) {
          if (d + (levels[level_idx + 1] - levels[level_idx]) * rand -
                  levels[level_idx + 1] >
              EPS) {
            level_idx++;
          }
          break;
        }
        level_idx++;
      }
      if (feedback) {
        float decode_value = norm[my_bucket] * (sign ? -1.0 : 1.0);
        if (bits > 1)
          decode_value *= levels[level_idx];
        feedback[i * parts + j] = x[i * parts + j] - decode_value;
      }
      level_idx |= (sign << (bits - 1));
      a += (level_idx << (j * bits));
    }
    y[i] = (unsigned char)a;
  }
  states[index] = local_state;
}

__global__ void _dequantize_Linf_normalized(const unsigned char* y,
                                            const float* norms,
                                            const float* levels, float* x,
                                            const int n, int bits,
                                            int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  int parts = 8 / bits;
  int divisor = 1 << bits;

  for (int i = index; i < n; i += stride) {
    int my_bucket = i / bucket_size;
    unsigned char encoded_value =
        (y[i / parts] >> ((i % parts) * bits)) & (divisor - 1);
    char sign = (encoded_value & (1 << (bits - 1))) ? -1 : 1;
    encoded_value &= (1 << (bits - 1)) - 1;
    x[i] = norms[my_bucket] * sign;
    if (bits > 1)
      x[i] *= levels[encoded_value];
  }
}

__global__ void _quantize_L2_normalized(unsigned char* y, const float* x,
                                        const float* norm,
                                        const unsigned char* max_logs,
                                        float* feedback, const float* levels,
                                        int n, int bits, int bucket_size,
                                        CurandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  CurandState local_state;
  local_state = states[index];
  int parts = 8 / bits;
  unsigned char num_levels = 1 << (bits - 1);
  for (int i = index; i < (n + parts - 1) / parts; i += stride) {
    int a = 0;
    for (int j = 0; j < parts && i * parts + j < n; j++) {
      int my_bucket = (i * parts + j) / bucket_size;
      float rand =
                        curand_uniform(&local_state);
//          HybridTaus(&local_state);
      float d = x[i * parts + j] / norm[my_bucket];
      char sign = (d < -EPS);
      d = fabsf(d);
      unsigned char level_idx = 0;
      unsigned char offset = max_logs[my_bucket] + num_levels - 1;

      while (level_idx + 1 < num_levels) {
        if (d - levels[offset - (level_idx + 1)] < EPS) {
          if (d +
                  (levels[offset - (level_idx + 1)] -
                   levels[offset - level_idx]) *
                      rand -
                  levels[offset - (level_idx + 1)] >
              EPS) {
            level_idx++;
          }
          break;
        }
        level_idx++;
      }
      if (feedback) {
        float decode_value = norm[my_bucket] * (sign ? -1.0 : 1.0);
        if (bits > 1)
          decode_value *= levels[level_idx];
        feedback[i * parts + j] = x[i * parts + j] - decode_value;
      }
      level_idx |= (sign << (bits - 1));
      a += (level_idx << (j * bits));
    }
    y[i] = (unsigned char)a;
  }
  states[index] = local_state;
}

__global__ void _dequantize_L2_normalized(const unsigned char* y,
                                          const float* norms,
                                          const unsigned char* max_logs,
                                          const float* levels, float* x, int n,
                                          int bits, int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  int parts = 8 / bits;
  int divisor = 1 << bits;
  int num_levels = 1 << (bits - 1);
  for (int i = index; i < n; i += stride) {
    int my_bucket = i / bucket_size;
    unsigned char offset = num_levels + max_logs[my_bucket] - 1;
    unsigned char encoded_value =
        (y[i / parts] >> ((i % parts) * bits)) & (divisor - 1);
    char sign = (encoded_value & (1 << (bits - 1))) ? -1 : 1;
    encoded_value &= (1 << (bits - 1)) - 1;
    float qp = (encoded_value == 0) ? 0.0 : levels[offset - encoded_value];
    x[i] = qp * norms[my_bucket] * sign;
  }
}

void CUDA_quantize_maxmin(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback_data, int num_elems, int bits,
                          int bucket_size, CurandState* states,
                          cudaStream_t stream) {
  float* input = (float*)input_data;
  unsigned char* meta_info = output_data;
  float* feedback = (float*)feedback_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* output = output_data + 2 * sizeof(float) * num_buckets;
  MaxMin_find_meta<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
                     stream>>>(input, meta_info, num_elems, bucket_size);
  CUDA_CHECK(cudaGetLastError());
  _quantize_maxmin<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
      stream>>>(output, input, (float*)meta_info, feedback, num_elems, bits,
                   bucket_size, states);
//  PackArrayMaxMin<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
//                    stream>>>(input, meta_info, output, feedback, num_elems,
//                              bucket_size, bits, NULL, states);
  CUDA_CHECK(cudaGetLastError());
}

void CUDA_dequantize_maxmin(unsigned char* input_data,
                            unsigned char* output_data, int num_elems, int bits,
                            int bucket_size, cudaStream_t stream) {
  float* output = (float*)output_data;
  unsigned char* meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* input = input_data + 2 * sizeof(float) * num_buckets;
  _dequantize_maxmin<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
      stream>>>(input, (float*)meta_info, output,num_elems, bits,bucket_size);
//  UnpackArrayMaxMin<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
//                      stream>>>(input, meta_info, output, num_elems,
//                                bucket_size, bits, NULL);
  CUDA_CHECK(cudaGetLastError());
}

void CUDA_quantize_LinfNorm(unsigned char* input_data,
                            unsigned char* output_data, unsigned char* feedback,
                            float* levels, int num_elems, int bits,
                            int bucket_size, CurandState* states,
                            cudaStream_t stream) {
  float* input = (float*)input_data;
  unsigned char* meta_info = output_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* output = output_data + sizeof(float) * num_buckets;

  LinfNorm_find_meta<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
                       stream>>>(input, meta_info, num_elems, bucket_size);
  CUDA_CHECK(cudaGetLastError());
  PackArrayLinfNorm<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
                      stream>>>(input, meta_info, output, (float*)feedback,
                                num_elems, bucket_size, bits, (void*)levels,
                                states);
  CUDA_CHECK(cudaGetLastError());
}

void CUDA_dequantize_LinfNorm(unsigned char* input_data,
                              unsigned char* output_data, float* levels,
                              int num_elems, int bits, int bucket_size,
                              cudaStream_t stream) {
  float* output = (float*)output_data;
  unsigned char* meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* input = input_data + sizeof(float) * num_buckets;
  UnpackArrayLinfNorm<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
                        stream>>>(input, meta_info, output, num_elems,
                                  bucket_size, bits, (void*)levels);
  CUDA_CHECK(cudaGetLastError());
}

void CUDA_quantize_L2Norm(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback, float* levels, int num_elems,
                          int bits, int bucket_size, CurandState* states,
                          cudaStream_t stream) {
  float* input = (float*)input_data;
  unsigned char* meta_info = output_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* output =
      output_data + (sizeof(float) + sizeof(char)) * num_buckets;
  L2Norm_find_meta<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
                     stream>>>(input, meta_info, num_elems, bucket_size);
  CUDA_CHECK(cudaGetLastError());
  PackArrayL2Norm<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
                    stream>>>(input, meta_info, output, (float*)feedback,
                              num_elems, bucket_size, bits, (void*)levels,
                              states);
  CUDA_CHECK(cudaGetLastError());
}

void CUDA_dequantize_L2Norm(unsigned char* input_data,
                            unsigned char* output_data, float* levels,
                            int num_elems, int bits, int bucket_size,
                            cudaStream_t stream) {
  float* output = (float*)output_data;
  unsigned char* meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* input =
      input_data + (sizeof(float) + sizeof(char)) * num_buckets;
  UnpackArrayL2Norm<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,
                      stream>>>(input, meta_info, output, num_elems,
                                bucket_size, bits, (void*)levels);
  CUDA_CHECK(cudaGetLastError());
}
