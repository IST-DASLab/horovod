#include "cuda_functions.h"
#include <cstdlib>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 1024
#define EPS 1e-6

__device__ int toInt(unsigned char* z) {
  return ((unsigned int)z[0] & 0xFF) << 24 | ((unsigned int)z[1] & 0xFF) << 16 |
         ((unsigned int)z[2] & 0xFF) << 8 | ((unsigned int)z[3] & 0xFF);
}

__global__ void _init_curand(unsigned int seed, CurandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  /* we have to initialize the state */
//    curand_init(seed,  /* the seed can be the same for each core, here we pass
//    the
//                          time in from the CPU */
//                index, /* the sequence number should be different for each
//                core
//                          (unless you want all cores to get the same sequence
//                          of numbers for some reason - use thread id! */
//                0, /* the offset is how much extra we advance in the sequence
//                for
//                      each call, can be 0 */
//                &states[index]);

  unsigned char z[4];
  for (int i = 0; i < 4; i++)
    z[i] = 128 + index % 128;
  states[index] = toInt(z);
}

__global__ void _add(int n, const float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

__global__ void _find_max_and_min_bucket_seq(const float *x, float *maxandmin,
                                             int n, int bucket_size) {
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

__global__ void _find_norms_bucket_seq(const float *x, float *max, float *norm,
                                       int n, const int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
    float bnorm = 0.0;
    float bmax = fabsf(x[i * bucket_size]);
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      bnorm += x[j] * x[j];
      bmax = fmaxf(bmax, fabsf(x[j]));
    }
    max[i] = bmax;
    bnorm = sqrt(bnorm);
    if (fabsf(bnorm) < EPS)
      bnorm += EPS;
    norm[i] = bnorm;
  }
}

__global__ void _find_Linf_bucket_seq(const float* x, float* norms, int n,
                                      const int bucket_size) {
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

__global__ void _find_L2_max_log_bucket_seq(const float* x, float* norm, unsigned char* max_logs,
                                            float rev_multiplier, int n, const int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
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
      unsigned char max_log = 0;
      while (bmax > EPS && bmax * rev_multiplier - 1.0 < EPS) {
          bmax *= rev_multiplier;
          max_log++;
      }
      max_logs[i] = max_log;
  }
}



__device__ void TausStep(unsigned char& z, unsigned int S1, unsigned int S2,
                         int S3, unsigned M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  z = (((z & M) << S3) ^ b);
}

__device__ void LCGStep(unsigned char& z, unsigned A, unsigned C) {
  z = (A * z + C);
}

__device__ float HybridTaus(int* state_p) {
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

__global__ void _quantize_value_bits(unsigned char *y, const float* x,
                                     const float *maxandmin, const int n,
                                     int bits, int bucket_size,
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
      float d = (x[i * parts + j] - maxandmin[my_bucket * 2 + 1]) / unit
                //               + (curand(&local_state) % 100001) / 100000.0;
//                                 + curand_uniform(&local_state);
                + HybridTaus(&local_state);
      a += ((int)floor(d)) << (j * bits);
    }
    y[i] = (unsigned char) a;
  }
  states[index] = local_state;
}

__global__ void _dequantize_value_bits(const unsigned char *y, const float *maxandmin,
                                       float *x, int n, int bits,
                                       int bucket_size) {
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

//__global__ void _quantize_normalized_exp2(unsigned char* y, const float* x,
//                                          const float* norm,
//                                          const float* levels, int n, int bits,
//                                          int bucket_size,
//                                          CurandState* states) {
//  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
//  unsigned int stride = gridDim.x * blockDim.x;
//
//  CurandState local_state;
//  local_state = states[index];
//  int parts = 8 / bits;
//  int divisor = 1 << bits;
//  for (int i = index; i < (n + parts - 1) / parts; i += stride) {
//    int a = 0;
//    for (int j = 0; j < parts && i * parts + j < n; j++) {
//      int my_bucket = (i * parts + j) / bucket_size;
//      float d = x[i * parts + j] / norm[my_bucket]
//                //    + curand_uniform(&local_state);
//                + HybridTaus(&local_state);
//      char sign = (d < -EPS);
//      int unit_idx = (int)floorf(-log2f(fabsf(d)));
//      if (isinf(unit_idx))
//        unit_idx = 0;
//      // If idx of unit overflows max level assign to it.
//      // It may overflow it only because randomness.
//
//      if (unit_idx > levels[1 << (bits - 1) - 1])
//        unit_idx = levels[1 << (bits - 1) - 1];
//      char c = ((char)unit_idx) & (sign << 7);
//      a += c << (j * bits);
//      float d = (y[i * parts + j] - maxandmin[my_bucket * 2 + 1]) / unit a +=
//          ((int)floor(d)) << (j * bits);
//    }
//    x[i] = (unsigned char)a;
//  }
//  states[index] = local_state;
//}

// Normalize each value and encode it with quantization points.
// Encoding is performed with bits - 1, levels must be prepared in advance.
// One bit per encoded value is used for signs.
__global__ void _quantize_Linf_normalized(unsigned char *y, const float* x,
                                          const float* norm, const float* levels,
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
//          curand_uniform(&local_state);
          HybridTaus(&local_state);

      float d = x[i * parts + j] / norm[my_bucket];
      char sign = (d < -EPS);
      d = fabsf(d);
      unsigned char level_idx = 0;
      while (level_idx + 1 < num_levels) {
        if (d - levels[level_idx + 1] < EPS) {
          if (d + (levels[level_idx + 1] - levels[level_idx]) * rand - levels[level_idx + 1] > EPS) {
            level_idx++;
          }
          break;
        }
        level_idx++;
      }
//      if (i * parts + j < 8)
//        printf("%i %f %f ", level_idx, levels[level_idx], levels[level_idx] * norm[my_bucket]);
      level_idx |= (sign << (bits - 1));
      a += (level_idx << (j * bits));
    }
    y[i] = (unsigned char)a;
  }
//  printf("\n");
  states[index] = local_state;
}

__global__ void _dequantize_Linf_normalized(const unsigned char *y, const float *norms,
                                            const float *levels, float *x, const int n,
                                            int bits, int bucket_size) {
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
    x[i] = levels[encoded_value] * norms[my_bucket] * sign;
  }
}

__global__ void _quantize_L2_normalized(unsigned char *y, const float* x,
                                          const float* norm, const unsigned char *max_logs, const float* levels,
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
//              curand_uniform(&local_state);
              HybridTaus(&local_state);
      float d = x[i * parts + j] / norm[my_bucket];
      char sign = (d < -EPS);
      d = fabsf(d);
      unsigned char level_idx = 0;
      unsigned char offset = max_logs[my_bucket] + num_levels - 1;

      while (level_idx + 1 < num_levels) {
        if (d - levels[offset - (level_idx + 1)] < EPS) {
          if (d + (levels[offset - (level_idx + 1)] - levels[offset - level_idx]) * rand - levels[offset - (level_idx + 1)] > EPS) {
            level_idx++;
          }
          break;
        }
        level_idx++;
      }
//      if (i * parts + j < 8)
//        printf("Value: %f, level %f, Appr %f\n", x[i * parts + j], levels[offset - level_idx],
//            levels[offset - level_idx] * norm[my_bucket]);
      level_idx |= (sign << (bits - 1));
//      if (i * parts + j < 8)
//        printf("%f %i ", x[i * parts + j], level_idx);

      a += (level_idx << (j * bits));
    }
    y[i] = (unsigned char)a;
  }
  states[index] = local_state;
}


__global__ void _dequantize_L2_normalized(const unsigned char *y, const float *norms, const unsigned char *max_logs,
                                          const float *levels, float *x, int n,
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
    float qp = (encoded_value == 0) ? 0.0: levels[offset - encoded_value];
    x[i] = qp * norms[my_bucket] * sign;
//    if (i < 8)
//      printf("decoded value: %f encoded value: %i num_level %i level: %f ",
//          x[i], encoded_value, offset - encoded_value, levels[offset - encoded_value]);
  }
}



#define BLOCKS_PER_GRID(n) (n + (MAX_THREADS_PER_BLOCK - 1)) / MAX_THREADS_PER_BLOCK

void CUDA_init_curand(CurandState* states, int num_elems, unsigned int seed,
                      cudaStream_t stream) {
  _init_curand<<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
      seed, states);
}

int CUDA_get_curand_array_size(int num_elems) {
  return BLOCKS_PER_GRID(num_elems) * MAX_THREADS_PER_BLOCK * sizeof(CurandState);
}

void CUDA_add(int n, const float* x, float* y, cudaStream_t stream) {
  _add<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream>>>(n, x, y);
  cudaStreamSynchronize(stream);
}

void CUDA_find_max_and_min_bucket(const float* x, float* maxandmin, int n,
                                  int bucket_size, cudaStream_t stream) {
  _find_max_and_min_bucket_seq<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0,
                                 stream>>>(x, maxandmin, n, bucket_size);
  //  _find_max_and_min<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x,
  //  maxandmin, n);
  cudaStreamSynchronize(stream);
}

void CUDA_find_norms_bucket(const float* x, float* max, float* norm, int n,
                            int bucket_size, cudaStream_t stream) {
  _find_norms_bucket_seq<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream>>>(
          x, max, norm, n, bucket_size);
  cudaStreamSynchronize(stream);
}

void CUDA_find_Linf_bucket(const float* x, float* maxs, int n, int bucket_size,
                           cudaStream_t stream) {
    _find_Linf_bucket_seq << < BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream >> > (
          x, maxs, n, bucket_size);
  cudaStreamSynchronize(stream);
}

void CUDA_find_L2_and_max_log_bucket(const float* x, float* norm, unsigned char* max_log, float rev_multiplier, int n,
                                     int bucket_size, cudaStream_t stream){
  _find_L2_max_log_bucket_seq<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream>>>(
          x, norm, max_log, rev_multiplier, n, bucket_size);
  cudaStreamSynchronize(stream);
}

void CUDA_quantize_value_bits(unsigned char* y, const float* x,
                              const float* maxandmin, int n, int bits,
                              int bucket_size, CurandState* states,
                              cudaStream_t stream) {
  _quantize_value_bits<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream>>>(
          y, x, maxandmin, n, bits, bucket_size, states);
  cudaStreamSynchronize(stream);
}

void CUDA_dequantize_value_bits(const unsigned char* y, const float* maxandmin,
                                float* x, int n, int bits, int bucket_size,
                                cudaStream_t stream) {
  _dequantize_value_bits<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream>>>(
          y, maxandmin, x, n, bits, bucket_size);
  cudaStreamSynchronize(stream);
}

void CUDA_Linf_normalized_quantize_values(unsigned char* y, const float* x,
                                          const float* norms, const float* levels,
                                          int n, int bits, int bucket_size,
                                          CurandState* states, cudaStream_t stream) {
  _quantize_Linf_normalized<<< BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream >> > (
          y, x, norms, levels, n, bits, bucket_size, states);
  cudaStreamSynchronize(stream);
}

void CUDA_L2_normalized_quantize_values(unsigned char* y, const float* x,
                                          const float* norms, const unsigned char *max_logs, const float* levels,
                                          int n, int bits, int bucket_size,
                                          CurandState* states, cudaStream_t stream) {
  _quantize_L2_normalized<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream>>>(
          y, x, norms, max_logs, levels, n, bits, bucket_size, states);
  cudaStreamSynchronize(stream);
}

void CUDA_Linf_normalized_dequantize_values(const unsigned char *y,
                                            const float *norms, const float *levels,
                                            float *x, int n, int bits,
                                            int bucket_size, cudaStream_t stream) {
  _dequantize_Linf_normalized<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream >> > (
      y, norms, levels, x, n, bits, bucket_size);
  cudaStreamSynchronize(stream);
}

void CUDA_L2_normalized_dequantize_values(const unsigned char *y, const float* norms,
                                          const unsigned char *max_logs, const float* levels,
                                          float *x, int n, int bits,
                                          int bucket_size, cudaStream_t stream) {
  _dequantize_L2_normalized<<<BLOCKS_PER_GRID(n), MAX_THREADS_PER_BLOCK, 0, stream>>>(
          y, norms, max_logs, levels, x, n, bits, bucket_size);
  cudaStreamSynchronize(stream);
}
