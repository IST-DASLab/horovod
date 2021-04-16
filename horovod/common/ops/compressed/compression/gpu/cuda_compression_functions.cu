#include "cuda_compression_functions.h"
#include "cuda_rand.h"
#include "fp16_util.h"

namespace horovod {
namespace common {
namespace gpu {
using uint64_t = unsigned long long int;

typedef union {
  float4 vec;
  float a[4];
} F4;

typedef union {
  uchar2 vec;
  unsigned char a[2];
} U2;

typedef union {
  uchar3 vec;
  unsigned char a[3];
} U3;

typedef union {
  uchar4 vec;
  unsigned char a[4];
} U4;

// Single value quantization functions
template <typename T, bool EF, int BITS>
inline __device__ unsigned char
MaxMinEncodeValue(T input, T* feedback, unsigned char* meta_info, T rand) {
  T* maxmin = ((T*)meta_info);
  float min = type2float(maxmin[1]);
  float unit = type2float(maxmin[0]);
  if (unit < EPS) {
    return (1 << BITS) - 1;
  }
  float input_f = type2float(input);
  float d = ((input_f - min) / unit) + type2float(rand);
  unsigned char level = floor(d);
  if (EF)
    *feedback = float2type<T>(input_f - (min + unit * level));
  return level;
//  T min = maxmin[1];
//  T unit = maxmin[0];
//  if (lt(unit, (T)EPS)) {
//    return (1 << BITS) - 1;
//  }
//  T d = add(div(sub(input, min), unit), rand);
//  unsigned char level = floor(d);
//  if (EF)
//    *feedback = sub(input, add(min, mul_int(unit, (int)level)));
//  return level;
}

template <typename T>
inline __device__ T MaxMinDecodeValue(unsigned char input,
                                      unsigned char* meta_info,
                                      unsigned int idx, int bucket_size) {
  int bucket_no = idx / bucket_size;
  T* maxmin = ((T*)meta_info) + 2 * bucket_no;
//  float min = type2float(maxmin[1]);
//  float unit = type2float(maxmin[0]);
//  return float2type<T>(min + unit * input);
  T min = maxmin[1];
  T unit = maxmin[0];
  return add(min, mul_int(unit, (int)input));
}

template <typename T, bool EF, int BITS>
inline __device__ unsigned char NormPosEncodeValue(T input, T* feedback,
                                                   unsigned char* meta_info,
                                                   T rand, void* ctx) {
  T norm = ((T*)meta_info)[0];
  float norm_f = type2float(norm);
  char sign;
  const int num_levels = 1 << (BITS - 1);
  float* levels = (float*)ctx;
  float input_f = type2float(input);
  float d = fabsf(input_f / norm_f);
  sign = (input_f < -EPS);
  if (norm_f < EPS) {
    return 0;
  }
  unsigned char level_idx = 0;
  float diff;
  float rand_f = type2float(rand);
  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
  while (level_idx + 1 < num_levels) {
    if (EPS < d - levels[level_idx + 1]) {
      diff = (levels[level_idx] - levels[level_idx + 1]) * rand_f;
      if (d + diff - levels[level_idx] < -EPS) {
        level_idx++;
      }
      break;
    }
    level_idx++;
  }
  // update error feedback
  if (EF) {
    float recovered_v = norm_f * (sign ? -1.0 : 1.0);
    if (BITS > 1)
      recovered_v = recovered_v * levels[level_idx];
    *feedback = float2type<T>(input_f - recovered_v);
  }
//  T norm = ((T*)meta_info)[0];
//  char sign;
//  const int num_levels = 1 << (BITS - 1);
//  T* levels = (T*)ctx;
//  T d = abs(div(input, norm));
//  sign = lt(input, (T)-EPS);
//  if (lt(norm, (T)EPS)) {
//    return sign;
//  }
//  unsigned char level_idx = 0;
//  T diff;
//  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
//  while (level_idx + 1 < num_levels) {
//    if (lt((T)EPS, sub(d, levels[level_idx + 1]))) {
//      diff = mul(sub(levels[level_idx], levels[level_idx + 1]), rand);
//      if (lt(sub(add(d, diff), levels[level_idx]), (T)-EPS)) {
//        level_idx++;
//      }
//      break;
//    }
//    level_idx++;
//  }
//  // update error feedback
//  if (EF) {
//    T recovered_v = mul_float(norm, sign ? -1.0 : 1.0);
//    if (BITS > 1)
//      recovered_v = mul(recovered_v, levels[level_idx]);
//    *feedback = sub(input, recovered_v);
//  }
  level_idx |= (sign << (BITS - 1));
  return level_idx;
}

template <typename T, int BITS>
inline __device__ T NormPosDecodeValue(unsigned char input,
                                       unsigned char* meta_info,
                                       unsigned int idx, int bucket_size,
                                       void* ctx) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int bucket_no = idx / bucket_size;
  T norm = ((T*)meta_info)[bucket_no];
  float norm_f = type2float(norm);
//  if (lt(norm, (T) EPS)) {
//    return mul_int(norm, input ? 1: -1);
//  }
//  T* levels = (T*)ctx;
//  const int num_levels = 1 << (BITS - 1);
//  char sign = (input & num_levels) ? -1 : 1;
//  input &= (num_levels - 1);
//  T decode_value = mul_int(norm, (int)sign);
//
//  if (BITS > 1) {
//    decode_value = mul(decode_value, levels[input]);
//  }
  if (norm_f < EPS) {
    return float2type<T>(norm_f * input? -1: 1);
  }
  float* levels = (float*)ctx;
  const int num_levels = 1 << (BITS - 1);
  char sign = (input & num_levels) ? -1 : 1;
  input &= (num_levels - 1);
  float decode_value = norm_f * sign;
  if (BITS > 1) {
    decode_value = decode_value * levels[input];
  }
  return float2type<T>(decode_value);
}

template <typename T, bool EF, int BITS>
inline __device__ unsigned char NormWideEncodeValue(T input, T* feedback,
                                                    unsigned char* meta_info,
                                                    T rand, void* ctx) {
  T norm = ((T*)meta_info)[0];
  const int num_levels = 1 << BITS;
  T* levels = (T*)ctx;
  T d = div(input, norm);
  T diff;
  unsigned char level_idx = 0;
  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
  while (level_idx + 1 < num_levels) {
    if (lt((T)EPS, sub(d, levels[level_idx + 1]))) {
      diff = mul(sub(levels[level_idx], levels[level_idx + 1]), rand);
      if (lt(sub(add(d, diff), levels[level_idx]), (T)-EPS)) {
        level_idx++;
      }
      break;
    }
    level_idx++;
  }
  // update error feedback
  if (EF) {
    T recovered_v = norm;
    if (BITS > 1)
      recovered_v = mul(recovered_v, levels[level_idx]);
    *feedback = sub(input, recovered_v);
  }
  return level_idx;
}

template <typename T, int BITS>
inline __device__ T NormWideDecodeValue(unsigned char input,
                                        unsigned char* meta_info,
                                        unsigned int idx, int bucket_size,
                                        void* ctx) {
  int bucket_no = idx / bucket_size;
  T norm = ((T*)meta_info)[bucket_no];
  T* levels = (T*)ctx;
  T decode_value = norm;
  if (BITS > 1) {
    decode_value = mul(decode_value, levels[input]);
  }
  return decode_value;
}

template <typename T, CompressFunc method, bool EF, int BITS>
inline __device__ unsigned char
EncodeValue(T input, T* feedback, unsigned char* meta_info, T rand, void* ctx) {

  switch (method) {
  case CompressFunc::MaxMinWide:
    return MaxMinEncodeValue<T, EF, BITS>(input, feedback, meta_info, rand);
  case CompressFunc::NormWide:
    return NormWideEncodeValue<T, EF, BITS>(input, feedback, meta_info, rand,
                                            ctx);
  case CompressFunc::NormPos:
    return NormPosEncodeValue<T, EF, BITS>(input, feedback, meta_info, rand,
                                           ctx);
  default:
    printf("Wrong compression type\n");
    return 0;
  }
}

template <typename T, CompressFunc method, int BITS>
inline __device__ T DecodeValue(unsigned char input, unsigned char* meta_info,
                                unsigned int idx, int bucket_size, void* ctx) {
  switch (method) {
  case CompressFunc::MaxMinWide:
    return MaxMinDecodeValue<T>(input, meta_info, idx, bucket_size);
  case CompressFunc::NormWide:
    return NormWideDecodeValue<T, BITS>(input, meta_info, idx, bucket_size,
                                        ctx);
  case CompressFunc::NormPos:
    return NormPosDecodeValue<T, BITS>(input, meta_info, idx, bucket_size, ctx);
  default:
    printf("Wrong compression type\n");
    return 0;
  }
}

template <typename T, CompressFunc FUNC, NormType NORM, int BITS>
__device__ void find_meta_parallel(T* input, unsigned char* meta,
                                   int num_elems) {
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  T* meta_buf = (T*)meta;
  const bool is_maxmin = FUNC == CompressFunc::MaxMinWide;
  const int shared_size = MAX_THREADS_PER_BLOCK * (is_maxmin ? 2 : 1);
  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* sdata = reinterpret_cast<T*>(my_smem);
  meta_buf[0] = input[0];
  if (is_maxmin)
    meta_buf[1] = input[0];
  unsigned int num_iters_per_bucket = (num_elems + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    unsigned int idx = i * block_size + tid;
    if (idx < num_elems) {
      if (is_maxmin) {
        sdata[tid] = input[idx];
        sdata[block_size + tid] = input[idx];
      } else {
        if (NORM == NormType::L2)
          sdata[tid] = mul(input[idx], input[idx]);
        else
          sdata[tid] = abs(input[idx]);
      }
    }
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s && idx + s < num_elems) {
        if (is_maxmin) {
          sdata[tid] = max(sdata[tid + s], sdata[tid]);
          sdata[block_size + tid] =
              min(sdata[block_size + tid + s], sdata[block_size + tid]);
        } else {
          if (NORM == NormType::Linf)
            sdata[tid] = max(sdata[tid + s], sdata[tid]);
          else
            sdata[tid] = add(sdata[tid + s], sdata[tid]);
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      if (is_maxmin) {
        meta_buf[0] = max(meta_buf[0], sdata[tid]);
        meta_buf[1] = min(meta_buf[1], sdata[block_size + tid]);
      } else {
        if (NORM == NormType::Linf) {
          meta_buf[0] = max(sdata[tid], meta_buf[0]);
        } else {
          meta_buf[0] = add(sdata[tid], meta_buf[0]);
        }
      }
    }
  }
  if (tid == 0) {
    if (is_maxmin) {
      const unsigned int divisor = (1 << BITS) - 1;
      meta_buf[0] = div_int(sub(meta_buf[0], meta_buf[1]), divisor);
    } else {
      if (lt(meta_buf[0], (T)EPS))
        meta_buf[0] = add(meta_buf[0], (T)EPS);
      if (NORM == NormType::L2)
        meta_buf[0] = sqrt(meta_buf[0]);
    }
  }
  __syncthreads();
}

template <int BITS>
inline __device__ void pack_value(const uint64_t value, unsigned char* output,
                                  unsigned int shift = 0) {
#pragma unroll BITS
  for (unsigned int j = 0; j < BITS; j++) {
    output[j] = value >> (PACK_SIZE * j) & 0xFF;
  }
}

template <>
inline __device__ void
pack_value<2>(const uint64_t value, unsigned char* output, unsigned int shift) {
  U2 output2;
#pragma unroll 2
  for (unsigned int j = 0; j < 2; j++) {
    output2.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar2* output_p = reinterpret_cast<uchar2*>(output);
  output_p[0] = output2.vec;
}

template <>
inline __device__ void
pack_value<3>(const uint64_t value, unsigned char* output, unsigned int shift) {
  U3 output3;
#pragma unroll 3
  for (unsigned int j = 0; j < 3; j++) {
    output3.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar3* output_p = reinterpret_cast<uchar3*>(output);
  output_p[0] = output3.vec;
}

template <>
inline __device__ void
pack_value<4>(const uint64_t value, unsigned char* output, unsigned int shift) {
  U4 output4;
#pragma unroll 4
  for (unsigned int j = 0; j < 4; j++) {
    output4.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar4* output_p = reinterpret_cast<uchar4*>(output);
  output_p[0] = output4.vec;
}

template <>
inline __device__ void
pack_value<6>(const uint64_t value, unsigned char* output, unsigned int shift) {
  pack_value<3>(value, output, 0);
  pack_value<3>(value, output + 3, 3);
}

template <>
inline __device__ void
pack_value<8>(const uint64_t value, unsigned char* output, unsigned int shift) {
  pack_value<4>(value, output, 0);
  pack_value<4>(value, output + 4, 4);
}

template <typename T, CompressFunc FUNC, bool EF, int BITS>
__device__ void CompressBucket(T* input, unsigned char* output,
                               T* feedback_data, unsigned char* meta_info,
                               int num_elems, GPURandState* state, void* ctx) {
  unsigned int tid = threadIdx.x;
  unsigned int num_threads = blockDim.x;
  float rand;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  T* feedback_ = nullptr;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += num_threads) {
    uint64_t value = 0;
    if (std::is_same<T, int>::value) {
      F4 input4;
      if (num_elems - i * PACK_SIZE >= PACK_SIZE) {
#pragma unroll PACK_SIZE / 4
        for (unsigned int j = 0; j < PACK_SIZE; j += 4) {
          int idx = i * PACK_SIZE + j;
          input4.vec = (reinterpret_cast<float4*>(input + idx))[0];
#pragma unroll 4
          for (int k = 0; k < 4; k++) {
            rand = GetRand(state);
            if (EF)
              feedback_ = feedback_data + idx + k;
            uint64_t encoded = EncodeValue<T, FUNC, EF, BITS>(
                input4.a[k], feedback_, meta_info, float2type<T>(rand), ctx);
            value += (encoded << ((j + k) * BITS));
          }
        }
      } else {
        for (unsigned int j = 0; j < num_elems - i * PACK_SIZE; j++) {
          int idx = i * PACK_SIZE + j;
          if (EF)
            feedback_ = feedback_data + idx;
          rand = GetRand(state);
          unsigned encoded = EncodeValue<T, FUNC, EF, BITS>(
              input[idx], feedback_, meta_info, rand, ctx);
          value += (encoded << (j * BITS));
        }
      }
      if (num_char - i * BITS < BITS) {
        for (unsigned int j = 0; j < num_char - i * BITS; j++) {
          output[i * BITS + j] = value >> (PACK_SIZE * j) & 0xFF;
        }
      } else {
        pack_value<BITS>(value, output + i * BITS);
      }
    } else {
      for (unsigned int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems;
           j++) {
        int idx = i * PACK_SIZE + j;
        if (EF)
          feedback_ = feedback_data + idx;
        rand = GetRand(state);
        uint64_t encoded = EncodeValue<T, FUNC, EF, BITS>(
            input[idx], feedback_, meta_info, float2type<T>(rand), ctx);
        value += (encoded << (j * BITS));
      }
      for (unsigned int j = 0; j < BITS && i * BITS + j < num_char; j++) {
        output[i * BITS + j] = value >> (PACK_SIZE * j) & 0xFF;
      }
    }
  }
}

template <typename T, CompressFunc FUNC, NormType NORM, bool EF, int BITS>
__global__ void quantize(unsigned char* input_data, unsigned char* output_data,
                         unsigned char* feedback_data, int num_elems,
                         int bucket_size, GPURandState* states, void* ctx) {
  unsigned num_blocks = gridDim.x;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int bid = blockIdx.x;
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned int cur_bucket_size;
  T* meta = (T*)output_data;
  unsigned char* output;
  const int META_MULTIPLIER = (FUNC == CompressFunc::MaxMinWide) ? 2 : 1;
  output = output_data + META_MULTIPLIER * sizeof(T) * num_buckets;

  unsigned int compressed_size =
      (bucket_size * BITS + PACK_SIZE - 1) / PACK_SIZE;

  T* input = (T*)input_data;
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    find_meta_parallel<T, FUNC, NORM, BITS>(
        input + bucket_size * bucket_id,
        (unsigned char*)(meta + META_MULTIPLIER * bucket_id), cur_bucket_size);
  }
  GPURandState local_state = states[tid];
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket<T, FUNC, EF, BITS>(
        input + bucket_size * bucket_id, output + compressed_size * bucket_id,
        (T*)feedback_data, (unsigned char*)(meta + META_MULTIPLIER * bucket_id),
        cur_bucket_size, &local_state, ctx);
  }
  states[tid] = local_state;
}

template <int BITS>
inline __device__ void unpack_value(unsigned char* input, uint64_t& value,
                                    unsigned shift = 0) {
  for (unsigned int j = 0; j < BITS; j++) {
    value |= ((uint64_t)input[j]) << (j * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<2>(unsigned char* input, uint64_t& value,
                                       unsigned int shift) {
  U2 input2;
  input2.vec = reinterpret_cast<uchar2*>(input)[0];
#pragma unroll 3
  for (unsigned int j = 0; j < 2; j++) {
    value |= ((uint64_t)input2.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<3>(unsigned char* input, uint64_t& value,
                                       unsigned int shift) {
  U3 input3;
  input3.vec = reinterpret_cast<uchar3*>(input)[0];
#pragma unroll 3
  for (unsigned int j = 0; j < 3; j++) {
    value |= ((uint64_t)input3.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<4>(unsigned char* input, uint64_t& value,
                                       unsigned int shift) {
  U4 input4;
  input4.vec = reinterpret_cast<uchar4*>(input)[0];
#pragma unroll 4
  for (unsigned int j = 0; j < 4; j++) {
    value |= ((uint64_t)input4.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<6>(unsigned char* input, uint64_t& value,
                                       unsigned int shift) {
  unpack_value<3>(input, value, 0);
  unpack_value<3>(input + 3, value, 3);
}

template <>
inline __device__ void unpack_value<8>(unsigned char* input, uint64_t& value,
                                       unsigned int shift) {
  unpack_value<4>(input, value, 0);
  unpack_value<4>(input + 4, value, 4);
}

template <typename T, CompressFunc FUNC, bool ADD, int BITS>
__global__ void UnpackArray(unsigned char* input, unsigned char* meta_info,
                            T* output, int num_elems, int bucket_size,
                            void* ctx) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  const unsigned int divisor = 1 << BITS;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    uint64_t value = 0;
    if (std::is_same<T, int>::value) {
      if ((i + 1) * BITS > num_char) {
        for (unsigned int j = 0; j < num_char - i * BITS; j++)
          value |= ((uint64_t)input[i * BITS + j]) << (j * PACK_SIZE);
      } else {
        unpack_value<BITS>(input + i * BITS, value);
      }

      if ((i + 1) * PACK_SIZE > num_elems) {
        for (unsigned int j = 0; j < num_elems - i * PACK_SIZE; j++) {
          unsigned char encoded_value = (value >> (j * BITS)) & (divisor - 1);
          T d = DecodeValue<T, FUNC, BITS>(encoded_value, meta_info,
                                           i * PACK_SIZE + j, bucket_size, ctx);
          if (ADD) {
            output[i * PACK_SIZE + j] = add(output[i * PACK_SIZE + j], d);
          } else {
            output[i * PACK_SIZE + j] = d;
          }
        }
      } else {
        F4 output4;
#pragma unroll(PACK_SIZE / 4)
        for (int j = 0; j < PACK_SIZE; j += 4) {
#pragma unroll 4
          for (int k = 0; k < 4; k++) {
            unsigned char encoded_value =
                (value >> ((j + k) * BITS)) & (divisor - 1);
            T d = DecodeValue<T, FUNC, BITS>(encoded_value, meta_info,
                                             i * PACK_SIZE + j + k, bucket_size,
                                             ctx);
            if (ADD) {
              output4.a[k] = add((T)(output4.a[k]), d);
            } else {
              output4.a[k] = d;
            }
          }
          float4* output_p =
              reinterpret_cast<float4*>(&output[i * PACK_SIZE + j]);
          *output_p = output4.vec;
        }
      }
    } else {
      for (int j = 0; j < BITS && i * BITS + j < num_char; j++) {
        value |= ((uint64_t)input[i * BITS + j]) << (j * PACK_SIZE);
      }
      for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
        unsigned char encoded_value = (value >> (j * BITS)) & (divisor - 1);
        T d = DecodeValue<T, FUNC, BITS>(encoded_value, meta_info,
                                         i * PACK_SIZE + j, bucket_size, ctx);
        if (ADD) {
          output[i * PACK_SIZE + j] = add(output[i * PACK_SIZE + j], d);
        } else {
          output[i * PACK_SIZE + j] = d;
        }
      }
    }
  }
}

__global__ void float2half(float* input, __half* output, int numel) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < numel; i += stride) {
    output[i] = __float2half(input[i]);
  }
}

void CUDA_convert_to_halves(float* input, Half* output, int numel, cudaStream_t stream) {
  float2half<<<1, numel, 0, stream>>>(input, output, numel);
}

template <typename T, CompressFunc FUNC, NormType NORM, bool EF>
inline void QUANTIZE2(unsigned char* input_data, unsigned char* output_data,
                      unsigned char* feedback_data, int num_elems, int bits,
                      int bucket_size, GPURandState* states, cudaStream_t stream,
                      void* ctx, int num_blocks, int num_threads,
                      int shared_memory_block_size) {
  switch (bits) {
  case 1:
    quantize<T, FUNC, NORM, EF, 1>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states, ctx);
    break;
  case 2:
    quantize<T, FUNC, NORM, EF, 2>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states, ctx);
    break;
  case 3:
    quantize<T, FUNC, NORM, EF, 3>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states, ctx);
    break;
  case 4:
    quantize<T, FUNC, NORM, EF, 4>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states, ctx);
    break;
  case 5:
    quantize<T, FUNC, NORM, EF, 5>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states, ctx);
    break;
  case 6:
    quantize<T, FUNC, NORM, EF, 6>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states, ctx);
    break;
  case 7:
    quantize<T, FUNC, NORM, EF, 7>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states, ctx);
    break;
  case 8:
    quantize<T, FUNC, NORM, EF, 8>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states, ctx);
    break;
  default:
    printf("Wrong number of bits %i!!!\n", bits);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, CompressFunc FUNC, NormType NORM>
inline void QUANTIZE1(unsigned char* input_data, unsigned char* output_data,
                      unsigned char* feedback_data, int num_elems, int bits,
                      int bucket_size, GPURandState* states, cudaStream_t stream,
                      void* ctx, int num_blocks, int num_threads,
                      int shared_memory_block_size) {
  if (feedback_data == nullptr) {
    QUANTIZE2<T, FUNC, NORM, false>(
        input_data, output_data, feedback_data, num_elems, bits, bucket_size,
        states, stream, ctx, num_blocks, num_threads, shared_memory_block_size);
  } else {
    QUANTIZE2<T, FUNC, NORM, true>(
        input_data, output_data, feedback_data, num_elems, bits, bucket_size,
        states, stream, ctx, num_blocks, num_threads, shared_memory_block_size);
  }
}

template <typename T>
void CUDA_quantize_maxmin(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback_data, int num_elems, int bits,
                          int bucket_size, GPURandState* states,
                          cudaStream_t stream) {
  int num_blocks =
      umin((num_elems + bucket_size - 1) / bucket_size, MAX_NUMBER_OF_BLOCKS);
  int num_threads = umin(THREADS_PER_BLOCK_COMPRESS, bucket_size);
  int shared_memory_block_size = 2 * MAX_THREADS_PER_BLOCK * sizeof(T);
  QUANTIZE1<T, CompressFunc::MaxMinWide, NormType::Linf>(
      input_data, output_data, feedback_data, num_elems, bits, bucket_size,
      states, stream, nullptr, num_blocks, num_threads,
      shared_memory_block_size);
}

template <typename T>
void CUDA_quantize_Norm(unsigned char* input_data, unsigned char* output_data,
                        unsigned char* feedback, T* levels, int num_elems,
                        int bits, int bucket_size, GPURandState* states,
                        NormType norm_type, LevelsType levels_type,
                        cudaStream_t stream) {
  int num_blocks =
      umin((num_elems + bucket_size - 1) / bucket_size, MAX_NUMBER_OF_BLOCKS);
  int num_threads = umin(THREADS_PER_BLOCK_COMPRESS, bucket_size);
  int shared_memory_block_size = MAX_THREADS_PER_BLOCK * sizeof(T);
  if (levels_type == LevelsType::Pos) {
    const CompressFunc func = CompressFunc::NormPos;
    if (norm_type == NormType::Linf) {
      QUANTIZE1<T, func, NormType::Linf>(input_data, output_data, feedback,
                                         num_elems, bits, bucket_size, states,
                                         stream, levels, num_blocks,
                                         num_threads, shared_memory_block_size);
    } else {
      QUANTIZE1<T, func, NormType::L2>(input_data, output_data, feedback,
                                       num_elems, bits, bucket_size, states,
                                       stream, levels, num_blocks, num_threads,
                                       shared_memory_block_size);
    }
  } else {
    const CompressFunc func = CompressFunc::NormWide;
    if (norm_type == NormType::Linf) {
      QUANTIZE1<T, func, NormType::Linf>(input_data, output_data, feedback,
                                         num_elems, bits, bucket_size, states,
                                         stream, levels, num_blocks,
                                         num_threads, shared_memory_block_size);
    } else {
      QUANTIZE1<T, func, NormType::L2>(input_data, output_data, feedback,
                                       num_elems, bits, bucket_size, states,
                                       stream, levels, num_blocks, num_threads,
                                       shared_memory_block_size);
    }
  }
}

template <typename T, CompressFunc FUNC, bool ADD>
inline void DEQUANTIZE(unsigned char* input, unsigned char* meta_info,
                       T* output, int num_elems, int bucket_size, int bits,
                       cudaStream_t stream, void* ctx, int num_blocks,
                       int num_threads) {
  switch (bits) {
  case 1:
    UnpackArray<T, FUNC, ADD, 1><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 2:
    UnpackArray<T, FUNC, ADD, 2><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 3:
    UnpackArray<T, FUNC, ADD, 3><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 4:
    UnpackArray<T, FUNC, ADD, 4><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 5:
    UnpackArray<T, FUNC, ADD, 5><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 6:
    UnpackArray<T, FUNC, ADD, 6><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 7:
    UnpackArray<T, FUNC, ADD, 7><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 8:
    UnpackArray<T, FUNC, ADD, 8><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size, ctx);
    break;
  default:
    printf("Wrong number of bits %i!!!\n", bits);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, bool ADD>
void CUDA_dequantize_maxmin(unsigned char* input_data,
                            unsigned char* output_data, int num_elems, int bits,
                            int bucket_size, cudaStream_t stream) {
  T* output = (T*)output_data;
  unsigned char* meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* input = input_data + 2 * sizeof(T) * num_buckets;
  int num_threads = THREADS_PER_BLOCK_DECOMPRESS;
  int num_blocks = BLOCKS_PER_GRID(num_elems / PACK_SIZE, num_threads);
  DEQUANTIZE<T, CompressFunc::MaxMinWide, ADD>(input, meta_info, output, num_elems,
                                           bucket_size, bits, stream, NULL,
                                           num_blocks, num_threads);
}

template <typename T, bool ADD>
void CUDA_dequantize_Norm(unsigned char* input_data, unsigned char* output_data,
                          T* levels, int num_elems, int bits, int bucket_size,
                          LevelsType levels_type, cudaStream_t stream) {
  T* output = (T*)output_data;
  unsigned char* meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* input = input_data + sizeof(T) * num_buckets;
  int num_threads = THREADS_PER_BLOCK_DECOMPRESS;
  int num_blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  if (levels_type == LevelsType::Wide) {
    DEQUANTIZE<T, CompressFunc::NormWide, ADD>(
        input, meta_info, output, num_elems, bucket_size, bits, stream,
        (void*)levels, num_blocks, num_threads);
  } else {
    DEQUANTIZE<T, CompressFunc::NormPos, ADD>(
        input, meta_info, output, num_elems, bucket_size, bits, stream,
        (void*)levels, num_blocks, num_threads);
  }
  CUDA_CHECK(cudaGetLastError());
}

template void CUDA_quantize_maxmin<float>(unsigned char* input_data,
                                          unsigned char* output_data,
                                          unsigned char* feedback_data,
                                          int num_elems, int bits,
                                          int bucket_size, GPURandState* states,
                                          cudaStream_t stream);
template void CUDA_quantize_maxmin<Half>(unsigned char* input_data,
                                         unsigned char* output_data,
                                         unsigned char* feedback_data,
                                         int num_elems, int bits,
                                         int bucket_size, GPURandState* states,
                                         cudaStream_t stream);

template void CUDA_dequantize_maxmin<float, true>(unsigned char* input_data,
                                                  unsigned char* output_data,
                                                  int num_elems, int bits,
                                                  int bucket_size,
                                                  cudaStream_t stream);
template void CUDA_dequantize_maxmin<float, false>(unsigned char* input_data,
                                                   unsigned char* output_data,
                                                   int num_elems, int bits,
                                                   int bucket_size,
                                                   cudaStream_t stream);

template void CUDA_dequantize_maxmin<Half, true>(unsigned char* input_data,
                                                 unsigned char* output_data,
                                                 int num_elems, int bits,
                                                 int bucket_size,
                                                 cudaStream_t stream);
template void CUDA_dequantize_maxmin<Half, false>(unsigned char* input_data,
                                                  unsigned char* output_data,
                                                  int num_elems, int bits,
                                                  int bucket_size,
                                                  cudaStream_t stream);

template void
CUDA_quantize_Norm<float>(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback, float* levels, int num_elems,
                          int bits, int bucket_size, GPURandState* states,
                          NormType norm_type, LevelsType levels_type,
                          cudaStream_t stream);

template void CUDA_quantize_Norm<Half>(unsigned char* input_data,
                                       unsigned char* output_data,
                                       unsigned char* feedback, Half* levels,
                                       int num_elems, int bits, int bucket_size,
                                       GPURandState* states, NormType norm_type,
                                       LevelsType levels_type,
                                       cudaStream_t stream);
template void CUDA_dequantize_Norm<float, true>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                float* levels, int num_elems,
                                                int bits, int bucket_size,
                                                LevelsType levels_type,
                                                cudaStream_t stream);

template void CUDA_dequantize_Norm<float, false>(unsigned char* input_data,
                                                 unsigned char* output_data,
                                                 float* levels, int num_elems,
                                                 int bits, int bucket_size,
                                                 LevelsType levels_type,
                                                 cudaStream_t stream);

template void CUDA_dequantize_Norm<Half, true>(unsigned char* input_data,
                                               unsigned char* output_data,
                                               Half* levels, int num_elems,
                                               int bits, int bucket_size,
                                               LevelsType levels_type,
                                               cudaStream_t stream);

template void CUDA_dequantize_Norm<Half, false>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                Half* levels, int num_elems,
                                                int bits, int bucket_size,
                                                LevelsType levels_type,
                                                cudaStream_t stream);

} // namespace gpu
} // namespace common
} // namespace horovod

#include "cuda_topk_compression.cu"