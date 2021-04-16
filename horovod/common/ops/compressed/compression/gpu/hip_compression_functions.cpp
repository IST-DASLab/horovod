#include "hip_compression_functions.h"
#include "fp16_util.h"

namespace horovod {
namespace common {
namespace gpu {
using uint64_t = unsigned long long int;

//-------------------- Quantization Functions ----------------------------------
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

template <typename T, CompressFunc method, bool EF, int BITS>
inline __device__ unsigned char
EncodeValue(T input, T* feedback, unsigned char* meta_info, T rand, void* ctx) {

  switch (method) {
  case CompressFunc::MaxMinWide:
    return MaxMinEncodeValue<T, EF, BITS>(input, feedback, meta_info, rand);
  case CompressFunc::NormWide:
  case CompressFunc::NormPos:
    printf("Not supported type\n") return 0;
  default:
    printf("Wrong compression type\n");
    return 0;
  }
}

template <typename T, CompressFunc FUNC, NormType NORM, int BITS>
__device__ void find_meta_parallel(T* input, unsigned char* meta,
                                   int num_elems) {
  unsigned int tid = hipThreadIdx_x;
  unsigned int block_size = hipBlockIdx_x;
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

template <typename T, CompressFunc FUNC, bool EF, int BITS>
__device__ void CompressBucket(T* input, unsigned char* output,
                               T* feedback_data, unsigned char* meta_info,
                               int num_elems, GPURandState* state, void* ctx) {
  unsigned int tid = hipThreadIdx_x;
  unsigned int num_threads = hipBlockDim_x;
  float rand;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  T* feedback_ = nullptr;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += num_threads) {
    uint64_t value = 0;
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

template <typename T, CompressFunc FUNC, NormType NORM, bool EF, int BITS>
__global__ void quantize(unsigned char* input_data, unsigned char* output_data,
                         unsigned char* feedback_data, int num_elems,
                         int bucket_size, GPURandState* states, void* ctx) {
  unsigned num_blocks = hipGridDim_x;
  unsigned int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  ;
  unsigned int bid = hipBlockIdx_x;
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

template <typename T, CompressFunc FUNC, NormType NORM, bool EF>
inline void QUANTIZE2(unsigned char* input_data, unsigned char* output_data,
                      unsigned char* feedback_data, int num_elems, int bits,
                      int bucket_size, GPURandState* states, hipStream_t stream,
                      void* ctx, int num_blocks, int num_threads,
                      int shared_memory_block_size) {
  switch (bits) {
  case 1:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize<T, FUNC, NORM, EF, 1>), dim3(num_blocks),
        dim3(num_threads), shared_memory_block_size, stream, input_data,
        output_data, feedback_data, num_elems, bucket_size, states, ctx);
    break;
  case 2:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize<T, FUNC, NORM, EF, 2>), dim3(num_blocks),
        dim3(num_threads), shared_memory_block_size, stream, input_data,
        output_data, feedback_data, num_elems, bucket_size, states, ctx);
    break;
  case 3:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize<T, FUNC, NORM, EF, 3>), dim3(num_blocks),
        dim3(num_threads), shared_memory_block_size, stream, input_data,
        output_data, feedback_data, num_elems, bucket_size, states, ctx);
    break;
  case 4:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize<T, FUNC, NORM, EF, 4>), dim3(num_blocks),
        dim3(num_threads), shared_memory_block_size, stream, input_data,
        output_data, feedback_data, num_elems, bucket_size, states, ctx);
    break;
  case 5:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize<T, FUNC, NORM, EF, 5>), dim3(num_blocks),
        dim3(num_threads), shared_memory_block_size, stream, input_data,
        output_data, feedback_data, num_elems, bucket_size, states, ctx);
    break;
  case 6:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize<T, FUNC, NORM, EF, 6>), dim3(num_blocks),
        dim3(num_threads), shared_memory_block_size, stream, input_data,
        output_data, feedback_data, num_elems, bucket_size, states, ctx);
    break;
  case 7:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(quantize<T, FUNC, NORM, EF, 7>), dim3(num_blocks),
        dim3(num_threads), shared_memory_block_size, stream, input_data,
        output_data, feedback_data, num_elems, bucket_size, states, ctx);
    break;
  case 8 hipLaunchKernelGGL(
      HIP_KERNEL_NAME(quantize<T, FUNC, NORM, EF, 8>), dim3(num_blocks),
      dim3(num_threads), shared_memory_block_size, stream, input_data,
      output_data, feedback_data, num_elems, bucket_size, states, ctx) break;
      default:
    printf("Wrong number of bits %i!!!\n", bits);
  }

  HIP_CHECK(hipGetLastError());
}

template <typename T, CompressFunc FUNC, NormType NORM>
inline void QUANTIZE1(unsigned char* input_data, unsigned char* output_data,
                      unsigned char* feedback_data, int num_elems, int bits,
                      int bucket_size, GPURandState* states, hipStream_t stream,
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
void HIP_quantize_maxmin(unsigned char* input_data, unsigned char* output_data,
                         unsigned char* feedback_data, int num_elems, int bits,
                         int bucket_size, GPURandState* states,
                         hipStream_t stream) {
  int num_blocks =
      umin((num_elems + bucket_size - 1) / bucket_size, MAX_NUMBER_OF_BLOCKS);
  int num_threads = umin(THREADS_PER_BLOCK_COMPRESS, bucket_size);
  int shared_memory_block_size = 2 * MAX_THREADS_PER_BLOCK * sizeof(T);
  QUANTIZE1<T, CompressFunc::MaxMinWide, NormType::Linf>(
      input_data, output_data, feedback_data, num_elems, bits, bucket_size,
      states, stream, nullptr, num_blocks, num_threads,
      shared_memory_block_size);
}
//-------------------- Dequantization Functions --------------------------------

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

template <typename T, CompressFunc method, int BITS>
inline __device__ T DecodeValue(unsigned char input, unsigned char* meta_info,
                                unsigned int idx, int bucket_size, void* ctx) {
  switch (method) {
  case CompressFunc::MaxMinWide:
    return MaxMinDecodeValue<T>(input, meta_info, idx, bucket_size);
  case CompressFunc::NormWide:
  case CompressFunc::NormPos:
    printf("Not supported type\n");
    return 0;
  default:
    printf("Wrong compression type\n");
    return 0;
  }
}

template <typename T, CompressFunc FUNC, bool ADD, int BITS>
__global__ void UnpackArray(unsigned char* input, unsigned char* meta_info,
                            T* output, int num_elems, int bucket_size,
                            void* ctx) {
  unsigned int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  unsigned int stride = hipGridDim_x * hipBlockDim_x;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  const unsigned int divisor = 1 << BITS;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    uint64_t value = 0;
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

template <typename T, CompressFunc FUNC, bool ADD>
inline void DEQUANTIZE(unsigned char* input, unsigned char* meta_info,
                       T* output, int num_elems, int bucket_size, int bits,
                       hipStream_t stream, void* ctx, int num_blocks,
                       int num_threads) {
  switch (bits) {
  case 1:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(UnpackArray<T, FUNC, ADD, 1>),
                       dim3(num_blocks), dim3(num_threads), 0, stream, input,
                       meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 2:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(UnpackArray<T, FUNC, ADD, 2>),
                       dim3(num_blocks), dim3(num_threads), 0, stream, input,
                       meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 3:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(UnpackArray<T, FUNC, ADD, 3>),
                       dim3(num_blocks), dim3(num_threads), 0, stream, input,
                       meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 4:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(UnpackArray<T, FUNC, ADD, 4>),
                       dim3(num_blocks), dim3(num_threads), 0, stream, input,
                       meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 5:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(UnpackArray<T, FUNC, ADD, 5>),
                       dim3(num_blocks), dim3(num_threads), 0, stream, input,
                       meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 6:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(UnpackArray<T, FUNC, ADD, 6>),
                       dim3(num_blocks), dim3(num_threads), 0, stream, input,
                       meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 7:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(UnpackArray<T, FUNC, ADD, 7>),
                       dim3(num_blocks), dim3(num_threads), 0, stream, input,
                       meta_info, output, num_elems, bucket_size, ctx);
    break;
  case 8:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(UnpackArray<T, FUNC, ADD, 8>),
                       dim3(num_blocks), dim3(num_threads), 0, stream, input,
                       meta_info, output, num_elems, bucket_size, ctx);
    break;
  default:
    printf("Wrong number of bits %i!!!\n", bits);
  }
  HIP_CHECK(hipGetLastError());
}

template <typename T, bool ADD>
void HIP_dequantize_maxmin(unsigned char* input_data,
                           unsigned char* output_data, int num_elems, int bits,
                           int bucket_size, hipStream_t stream) {
  T* output = (T*)output_data;
  unsigned char* meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* input = input_data + 2 * sizeof(T) * num_buckets;
  int num_threads = THREADS_PER_BLOCK_DECOMPRESS;
  int num_blocks = BLOCKS_PER_GRID(num_elems / PACK_SIZE, num_threads);
  DEQUANTIZE<T, CompressFunc::MaxMinWide, ADD>(
      input, meta_info, output, num_elems, bucket_size, bits, stream, NULL,
      num_blocks, num_threads);
}

//----------------------- Utilite functions --------------------------------
__global__ void float2half(float* input, __half* output, int numel) {
  int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int stride = hipBlockDim_x * hipGridDim_x;
  for (int i = index; i < numel; i += stride) {
    output[i] = __float2half(input[i]);
  }
}

void HIP_convert_to_halves(float* input, Half* output, int numel,
                           hipStream_t stream) {
  hipLaunchKernelGGL(float2half, dim3(1), dim3(numel), 0, stream, input, output,
                     numel);
  HIP_CHECK(hipGetLastError());
}

template <typename T>
__global__ void _add(int64_t n, const T* x, const T* y, T* sum) {
  int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int stride = hipBlockDim_x * hipGridDim_x;
  for (int i = index; i < n; i += stride) {
    sum[i] = x[i] + y[i];
  }
}

template <>
__global__ void _add<__half>(int64_t n, const __half* x, const __half* y,
                             __half* sum) {
  int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int stride = hipBlockDim_x * hipGridDim_x;
  for (int i = index; i < n; i += stride) {
    sum[i] = __hadd(x[i], y[i]);
  }
}

template <typename T>
void HIP_add(int n, const T* x, T* y, T* sum, hipStream_t stream) {
  int num_threads = min(n, MAX_THREADS_PER_BLOCK);
  int blocks = BLOCKS_PER_GRID(n, num_threads);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(_add<T>), dim3(blocks), dim3(num_threads),
                     0, stream, n, x, y, sum);
  HIP_CHECK(hipGetLastError());
}

//-------------------- Functions instantiations --------------------------------

template void HIP_quantize_maxmin<float>(unsigned char* input_data,
                                         unsigned char* output_data,
                                         unsigned char* feedback_data,
                                         int num_elems, int bits,
                                         int bucket_size, GPURandState* states,
                                         hipStream_t stream);
template void HIP_quantize_maxmin<Half>(unsigned char* input_data,
                                        unsigned char* output_data,
                                        unsigned char* feedback_data,
                                        int num_elems, int bits,
                                        int bucket_size, GPURandState* states,
                                        hipStream_t stream);

template void HIP_dequantize_maxmin<float, true>(unsigned char* input_data,
                                                 unsigned char* output_data,
                                                 int num_elems, int bits,
                                                 int bucket_size,
                                                 hipStream_t stream);
template void HIP_dequantize_maxmin<float, false>(unsigned char* input_data,
                                                  unsigned char* output_data,
                                                  int num_elems, int bits,
                                                  int bucket_size,
                                                  hipStream_t stream);

template void HIP_dequantize_maxmin<Half, true>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                int num_elems, int bits,
                                                int bucket_size,
                                                hipStream_t stream);
template void HIP_dequantize_maxmin<Half, false>(unsigned char* input_data,
                                                 unsigned char* output_data,
                                                 int num_elems, int bits,
                                                 int bucket_size,
                                                 hipStream_t stream);

template void HIP_add<float>(int n, const float* x, float* y, float* sum,
                             hipStream_t stream);
template void HIP_add<Half>(int n, const Half* x, Half* y, Half* sum,
                            hipStream_t stream);

} // namespace gpu
} // namespace common
} // namespace horovod
