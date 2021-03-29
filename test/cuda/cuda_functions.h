#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_def.h"
#include "cuda_functions_util.h"
#include "cuda_rand.h"

namespace cfu = cuda_functions_util;

/* UNUSED */

#define FULL_MASK 0xffffffff
template <typename T>
__device__ void maxmin_warp(const T* input, T* maxmin, int n, int bits) {
  unsigned int tid = threadIdx.x % WARP_SIZE;
  int num_per_thread = n / WARP_SIZE;
  T max_val, min_val;
  unsigned int idx = num_per_thread * tid;
  if (idx < n)
    max_val = min_val = input[idx];
  for (int i = 1; (i < num_per_thread) && (idx + i < n); i++) {
    max_val = fmaxf(max_val, input[idx + i]);
    min_val = fminf(min_val, input[idx + i]);
  }
  unsigned mask = __ballot_sync(FULL_MASK, idx < n);
  if (idx < n) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      max_val = fmaxf(max_val, __shfl_down_sync(mask, max_val, offset));
      min_val = fminf(min_val, __shfl_down_sync(mask, min_val, offset));
    }
  }

  if (tid == 0) {
    unsigned int divisor = (1 << bits) - 1;
    maxmin[0] = (max_val - min_val) / divisor;
    maxmin[1] = min_val;
  }
  __syncwarp(FULL_MASK);
}

template <typename T>
__global__ void MaxMin_find_meta_parallel_warp(const T* input,
                                               unsigned char* meta, int n,
                                               int bucket_size, int bits) {
  T* maxmin = (T*)meta;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  int warp_idx = index / WARP_SIZE;
  int num_warps = stride / WARP_SIZE;
  for (int idx = warp_idx; idx < (n + bucket_size - 1) / bucket_size;
       idx += num_warps) {
    maxmin_warp(input + idx * bucket_size, maxmin + 2 * idx,
                fmaxf(bucket_size, n - idx * bucket_size), bits);
  }
}

template <>
__global__ void
MaxMin_find_meta_parallel_warp<__half>(const __half* input, unsigned char* meta,
                                       int n, int bucket_size, int bits) {}

/*
==== Functions for quantization preparation. ===
*/

template <typename T>
__global__ void MaxMin_find_meta_parallel(const T* input, unsigned char* meta,
                                          int n, int bucket_size, int bits) {
  T* maxmin = (T*)meta;
  unsigned int bstride = gridDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int num_buckets = (n + bucket_size - 1) / bucket_size;

  // We have a conception where one block works on one bucket.
  unsigned int bid = blockIdx.x;
  extern __shared__ T sdata[];
  T* sdata_max = sdata;
  T* sdata_min = &sdata[blockDim.x];
  for (unsigned int bucket_idx = bid; bucket_idx < num_buckets;
       bucket_idx += bstride) {
    maxmin[2 * bucket_idx] = input[bucket_idx * bucket_size];
    maxmin[2 * bucket_idx + 1] = input[bucket_idx * bucket_size];
    unsigned int num_elems_in_bucket =
        umin(n - bucket_idx * bucket_size, bucket_size);
    unsigned int num_iters_per_bucket =
        (num_elems_in_bucket + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < num_iters_per_bucket; i++) {
      unsigned int idx = bucket_size * bucket_idx + i * blockDim.x + tid;
      if (idx < n) {
        sdata_max[tid] = input[idx];
        sdata_min[tid] = input[idx];
      }
      __syncthreads();

      for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
          sdata_max[tid] = cfu::max(sdata_max[tid + s], sdata_max[tid]);
          sdata_min[tid] = cfu::min(sdata_min[tid + s], sdata_min[tid]);
        }
        __syncthreads();
      }

      if (tid == 0) {
        maxmin[2 * bucket_idx] =
            cfu::max(maxmin[2 * bucket_idx], sdata_max[tid]);
        maxmin[2 * bucket_idx + 1] =
            cfu::min(maxmin[2 * bucket_idx + 1], sdata_min[tid]);
      }
    }
    if (tid == 0) {
      unsigned int divisor = (1 << bits) - 1;
      maxmin[2 * bucket_idx] = cfu::div_int(
          cfu::sub(maxmin[2 * bucket_idx], maxmin[2 * bucket_idx + 1]),
          divisor);
    }
  }
}

template <typename T>
__global__ void MaxMin_find_meta(const T* input, unsigned char* meta, int n,
                                 int bucket_size, int bits) {
  T* maxmin = (T*)meta;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int divisor = (1 << bits) - 1;

  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
    T mmin = input[i * bucket_size];
    T mmax = input[i * bucket_size];
    for (int j = i * bucket_size + 1; j < fminf((i + 1) * bucket_size, n);
         j++) {
      mmin = cfu::min(mmin, input[j]);
      mmax = cfu::max(mmax, input[j]);
    }
    maxmin[2 * i] = cfu::div_int(cfu::sub(mmax, mmin), divisor);
    maxmin[2 * i + 1] = mmin;
  }
}

template <typename T>
__global__ void LinfNorm_find_meta(const T* input, unsigned char* meta, int n,
                                   const int bucket_size) {
  T* norms = (T*)meta;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
    T bmax = cfu::abs(input[i * bucket_size]);
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      bmax = cfu::max(bmax, fabsf(input[j]));
    }
    if (cfu::lt(cfu::abs(bmax), (T)EPS))
      bmax = cfu::add(bmax, (T)EPS);
    norms[i] = bmax;
  }
}

template <typename T>
__global__ void L2Norm_find_meta(const T* input, unsigned char* meta, int n,
                                 const int bucket_size) {
  T* norm = (T*)meta;
  int num_buckets = (n + bucket_size - 1) / bucket_size;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (int i = index; i < num_buckets; i += stride) {
    T bnorm = 0.0;
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      bnorm = cfu::add(bnorm, cfu::mul(input[j], input[j]));
    }
    if (cfu::lt(bnorm, EPS))
      bnorm = cfu::add(bnorm, EPS);
    //    if ((i + 1) * bucket_size > n) {
    //      // not full bucket. Need to rescale.
    //      bnorm *= bucket_size / ((n - i * bucket_size) * 1.0);
    //    }
    bnorm = cfu::sqrt(bnorm);
    norm[i] = bnorm;
  }
}



// Single value quantization functions
template <typename T>
inline __device__ unsigned char
MaxMinEncodeValue(T input, T* feedback, unsigned char* meta_info,
                  unsigned int idx, int bucket_size, int bits, float rand) {
  int bucket_no = idx / bucket_size;
  T* maxmin = ((T*)meta_info) + 2 * bucket_no;
  if (cfu::lt(maxmin[0], EPS)) {
    return 0;
  }
  T min = maxmin[1];
  T unit = maxmin[0];
  T d = cfu::add_float(cfu::div(cfu::sub(input, min), unit), rand);
  unsigned char level = cfu::floor(d);
  if (feedback)
    *feedback = cfu::sub(input, cfu::add(min, cfu::mul_int(unit, (int)level)));
  return level;
}

template <typename T>
inline __device__ T MaxMinDecodeValue(unsigned char input,
                                      unsigned char* meta_info,
                                      unsigned int idx, int bucket_size,
                                      int bits) {
  int bucket_no = idx / bucket_size;
  T* maxmin = ((T*)meta_info) + 2 * bucket_no;
  T min = maxmin[1];
  T unit = maxmin[0];
  return cfu::add(min, cfu::mul_int(unit, (int)input));
}

template <typename T>
inline __device__ unsigned char
NormPosEncodeValue(T input, T* feedback, unsigned char* meta_info,
                   unsigned int idx, int bucket_size, int bits, float rand,
                   void* ctx) {
  int bucket_no = idx / bucket_size;
  T norm = ((T*)meta_info)[bucket_no];
  char sign;
  int num_levels = 1 << (bits - 1);
  T* levels = (T*)ctx;
  T d = cfu::abs(cfu::div(input, norm));
  sign = cfu::lt(input, (T)-EPS);
  unsigned char level_idx = 0;
  T diff;
  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
  while (level_idx + 1 < num_levels) {
    if (cfu::lt(EPS, cfu::sub(d, levels[level_idx + 1]))) {
      diff = cfu::mul_float(cfu::sub(levels[level_idx], levels[level_idx + 1]), rand);
      if (cfu::lt(cfu::sub(cfu::add(d, diff), levels[level_idx]),(T)-EPS)) {
        level_idx++;
      }
      break;
    }
    level_idx++;
  }
  // update error feedback
  if (feedback) {
    T recovered_v = cfu::mul_float(norm, sign ? -1.0 : 1.0);
    if (bits > 1)
      recovered_v = cfu::mul(recovered_v, levels[level_idx]);
    *feedback = cfu::sub(input, recovered_v);
  }
  level_idx |= (sign << (bits - 1));
  return level_idx;
}

template <typename T>
inline __device__ T NormPosDecodeValue(unsigned char input,
                                       unsigned char* meta_info,
                                       unsigned int idx, int bucket_size,
                                       int bits, void* ctx) {
  int bucket_no = idx / bucket_size;
  T norm = ((T*)meta_info)[bucket_no];
  T* levels = (T*)ctx;
  int num_levels = 1 << (bits - 1);
  char sign = (input & num_levels) ? -1 : 1;
  input &= num_levels - 1;
  T decode_value = cfu::mul_int(norm, (int)sign);

  if (bits > 1) {
    decode_value = cfu::mul(decode_value, levels[input]);
  }
  return decode_value;
}

template <typename T>
inline __device__ unsigned char
NormWideEncodeValue(T input, T* feedback, unsigned char* meta_info,
                    unsigned int idx, int bucket_size, int bits, float rand,
                    void* ctx) {
  int bucket_no = idx / bucket_size;
  T norm = ((T*)meta_info)[bucket_no];
  int num_levels = 1 << bits;
  T* levels = (T*)ctx;
  T d = cfu::div(input, norm);
  T diff;
  unsigned char level_idx = 0;
  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
  while (level_idx + 1 < num_levels) {
    if (cfu::lt(EPS, cfu::sub(d, levels[level_idx + 1]))) {
      diff = cfu::mul_float(cfu::sub(levels[level_idx], levels[level_idx + 1]), rand);
      if (cfu::lt(cfu::sub(cfu::add(d, diff), levels[level_idx]),(T)-EPS)) {
        level_idx++;
      }
      break;
    }
    level_idx++;
  }
  // update error feedback
  if (feedback) {
    T recovered_v = norm;
    if (bits > 1)
      recovered_v = cfu::mul(recovered_v, levels[level_idx]);
    *feedback = cfu::sub(input, recovered_v);
  }
  return level_idx;
}

template <typename T>
inline __device__ T NormWideDecodeValue(unsigned char input,
                                        unsigned char* meta_info,
                                        unsigned int idx, int bucket_size,
                                        int bits, void* ctx) {
  int bucket_no = idx / bucket_size;
  T norm = ((T*)meta_info)[bucket_no];
  T* levels = (float*)ctx;
  T decode_value = norm;
  if (bits > 1) {
    decode_value = cfu::mul(decode_value, levels[input]);
  }
  return decode_value;
}

enum CompressMethod { MaxMin, NormWide, NormPos };

template <typename T, CompressMethod method>
inline __device__ T EncodeValue(T input, T* feedback, unsigned char* meta_info,
                                unsigned int idx, int bucket_size, int bits,
                                float rand, void* ctx) {

  switch (method) {
  case CompressMethod::MaxMin:
    return MaxMinEncodeValue(input, feedback, meta_info, idx, bucket_size, bits,
                             rand);
  case CompressMethod::NormWide:
    return NormWideEncodeValue(input, feedback, meta_info, idx, bucket_size,
                               bits, rand, ctx);
  case CompressMethod::NormPos:
    return NormPosEncodeValue(input, feedback, meta_info, idx, bucket_size,
                              bits, rand, ctx);
  default:
    assert(false);
  }
  return 0;
}

template <typename T, CompressMethod method>
inline __device__ T DecodeValue(unsigned char input, unsigned char* meta_info,
                                unsigned int idx, int bucket_size, int bits,
                                void* ctx) {
  switch (method) {
  case CompressMethod::MaxMin:
    return MaxMinDecodeValue<T>(input, meta_info, idx, bucket_size, bits);
  case CompressMethod::NormWide:
    return NormWideDecodeValue<T>(input, meta_info, idx, bucket_size, bits, ctx);
  case CompressMethod::NormPos:
    return NormPosDecodeValue<T>(input, meta_info, idx, bucket_size, bits, ctx);
  default:
    assert(false);
  }
  return 0;
}





template <typename T>
__device__ void find_maxmin_parallel(const T* input, unsigned char* meta,
                                          int num_elems, int bits) {
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  T* maxmin = (T*) meta;
  extern __shared__ T sdata[];
  T* sdata_max = sdata;
  T* sdata_min = &sdata[blockDim.x];
  maxmin[0] = input[0];
  maxmin[1] = input[0];
  unsigned int num_iters_per_bucket =
      (num_elems + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    unsigned int idx = i * blockDim.x + tid;
    if (idx < num_elems) {
      sdata_max[tid] = input[idx];
      sdata_min[tid] = input[idx];
    }
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s && idx + s < num_elems) {
        sdata_max[tid] = cfu::max(sdata_max[tid + s], sdata_max[tid]);
        sdata_min[tid] = cfu::min(sdata_min[tid + s], sdata_min[tid]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      maxmin[0] =
          cfu::max(maxmin[0], sdata_max[tid]);
      maxmin[1] =
          cfu::min(maxmin[1], sdata_min[tid]);
    }
  }
  if (tid == 0) {
    unsigned int divisor = (1 << bits) - 1;
    maxmin[0] = cfu::div_int(
        cfu::sub(maxmin[0], maxmin[1]),
        divisor);
  }
  __syncthreads();
}


template <typename T>
__device__ void CompressBucket(T* input, unsigned char* output, T* feedback_data, unsigned char* meta_info,
    int num_elems, int bucket_size, int bits, int offset, CurandState* state) {
  using uint64_t = unsigned long long int;
  unsigned int tid = threadIdx.x;
  unsigned int num_threads = blockDim.x;
  float rand;
  extern __shared__ uint64_t compute_data[];
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  T* feedback_ = nullptr;

  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += num_threads) {
    uint64_t value = 0;
    _Pragma("unroll 4")
    for (unsigned int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
      int idx = i * PACK_SIZE + j;
      if (feedback_data)
        feedback_ = feedback_data + idx;
      rand = GetRand(state);
      uint64_t encoded = MaxMinEncodeValue(input[idx], feedback_,
          meta_info, offset + idx, bucket_size, bits, rand);
      value += (encoded << (j * bits));
    }
    for (unsigned int j = 0; j < bits && i * bits + j < num_char; j++) {
      output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;
    }
  }
}

//int64_t __device__ round_to(int64_t x, int64_t m) { return x + ((m - x % m) % m); }

template <typename T>
__global__ void quantize_maxmin(T* input_data, unsigned char* output_data,
                                T* feedback_data, int num_elems, int bits,
                                int bucket_size, CurandState* states) {
  unsigned num_blocks = gridDim.x;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int bid = blockIdx.x;
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned int cur_bucket_size;
  T* meta = (T*) output_data;
  unsigned char* output = output_data + 2 * sizeof(T) * num_buckets;
  unsigned int compressed_size = (bucket_size * bits + PACK_SIZE - 1) / PACK_SIZE;

  T* input = (T*) input_data;
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    find_maxmin_parallel<T>(input + bucket_size * bucket_id, (unsigned char*) (meta + 2 * bucket_id),
        cur_bucket_size, bits);
  }
  CurandState local_state = states[tid];
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket<T>(input + bucket_size * bucket_id,
        output + compressed_size * bucket_id,
        feedback_data, (unsigned char*) meta,
        cur_bucket_size, bucket_size, bits, bucket_size * bucket_id, &local_state);
  }
  states[tid] = local_state;
}

template <typename T>
void CUDA_quantize_maxmin_new(T* input_data, unsigned char* output_data,
                          T* feedback_data, int num_elems, int bits,
                          int bucket_size, CurandState* states,
                          cudaStream_t stream) {
  quantize_maxmin<T><<<(num_elems + bucket_size - 1) / bucket_size, umin(MAX_THREADS_PER_BLOCK, bucket_size),
      2 * MAX_THREADS_PER_BLOCK * sizeof(T), stream>>>(input_data, output_data,
          feedback_data, num_elems, bits, bucket_size, states);
  CUDACHECK(cudaGetLastError());
}

template <typename T, CompressMethod method>
__global__ void PackArray(T* input, unsigned char* meta_info,
                          unsigned char* output, T* feedback, int num_elems,
                          int bucket_size, int bits, void* ctx,
                          CurandState* states) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  T* feedback_ = nullptr;
  CurandState local_state = states[tid];
  float rand;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    uint64_t value = 0;
    _Pragma("unroll 4")
    for (unsigned int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
      int idx = i * PACK_SIZE + j;
      if (feedback)
        feedback_ = feedback + idx;
      rand = GetRand(&local_state);
      uint64_t encoded = EncodeValue<T, method>(
          input[idx], feedback_, meta_info, idx, bucket_size, bits, rand, ctx);
      value += (encoded << (j * bits));
    }
    for (unsigned int j = 0; j < bits && i * bits + j < num_char; j++) {
      output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;
    }
  }
  states[tid] = local_state;
}

template <typename T, CompressMethod METHOD, bool ADD>
__global__ void UnpackArray(unsigned char* input, unsigned char* meta_info,
                            T* output, int num_elems, int bucket_size, int bits,
                            void* ctx) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  unsigned int divisor = 1 << bits;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    uint64_t value = 0;
    for (int j = 0; j < bits && i * bits + j < num_char; j++) {
      value |= ((uint64_t)input[i * bits + j]) << (j * PACK_SIZE);
    }
    _Pragma("unroll 4")
    for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
      unsigned char encoded_value = (value >> (j * bits)) & (divisor - 1);
      T d = DecodeValue<T, METHOD>(encoded_value, meta_info, i * PACK_SIZE + j,
                                   bucket_size, bits, ctx);
      if (ADD) {
        output[i * PACK_SIZE + j] += d;
      } else {
        output[i * PACK_SIZE + j] = d;
      }
    }
  }
}

template <typename T>
void CUDA_quantize_maxmin(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback_data, int num_elems, int bits,
                          int bucket_size, CurandState* states,
                          cudaStream_t stream) {
  T* input = (T*)input_data;
  unsigned char* meta_info = output_data;
  T* feedback = (T*)feedback_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* output = output_data + 2 * sizeof(T) * num_buckets;
  if (bucket_size > WARP_SIZE) {
    MaxMin_find_meta_parallel<T>
        <<<(num_elems + bucket_size - 1) / bucket_size, MAX_THREADS_PER_BLOCK,
           2 * MAX_THREADS_PER_BLOCK * sizeof(T), stream>>>(
            input, meta_info, num_elems, bucket_size, bits);
  } else {
    MaxMin_find_meta<T>
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
            input, meta_info, num_elems, bucket_size, bits);
  }
  CUDACHECK(cudaGetLastError());
  PackArray<T, CompressMethod::MaxMin>
      <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
          input, meta_info, output, feedback, num_elems, bucket_size, bits,
          NULL, states);
  CUDACHECK(cudaGetLastError());
}

template <typename T, bool ADD>
void CUDA_dequantize_maxmin(unsigned char* input_data,
                            unsigned char* output_data, int num_elems, int bits,
                            int bucket_size, cudaStream_t stream) {
  T* output = (T*)output_data;
  unsigned char* meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* input = input_data + 2 * sizeof(T) * num_buckets;
  UnpackArray<T, CompressMethod::MaxMin, ADD>
      <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
          input, meta_info, output, num_elems, bucket_size, bits, NULL);
  CUDACHECK(cudaGetLastError());
}

template <typename T>
void CUDA_quantize_Norm(unsigned char* input_data, unsigned char* output_data,
                        unsigned char* feedback, T* levels, int num_elems,
                        int bits, int bucket_size, CurandState* states,
                        NormType norm_type, LevelsType levels_type,
                        cudaStream_t stream) {
  T* input = (T*)input_data;
  unsigned char* meta_info = output_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* output = output_data + sizeof(T) * num_buckets;
  if (norm_type == NormType::L2) {
    L2Norm_find_meta<T>
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
            input, meta_info, num_elems, bucket_size);
  } else if (norm_type == NormType::Linf) {
    LinfNorm_find_meta<T>
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
            input, meta_info, num_elems, bucket_size);
  }
  CUDACHECK(cudaGetLastError());
  if (levels_type == LevelsType::Wide) {
    PackArray<T, CompressMethod::NormWide>
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
            input, meta_info, output, (T*)feedback, num_elems, bucket_size,
            bits, (void*)levels, states);
  } else {
    PackArray<T, CompressMethod::NormPos>
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
            input, meta_info, output, (T*)feedback, num_elems, bucket_size,
            bits, (void*)levels, states);
  }
  CUDACHECK(cudaGetLastError());
}

template <typename T, bool ADD>
void CUDA_dequantize_Norm(unsigned char* input_data, unsigned char* output_data,
                          T* levels, int num_elems, int bits, int bucket_size,
                          LevelsType levels_type,
                          cudaStream_t stream) {
  T* output = (T*)output_data;
  unsigned char* meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char* input = input_data + sizeof(T) * num_buckets;
  if (levels_type == LevelsType::Wide) {
    UnpackArray<T, CompressMethod::NormWide, ADD>
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
            input, meta_info, output, num_elems, bucket_size, bits,
            (void*)levels);
  } else {
    UnpackArray<T, CompressMethod::NormPos, ADD>
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(
            input, meta_info, output, num_elems, bucket_size, bits,
            (void*)levels);
  }
  CUDACHECK(cudaGetLastError());
}

void CUDA_convert_to_halves(float* input, Half* output, int numel) {
  cfu::float2half<<<numel, 1, 0, 0>>>(input, output, numel);
  assert(cudaStreamSynchronize(0) == cudaSuccess);
}

void set_blocks(int blocks);
#endif // CUDA_FUNCTIONS_H