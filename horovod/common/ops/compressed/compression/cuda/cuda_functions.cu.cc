#include "cuda_functions.h"
#include <assert.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <iostream>
#include <string>

const float EPS = 1e-10;
const int PACK_SIZE = 8;
const int MAX_THREADS_PER_BLOCK = 1024;
const int WARP_SIZE = 32;
constexpr int BLOCKS_PER_GRID(int num_elems) {
  return (num_elems + (MAX_THREADS_PER_BLOCK - 1)) / MAX_THREADS_PER_BLOCK;
}

#include "cuda_rand.h"

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

__device__ __half habs(__half a) {
  return __hlt(a, (__half)(-EPS)) ? __hneg(a) : a;
}

__device__ __half hmax(__half a, __half b) { return __hge(a, b) ? a : b; }

__device__ __half hmin(__half a, __half b) { return __hge(a, b) ? b : a; }

__global__ void float2half(float* input, __half* output, int numel) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < numel; i += stride) {
    output[i] = __float2half(input[i]);
  }
}
template <typename T>
__global__ void _add(int n, const T* x, const T* y, T* sum) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    sum[i] = x[i] + y[i];
  }
}

template <>
__global__ void _add<__half>(int n, const __half* x, const __half* y,
                             __half* sum) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    sum[i] = __hadd(x[i], y[i]);
  }
}

int CUDA_get_curand_array_size(int num_elems) {
  return BLOCKS_PER_GRID(num_elems) * MAX_THREADS_PER_BLOCK *
         sizeof(CurandState);
}

#define Add_Impl(type_name, T)                                                 \
  void CUDA_add_##type_name(int n, const T* x, T* y, T* sum,                   \
                            cudaStream_t stream) {                             \
    int blocks = BLOCKS_PER_GRID(n);                                           \
    int num_threads = MAX_THREADS_PER_BLOCK;                                   \
    _add<T><<<blocks, num_threads, 0, stream>>>(n, x, y, sum);                 \
  }

Add_Impl(fp32, float) Add_Impl(fp16, Half)
    /*
    ==== Functions for quantization preparation. ===
    */
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
      mmin = fminf(mmin, input[j]);
      mmax = fmaxf(mmax, input[j]);
    }
    maxmin[2 * i] = (mmax - mmin) / divisor;
    maxmin[2 * i + 1] = mmin;
  }
}

template <>
__global__ void MaxMin_find_meta<__half>(const __half* input,
                                         unsigned char* meta, int n,
                                         int bucket_size, int bits) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int divisor = (1 << bits) - 1;
  __half* maxmin = (__half*)meta;

  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
    __half mmax = input[i * bucket_size];
    __half mmin = input[i * bucket_size];
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      mmax = hmax(mmax, input[j]);
      mmin = hmin(mmin, input[j]);
    }
    maxmin[2 * i] = __hdiv(__hsub(mmax, mmin), __uint2half_rd(divisor));
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
    T bmax = fabsf(input[i * bucket_size]);
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      bmax = fmaxf(bmax, fabsf(input[j]));
    }
    if (fabsf(bmax) < EPS)
      bmax += EPS;
    norms[i] = bmax;
  }
}

template <>
__global__ void LinfNorm_find_meta<__half>(const __half* input,
                                           unsigned char* meta, int n,
                                           const int bucket_size) {
  __half* norms = (__half*)meta;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
    __half bmax = habs(input[i * bucket_size]);
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      bmax = hmax(bmax, habs(input[j]));
    }
    if (__hlt(habs(bmax), (__half)EPS))
      __hadd(bmax, (__half)EPS);
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
      bnorm += input[j] * input[j];
    }
    if (bnorm < EPS)
      bnorm += EPS;
    if ((i + 1) * bucket_size > n) {
      // not full bucket. Need to rescale.
      bnorm *= bucket_size / ((n - i * bucket_size) * 1.0);
    }
    bnorm = sqrt(bnorm);
    norm[i] = bnorm;
  }
}

template <>
__global__ void L2Norm_find_meta<__half>(const __half* input,
                                         unsigned char* meta, int n,
                                         const int bucket_size) {
  __half* norm = (__half*)meta;
  int num_buckets = (n + bucket_size - 1) / bucket_size;
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (int i = index; i < num_buckets; i += stride) {
    __half bnorm = (__half)0.0;
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      bnorm += __hmul(input[j], input[j]);
    }
    if (__hlt(bnorm, (__half)EPS))
      bnorm = __hadd(bnorm, (__half)EPS);
    if ((i + 1) * bucket_size > n) {
      // not full bucket. Need to rescale.
      float rescale_factor = bucket_size / ((n - i * bucket_size) * 1.0);
      bnorm = __hmul(bnorm, __float2half(rescale_factor));
    }
    bnorm = hsqrt(bnorm);
    norm[i] = bnorm;
  }
}

// Parallel reduction algorithms.
// One block reduces one bucket.
template <typename T>
__global__ void MaxMin_find_meta_parallel(const T* input, unsigned char* meta,
                                          int n, int bucket_size, int bits) {
  T* maxmin = (T*)meta;
  unsigned int bstride = gridDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int num_buckets = (n + bucket_size - 1) / bucket_size;

  unsigned int bid = blockIdx.x;
  __shared__ T sdata[2 * MAX_THREADS_PER_BLOCK];
  T* sdata_max = sdata;
  T* sdata_min = &sdata[MAX_THREADS_PER_BLOCK];
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

      for (unsigned int s = bucket_size / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n && tid + s < blockDim.x) {
          sdata_max[tid] = fmaxf(sdata_max[tid + s], sdata_max[tid]);
          sdata_min[tid] = fminf(sdata_min[tid + s], sdata_min[tid]);
        }
        __syncthreads();
      }

      if (tid == 0) {
        maxmin[2 * bucket_idx] = fmaxf(maxmin[2 * bucket_idx], sdata_max[tid]);
        maxmin[2 * bucket_idx + 1] =
            fminf(maxmin[2 * bucket_idx + 1], sdata_min[tid]);
      }
    }
    if (tid == 0) {
      unsigned int divisor = (1 << bits) - 1;
      maxmin[2 * bucket_idx] =
          (maxmin[2 * bucket_idx] - maxmin[2 * bucket_idx + 1]) / divisor;
    }
  }
}

template <>
__global__ void MaxMin_find_meta_parallel<__half>(const __half* input,
                                                  unsigned char* meta, int n,
                                                  int bucket_size, int bits) {
  __half* maxmin = (__half*)meta;

  unsigned int bstride = gridDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int num_buckets = (n + bucket_size - 1) / bucket_size;

  unsigned int bid = blockIdx.x;
  __shared__ __half sdata[2 * MAX_THREADS_PER_BLOCK];
  __half* sdata_max = sdata;
  __half* sdata_min = &sdata[MAX_THREADS_PER_BLOCK];
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

      for (unsigned int s = bucket_size / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n && tid + s < blockDim.x) {
          sdata_max[tid] = hmax(sdata_max[tid + s], sdata_max[tid]);
          sdata_min[tid] = hmin(sdata_min[tid + s], sdata_min[tid]);
        }
        __syncthreads();
      }

      if (tid == 0) {
        maxmin[2 * bucket_idx] = hmax(maxmin[2 * bucket_idx], sdata_max[tid]);
        maxmin[2 * bucket_idx + 1] =
            hmin(maxmin[2 * bucket_idx + 1], sdata_min[tid]);
      }
    }
    if (tid == 0) {
      unsigned int divisor = (1 << bits) - 1;
      maxmin[2 * bucket_idx] =
          __hdiv(__hsub(maxmin[2 * bucket_idx], maxmin[2 * bucket_idx + 1]),
                 __uint2half_rd(divisor));
    }
  }
}

template <typename T>
__global__ void LinfNorm_find_meta_parallel(const T* input, unsigned char* meta,
                                            int n, int bucket_size) {
  T* norm = (T*)meta;
  unsigned int bstride = gridDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int num_buckets = (n + bucket_size - 1) / bucket_size;

  unsigned int bid = blockIdx.x;
  __shared__ T sdata[MAX_THREADS_PER_BLOCK];
  for (unsigned int bucket_idx = bid; bucket_idx < num_buckets;
       bucket_idx += bstride) {
    norm[bucket_idx] = input[bucket_idx * bucket_size];
    unsigned int num_elems_in_bucket =
        umin(n - bucket_idx * bucket_size, bucket_size);
    unsigned int num_iters_per_bucket =
        (num_elems_in_bucket + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < num_iters_per_bucket; i++) {
      unsigned int idx = bucket_size * bucket_idx + i * blockDim.x + tid;
      if (idx < n) {
        sdata[tid] = input[idx];
      }
      __syncthreads();

      for (unsigned int s = bucket_size / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n && tid + s < blockDim.x) {
          sdata[tid] = fmaxf(fabs(sdata[tid + s]), fabs(sdata[tid]));
        }
        __syncthreads();
      }

      if (tid == 0) {
        norm[bucket_idx] = fmaxf(norm[bucket_idx], sdata[tid]);
      }
    }
    if (tid == 0) {
      if (norm[bucket_idx] < EPS)
        norm[bucket_idx] += EPS;
    }
  }
}

template <>
__global__ void LinfNorm_find_meta_parallel<__half>(const __half* input,
                                                    unsigned char* meta, int n,
                                                    int bucket_size) {
  __half* norm = (__half*)meta;

  unsigned int bstride = gridDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int num_buckets = (n + bucket_size - 1) / bucket_size;

  unsigned int bid = blockIdx.x;
  __shared__ __half sdata[MAX_THREADS_PER_BLOCK];
  for (unsigned int bucket_idx = bid; bucket_idx < num_buckets;
       bucket_idx += bstride) {
    norm[bucket_idx] = input[bucket_idx * bucket_size];
    unsigned int num_elems_in_bucket =
        umin(n - bucket_idx * bucket_size, bucket_size);
    unsigned int num_iters_per_bucket =
        (num_elems_in_bucket + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < num_iters_per_bucket; i++) {
      unsigned int idx = bucket_size * bucket_idx + i * blockDim.x + tid;
      if (idx < n) {
        sdata[tid] = input[idx];
      }
      __syncthreads();

      for (unsigned int s = bucket_size / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n && tid + s < blockDim.x) {
          sdata[tid] = hmax(habs(sdata[tid + s]), habs(sdata[tid]));
        }
        __syncthreads();
      }

      if (tid == 0) {
        norm[bucket_idx] = hmax(norm[bucket_idx], sdata[tid]);
      }
    }
    if (tid == 0) {
      if (__hlt(habs(norm[bucket_idx]), (__half)EPS))
        __hadd(norm[bucket_idx], (__half)EPS);
    }
  }
}

template <typename T>
__global__ void L2Norm_find_meta_parallel(const T* input, unsigned char* meta,
                                          int n, int bucket_size) {
  T* norm = (T*)meta;
  unsigned int bstride = gridDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int num_buckets = (n + bucket_size - 1) / bucket_size;

  unsigned int bid = blockIdx.x;
  __shared__ T sdata[MAX_THREADS_PER_BLOCK];
  for (unsigned int bucket_idx = bid; bucket_idx < num_buckets;
       bucket_idx += bstride) {
    norm[bucket_idx] = input[bucket_idx * bucket_size];
    unsigned int num_elems_in_bucket =
        umin(n - bucket_idx * bucket_size, bucket_size);
    unsigned int num_iters_per_bucket =
        (num_elems_in_bucket + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < num_iters_per_bucket; i++) {
      unsigned int idx = bucket_size * bucket_idx + i * blockDim.x + tid;
      if (idx < n) {
        sdata[tid] = input[idx] * input[idx];
      }
      __syncthreads();

      for (unsigned int s = bucket_size / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n && tid + s < blockDim.x) {
          sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
      }

      if (tid == 0) {
        norm[bucket_idx] += sdata[tid];
      }
    }
    if (tid == 0) {
      if (norm[bucket_idx] < EPS)
        norm[bucket_idx] += EPS;
      norm[bucket_idx] = sqrt(norm[bucket_idx]);
    }
  }
}

template <>
__global__ void L2Norm_find_meta_parallel<__half>(const __half* input,
                                                  unsigned char* meta, int n,
                                                  int bucket_size) {
  __half* norm = (__half*)meta;

  unsigned int bstride = gridDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int num_buckets = (n + bucket_size - 1) / bucket_size;

  unsigned int bid = blockIdx.x;
  __shared__ __half sdata[MAX_THREADS_PER_BLOCK];
  for (unsigned int bucket_idx = bid; bucket_idx < num_buckets;
       bucket_idx += bstride) {
    norm[bucket_idx] = input[bucket_idx * bucket_size];
    unsigned int num_elems_in_bucket =
        umin(n - bucket_idx * bucket_size, bucket_size);
    unsigned int num_iters_per_bucket =
        (num_elems_in_bucket + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < num_iters_per_bucket; i++) {
      unsigned int idx = bucket_size * bucket_idx + i * blockDim.x + tid;
      if (idx < n) {
        sdata[tid] = __hmul(input[idx], input[idx]);
      }
      __syncthreads();

      for (unsigned int s = bucket_size / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n && tid + s < blockDim.x) {
          sdata[tid] = __hadd(sdata[tid + s], sdata[tid]);
        }
        __syncthreads();
      }

      if (tid == 0) {
        norm[bucket_idx] = __hadd(norm[bucket_idx], sdata[tid]);
      }
    }
    if (tid == 0) {
      if (__hlt(habs(norm[bucket_idx]), (__half)EPS))
        __hadd(norm[bucket_idx], (__half)EPS);
    }
  }
}

// Single value quantization functions
template <typename T>
inline __device__ unsigned char
MaxMinEncodeValue(T input, T* feedback, unsigned char* meta_info,
                  unsigned int idx, int bucket_size, int bits, float rand,
                  void* ctx) {
  int bucket_no = idx / bucket_size;
  T* maxmin = ((T*)meta_info) + 2 * bucket_no;
  if (maxmin[0] - maxmin[1] < EPS) {
    return 0;
  }
  T min = maxmin[1];
  T unit = maxmin[0];
  T d = ((input - min) / unit) + rand;
  unsigned char level = (unsigned char)floor(d);
  if (feedback)
    *feedback = input - (min + level * unit);
  return level;
}

template <>
inline __device__ unsigned char
MaxMinEncodeValue<__half>(__half input, __half* feedback,
                          unsigned char* meta_info, unsigned int idx,
                          int bucket_size, int bits, float rand, void* ctx) {
  int bucket_no = idx / bucket_size;
  __half* maxmin = ((__half*)meta_info) + 2 * bucket_no;
  if (__hle(__hsub(maxmin[0], maxmin[1]), (__half)EPS)) {
    return 0;
  }
  __half rand_fp16 = __float2half(rand);
  __half min = maxmin[1];
  __half unit = maxmin[0];
  __half d = __hsub(input, min);
  d = __hdiv(d, unit);
  d = __hadd(d, rand_fp16);
  unsigned char level = (unsigned char)__half2uint_rd(hfloor(d));
  if (feedback) {
    d = __hadd(min, __hmul(level, unit));
    *feedback = __hsub(input, d);
  }
  return level;
}

template <typename T>
inline __device__ T MaxMinDecodeValue(unsigned char input,
                                      unsigned char* meta_info,
                                      unsigned int idx, int bucket_size,
                                      int bits, void* ctx) {
  int bucket_no = idx / bucket_size;
  T* maxmin = ((T*)meta_info) + 2 * bucket_no;
  T min = maxmin[1];
  T unit = maxmin[0];
  return min + unit * input;
}

template <>
inline __device__ __half MaxMinDecodeValue<__half>(unsigned char input,
                                                   unsigned char* meta_info,
                                                   unsigned int idx,
                                                   int bucket_size, int bits,
                                                   void* ctx) {
  int bucket_no = idx / bucket_size;
  __half* maxmin = ((__half*)meta_info) + 2 * bucket_no;
  __half unit = maxmin[0];
  __half min = maxmin[1];
  return __hadd(min, __hmul(unit, __uint2half_rd((unsigned int)input)));
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
  T d = fabs(input / norm);
  sign = (input < -EPS);
  unsigned char level_idx = 0;

  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
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
    T recovered_v = norm * (sign ? -1.0 : 1.0);
    if (bits > 1)
      recovered_v *= levels[level_idx];
    *feedback = input - recovered_v;
  }
  level_idx |= (sign << (bits - 1));
  return level_idx;
}

template <>
inline __device__ unsigned char
NormPosEncodeValue<__half>(__half input, __half* feedback,
                           unsigned char* meta_info, unsigned int idx,
                           int bucket_size, int bits, float rand, void* ctx) {
  int bucket_no = idx / bucket_size;
  __half norm = ((__half*)meta_info)[bucket_no];
  int num_levels = 1 << (bits - 1);
  __half* levels = (__half*)ctx;
  __half d = habs(__hdiv(input, norm));
  bool sign = __hlt(input, (__half)(-EPS));
  unsigned char level_idx = 0;
  __half rand_fp16 = __float2half(rand);
  while (level_idx + 1 < num_levels) {
    if (__hgt(__hsub(d, levels[level_idx + 1]), (__half)EPS)) {
      __half diff =
          __hmul(__hsub(levels[level_idx], levels[level_idx + 1]), rand_fp16);
      if (__hlt(__hsub(__hadd(d, diff), levels[level_idx]), (__half)(-EPS))) {
        level_idx++;
      }
      break;
    }
    level_idx++;
  }
  // update error feedback
  if (feedback) {
    __half recovered_v = __hmul(norm, (sign ? (__half)(-1.0) : (__half)(1.0)));
    if (bits > 1)
      recovered_v = __hmul(recovered_v, levels[level_idx]);
    *feedback = __hsub(input, recovered_v);
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
  T decode_value = norm * sign;

  if (bits > 1) {
    decode_value *= levels[input];
  }
  return decode_value;
}

template <>
inline __device__ __half NormPosDecodeValue<__half>(unsigned char input,
                                                    unsigned char* meta_info,
                                                    unsigned int idx,
                                                    int bucket_size, int bits,
                                                    void* ctx) {
  int bucket_no = idx / bucket_size;
  __half norm = ((__half*)meta_info)[bucket_no];
  __half* levels = (__half*)ctx;
  int num_levels = 1 << (bits - 1);
  __half sign = (input & num_levels) ? (__half)(-1.0) : (__half)(1.0);
  input &= num_levels - 1;
  __half decode_value = __hmul(norm, sign);
  if (bits > 1) {
    decode_value = __hmul(decode_value, levels[input]);
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
  T d = input / norm;
  unsigned char level_idx = 0;
  unsigned char flevel = 0;
  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
  while (level_idx + 1 < num_levels) {
    if (d - levels[level_idx + 1] > EPS) {
      flevel = level_idx;
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
    T recovered_v = norm;
    if (bits > 1)
      recovered_v *= levels[level_idx];
    *feedback = input - recovered_v;
  }
  return level_idx;
}

template <>
inline __device__ unsigned char
NormWideEncodeValue<__half>(__half input, __half* feedback,
                            unsigned char* meta_info, unsigned int idx,
                            int bucket_size, int bits, float rand, void* ctx) {
  int bucket_no = idx / bucket_size;
  __half norm = ((__half*)meta_info)[bucket_no];
  int num_levels = 1 << bits;
  __half* levels = (__half*)ctx;
  __half d = __hdiv(input, norm);
  __half rand_fp16 = __float2half(rand);
  unsigned char level_idx = 0;
  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
  while (level_idx + 1 < num_levels) {
    if (__hgt(__hsub(d, levels[level_idx + 1]), (__half)EPS)) {
      __half diff =
          __hmul(__hsub(levels[level_idx], levels[level_idx + 1]), rand_fp16);
      if (__hlt(__hsub(__hadd(d, diff), levels[level_idx]), (__half)(-EPS))) {
        level_idx++;
      }
      break;
    }
    level_idx++;
  }
  // update error feedback
  if (feedback) {
    __half recovered_v = norm;
    if (bits > 1)
      recovered_v = __hmul(recovered_v, levels[level_idx]);
    *feedback = __hsub(input, recovered_v);
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
    decode_value *= levels[input];
  }
  return decode_value;
}

template <>
inline __device__ __half NormWideDecodeValue<__half>(unsigned char input,
                                                     unsigned char* meta_info,
                                                     unsigned int idx,
                                                     int bucket_size, int bits,
                                                     void* ctx) {
  int bucket_no = idx / bucket_size;
  __half norm = ((__half*)meta_info)[bucket_no];
  __half* levels = (__half*)ctx;
  __half decode_value = norm;
  if (bits > 1) {
    decode_value = __hmul(decode_value, levels[input]);
  }
  return decode_value;
}

template <typename T> inline __device__ T single_mult_add(T a, T b, T c) {
  return a * b + c;
}

template <>
inline __device__ __half single_mult_add<__half>(__half a, __half b, __half c) {
  return __hadd(__hmul(a, b), c);
}

// Packaging functions
#define PACK_ARRAY_FUNC(type)                                                  \
  template <typename T>                                                        \
  __global__ void PackArray##type(T* input, unsigned char* meta_info,          \
                                  unsigned char* output, T* feedback,          \
                                  int num_elems, int bucket_size, int bits,    \
                                  void* ctx, CurandState* states) {            \
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;                  \
    unsigned int stride = gridDim.x * blockDim.x;                              \
    int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;             \
    T* feedback_ = nullptr;                                                    \
    CurandState local_state = states[tid];                                     \
    float rand;                                                                \
    for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;    \
         i += stride) {                                                        \
      uint64_t value = 0;                                                      \
      _Pragma("unroll 4") for (unsigned int j = 0;                             \
                               j < PACK_SIZE && i * PACK_SIZE + j < num_elems; \
                               j++) {                                          \
        int idx = i * PACK_SIZE + j;                                           \
        if (feedback)                                                          \
          feedback_ = feedback + idx;                                          \
        rand = GetRand(&local_state);                                          \
        uint64_t encoded =                                                     \
            type##EncodeValue<T>(input[idx], feedback_, meta_info, idx,        \
                                 bucket_size, bits, rand, ctx);                \
        value += (encoded << (j * bits));                                      \
      }                                                                        \
      for (unsigned int j = 0; j < bits && i * bits + j < num_char; j++) {     \
        output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;                \
      }                                                                        \
    }                                                                          \
    states[tid] = local_state;                                                 \
  }

#define UNPACK_ARRAY_FUNC(type)                                                \
  template <typename T>                                                        \
  __global__ void UnpackArray##type(                                           \
      unsigned char* input, unsigned char* meta_info, T* output,               \
      int num_elems, int bucket_size, int bits, void* ctx, bool add) {         \
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;                  \
    unsigned int stride = gridDim.x * blockDim.x;                              \
    int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;             \
    unsigned int divisor = 1 << bits;                                          \
    T add_t = (T)add;                                                          \
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
        T d = type##DecodeValue<T>(encoded_value, meta_info,                   \
                                   i * PACK_SIZE + j, bucket_size, bits, ctx); \
        output[i * PACK_SIZE + j] =                                            \
            single_mult_add(add_t, output[i * PACK_SIZE + j], d);              \
      }                                                                        \
    }                                                                          \
  }

PACK_ARRAY_FUNC(MaxMin)
UNPACK_ARRAY_FUNC(MaxMin)
PACK_ARRAY_FUNC(NormPos)
UNPACK_ARRAY_FUNC(NormPos)
PACK_ARRAY_FUNC(NormWide)
UNPACK_ARRAY_FUNC(NormWide)

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
      float d = (x[i * parts + j] - maxandmin[my_bucket * 2 + 1]) / unit +
                GetRand(&local_state);
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

#define MaxminQuantizeImpl(type_name, T)                                       \
  void CUDA_quantize_maxmin_##type_name(                                       \
      unsigned char* input_data, unsigned char* output_data,                   \
      unsigned char* feedback_data, int num_elems, int bits, int bucket_size,  \
      CurandState* states, cudaStream_t stream) {                              \
    T* input = (T*)input_data;                                                 \
    unsigned char* meta_info = output_data;                                    \
    T* feedback = (T*)feedback_data;                                           \
    int num_buckets = (num_elems + bucket_size - 1) / bucket_size;             \
    unsigned char* output = output_data + 2 * sizeof(T) * num_buckets;         \
    if (bucket_size < WARP_SIZE) {                                             \
      MaxMin_find_meta<T>                                                      \
          <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(  \
              input, meta_info, num_elems, bucket_size, bits);                 \
    } else {                                                                   \
      MaxMin_find_meta_parallel<T>                                             \
          <<<(num_elems + bucket_size - 1) / bucket_size,                      \
             MAX_THREADS_PER_BLOCK, 2 * sizeof(T) * MAX_THREADS_PER_BLOCK,     \
             stream>>>(input, meta_info, num_elems, bucket_size, bits);        \
    }                                                                          \
    CUDA_CHECK(cudaGetLastError());                                            \
    PackArrayMaxMin<T>                                                         \
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(    \
            input, meta_info, output, feedback, num_elems, bucket_size, bits,  \
            NULL, states);                                                     \
    CUDA_CHECK(cudaGetLastError());                                            \
  }

#define MaxminDequantizeImpl(type_name, T)                                     \
  void CUDA_dequantize_maxmin_##type_name(                                     \
      unsigned char* input_data, unsigned char* output_data, int num_elems,    \
      int bits, int bucket_size, bool add, cudaStream_t stream) {              \
    T* output = (T*)output_data;                                               \
    unsigned char* meta_info = input_data;                                     \
    int num_buckets = (num_elems + bucket_size - 1) / bucket_size;             \
    unsigned char* input = input_data + 2 * sizeof(T) * num_buckets;           \
    UnpackArrayMaxMin<T>                                                       \
        <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(    \
            input, meta_info, output, num_elems, bucket_size, bits, NULL,      \
            add);                                                              \
    CUDA_CHECK(cudaGetLastError());                                            \
  }

#define NormQuantizeImpl(type_name, T)                                         \
  void CUDA_quantize_Norm_##type_name(                                         \
      unsigned char* input_data, unsigned char* output_data,                   \
      unsigned char* feedback, T* levels, int num_elems, int bits,             \
      int bucket_size, CurandState* states,                                    \
      horovod::common::NormType norm_type,                                     \
      horovod::common::LevelsType levels_type, cudaStream_t stream) {          \
    T* input = (T*)input_data;                                                 \
    unsigned char* meta_info = output_data;                                    \
    int num_buckets = (num_elems + bucket_size - 1) / bucket_size;             \
    unsigned char* output = output_data + sizeof(T) * num_buckets;             \
    if (norm_type == horovod::common::NormType::L2) {                          \
      if (bucket_size < WARP_SIZE) {                                           \
        L2Norm_find_meta<T>                                                    \
            <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,           \
               stream>>>(input, meta_info, num_elems, bucket_size);            \
      } else {                                                                 \
        L2Norm_find_meta_parallel<T>                                           \
            <<<(num_elems + bucket_size - 1) / bucket_size,                    \
               MAX_THREADS_PER_BLOCK, sizeof(T) * MAX_THREADS_PER_BLOCK,       \
               stream>>>(input, meta_info, num_elems, bucket_size);            \
      }                                                                        \
    } else if (norm_type == horovod::common::NormType::Linf) {                 \
      if (bucket_size < WARP_SIZE) {                                           \
        LinfNorm_find_meta<T>                                                  \
            <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0,           \
               stream>>>(input, meta_info, num_elems, bucket_size);            \
      } else {                                                                 \
        LinfNorm_find_meta_parallel<T>                                         \
            <<<(num_elems + bucket_size - 1) / bucket_size,                    \
               MAX_THREADS_PER_BLOCK, sizeof(T) * MAX_THREADS_PER_BLOCK,       \
               stream>>>(input, meta_info, num_elems, bucket_size);            \
      }                                                                        \
    }                                                                          \
    CUDA_CHECK(cudaGetLastError());                                            \
    if (levels_type == horovod::common::LevelsType::Wide) {                    \
      PackArrayNormWide<T>                                                     \
          <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(  \
              input, meta_info, output, (T*)feedback, num_elems, bucket_size,  \
              bits, (void*)levels, states);                                    \
    } else {                                                                   \
      PackArrayNormPos<T>                                                      \
          <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(  \
              input, meta_info, output, (T*)feedback, num_elems, bucket_size,  \
              bits, (void*)levels, states);                                    \
    }                                                                          \
    CUDA_CHECK(cudaGetLastError());                                            \
  }

#define NormDequantizeImpl(type_name, T)                                       \
  void CUDA_dequantize_Norm_##type_name(                                       \
      unsigned char* input_data, unsigned char* output_data, T* levels,        \
      int num_elems, int bits, int bucket_size,                                \
      horovod::common::LevelsType levels_type, bool add,                       \
      cudaStream_t stream) {                                                   \
    T* output = (T*)output_data;                                               \
    unsigned char* meta_info = input_data;                                     \
    int num_buckets = (num_elems + bucket_size - 1) / bucket_size;             \
    unsigned char* input = input_data + sizeof(T) * num_buckets;               \
    if (levels_type == horovod::common::LevelsType::Wide) {                    \
      UnpackArrayNormWide<T>                                                   \
          <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(  \
              input, meta_info, output, num_elems, bucket_size, bits,          \
              (void*)levels, add);                                             \
    } else {                                                                   \
      UnpackArrayNormPos<T>                                                    \
          <<<BLOCKS_PER_GRID(num_elems), MAX_THREADS_PER_BLOCK, 0, stream>>>(  \
              input, meta_info, output, num_elems, bucket_size, bits,          \
              (void*)levels, add);                                             \
    }                                                                          \
    CUDA_CHECK(cudaGetLastError());                                            \
  }

MaxminQuantizeImpl(fp32, float) MaxminQuantizeImpl(fp16, Half)
    MaxminDequantizeImpl(fp32, float) MaxminDequantizeImpl(fp16, Half)
        NormQuantizeImpl(fp32, float) NormQuantizeImpl(fp16, Half)
            NormDequantizeImpl(fp32, float) NormDequantizeImpl(fp16, Half)

                void CUDA_convert_to_halves(float* input, Half* output,
                                            int numel) {
  float2half<<<numel, 1, 0, 0>>>(input, output, numel);
  assert(cudaStreamSynchronize(0) == cudaSuccess);
}