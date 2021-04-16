#include "cuda_compression_functions.h"
#include "fp16_util.h"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#define MARK_VALUE (2<<31 - 1)
namespace horovod {
namespace common {
namespace gpu {

template <typename T>
__global__ void find_stats(T* input, T* stats, int num_elems) {
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  unsigned int stride = block_size * gridDim.x;
  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* sdata = reinterpret_cast<T*>(my_smem);
  unsigned int num_iters = (num_elems + stride - 1) / stride;
  T local_sum = 0.0, local_sum_sq = 0.0;
  for (int i = 0; i < num_iters; i++) {
    unsigned int idx = stride * i + blockIdx.x * block_size + tid;
    if (idx < num_elems) {
      sdata[tid] = isnan(input[idx]) ? (T)0.0 : abs(input[idx]);
      sdata[block_size + tid] =
          isnan(input[idx]) ? (T)0.0 : mul(input[idx], input[idx]);
    }
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s && idx + s < num_elems) {
        sdata[tid] = add(sdata[tid + s], sdata[tid]);
        sdata[block_size + tid] =
            add(sdata[block_size + tid + s], sdata[block_size + tid]);
      }
      __syncthreads();
    }
    if (tid == 0) {
      //      local_sum = add(local_sum, div_int(sdata[tid], num_elems));
      //      local_sum_sq =
      //          add(local_sum_sq, div_int(sdata[tid + block_size],
      //          num_elems));
      local_sum = add(local_sum, sdata[tid]);
      local_sum_sq = add(local_sum_sq, sdata[tid + block_size]);
    }
  }
  if (tid == 0) {
    //    printf("Local sum %f\n", local_sum);
    atomicAdd(&stats[0], local_sum);
    atomicAdd(&stats[1], local_sum_sq);
  }
  __syncthreads();
}

__device__ __inline__ double pow(double x, int y) {
  double result = 1.0;
  while (y > 0) {
    if (y & 1) {
      result *= x;
    }
    y >>= 1;
    x *= x;
  }
  return result;
}

template <typename T>
__global__ void my_memset(T* buf, unsigned int num_values, T value) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < num_values; i += stride) {
    buf[i] = value;
  }
}

__device__ double inv_cdf(double quantile) {
  static double a[4] = {2.50662823884, -18.61500062529, 41.39119773534,
                        -25.44106049637};

  static double b[4] = {-8.47351093090, 23.08336743743, -21.06224101826,
                        3.13082909833};

  static double c[9] = {
      0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
      0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
      0.0000321767881768, 0.0000002888167364, 0.0000003960315187};
  if (quantile >= 0.5 && quantile <= 0.92) {
    double num = 0.0;
    double denom = 1.0;
    double r = (quantile - 0.5) * (quantile - 0.5);
    double pow_cur = 1.0;
    for (int i = 0; i < 4; i++) {
      num += a[i] * (quantile - 0.5) * pow_cur;
      pow_cur *= r;
      denom += b[i] * pow_cur;
    }
    return num / denom;
  } else if (quantile > 0.92 && quantile < 1) {
    double num = 0.0;

    for (int i = 0; i < 9; i++) {
      num += c[i] * pow((logf(-logf(1 - quantile))), i);
    }
    return num;

  } else {
    return -1.0 * inv_cdf(1 - quantile);
  }
}

template <typename T>
void __global__ find_normal_quantile(T* stats, int num_elems, int num_result) {
  //  printf("Sum: %f, sum_sq %f\n", stats[0], stats[1]);
  T mean = div_int(stats[0], num_elems);
  T std = div_int(stats[1], num_elems);
  std = sqrt(sub(std, mul(mean, mean)));
  double quantile = 1.0 - (num_result * 1.0 / num_elems);
  T probit = float2type<T>((float)inv_cdf(quantile));
  memset((void*)stats, 0, sizeof(T) * 2);
  //  printf("Mean: %f, std:%f\n", mean, std);
  stats[0] = add(mean, mul(probit, std));
}
namespace thrust_reduction {
template <typename T, typename T1>
struct transtuple : public thrust::unary_function<T, T1> {
  __host__ __device__ T1 operator()(const T& a) { return T1(abs(a), a * a); }
};

template <typename T> struct sum : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(const T& a, const T& b) {
    return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                              thrust::get<1>(a) + thrust::get<1>(b));
  }
};
} // namespace thrust_reduction
template <typename T>
void topk_find_threshold_thrust(T* input, unsigned char* utility_buf,
                                int num_elems, int num_result,
                                cudaStream_t stream) {
  T* stats = (T*)utility_buf;
  thrust::device_ptr<T> stats_thr = thrust::device_pointer_cast(stats);
  thrust::device_ptr<T> input_thr = thrust::device_pointer_cast(input);
  typedef thrust::tuple<T, T> pair_type;
  pair_type init(0.0, 0.0);
  thrust_reduction::sum<pair_type> binary;
  thrust_reduction::transtuple<T, pair_type> unary;
  pair_type result =
      thrust::transform_reduce(thrust::cuda::par.on(stream), input_thr,
                               input_thr + num_elems, unary, init, binary);
  stats_thr[0] = thrust::get<0>(result);
  stats_thr[1] = thrust::get<1>(result);
  find_normal_quantile<<<1, 1, 0, stream>>>(stats, num_elems, num_result);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void topk_find_threshold(T* input, unsigned char* utility_buf, int num_elems,
                         int num_result, cudaStream_t stream) {
  int num_threads = std::min(MAX_THREADS_PER_BLOCK, num_elems);
  int blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  int shared_mem = 2 * num_threads * sizeof(T);
  T* stats = (T*)utility_buf;
  my_memset<<<1, 1, 0, stream>>>(stats, 2, (T)0.0);
  CUDA_CHECK(cudaGetLastError());
  find_stats<<<blocks, num_threads, shared_mem, stream>>>(input, stats,
                                                          num_elems);
  CUDA_CHECK(cudaGetLastError());
  find_normal_quantile<<<1, 1, 0, stream>>>(stats, num_elems, num_result);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void __global__ abs_buffer(T* src, T* dst, int num_values) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < num_values; i += stride) {
    dst[i] = abs(src[i]);
  }
}

void* copy_buf_ = nullptr;
int size_copy = 0;
template <typename T>
void topk_find_threshold_sort(T* input, unsigned char* utility_buf,
                              int num_elems, int num_result,
                              cudaStream_t stream) {
  T* stats = (T*)utility_buf;
  if (num_elems > size_copy) {
    if (copy_buf_) {
      CUDA_CHECK(cudaFree(copy_buf_));
    }
    CUDA_CHECK(cudaMalloc((void**)&copy_buf_, num_elems * sizeof(T)));
    size_copy = num_elems;
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  T* copy_buf = (T*)copy_buf_;
  int num_threads = std::min(MAX_THREADS_PER_BLOCK, num_result);
  int num_blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  abs_buffer<<<num_blocks, num_threads, 0, stream>>>(input, copy_buf,
                                                     num_elems);
  thrust::device_ptr<T> copy_thr = thrust::device_pointer_cast(copy_buf);
  thrust::device_ptr<T> input_thr = thrust::device_pointer_cast(input);
  thrust::sort(thrust::cuda::par.on(stream), copy_thr, copy_thr + num_elems);
  thrust::device_ptr<T> stats_thr = thrust::device_pointer_cast(stats);
  T threshold = copy_thr[num_elems - num_result - 1];
  int larger_than = thrust::count_if(thrust::cuda::par.on(stream), copy_buf,
                                     copy_buf + num_elems,
                   thrust::placeholders::_1 > threshold);
//  printf("Larger than: %i, expected: %i\n", larger_than, num_result);
  stats_thr[0] = threshold;
  stats_thr[1] = 0.0;
}

template <typename T, bool EF>
__global__ void topk_compress(T* input, unsigned int* indices, T* values,
                              unsigned char* utility_buf, T* feedback,
                              int num_elem, int num_result) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  T threshold = *((T*)utility_buf);
  unsigned int* index_p = (unsigned int*)(utility_buf + sizeof(T));
  unsigned int idx = 0;
  T value;
  //  if (tid == 0) {
  //    printf("Threshold: %f\n", threshold);
  //  }
  for (unsigned int i = tid; i < num_elem; i += stride) {
    value = input[i];
    if (EF)
      feedback[i] = value;
    if (le(threshold, abs(value)) and lt((T)1e-9, abs(value))) {
      idx = atomicAdd(index_p, 1);
      if (idx < num_result) {
        indices[idx] = i;
        values[idx] = value;
        if (EF)
          feedback[i] = 0.0;
      } else {
//          printf("Something is wrong value: %f, threshold %f\n", value, threshold);
      }
    }
  }
//    if (tid == 0) {
//      printf("Num values %i(expected %i) out of %i: %f\n", idx, num_result,
//             num_elem, 1.0 * idx / num_elem);
//    }
}

template <typename T, bool ADD>
__global__ void topk_decompress(unsigned int* indices, T* values, T* output,
                                int num_result) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  int idx = 0;
  for (unsigned int i = tid; i < num_result; i += stride) {
    if (indices[i] == MARK_VALUE) {
//      printf("Met Mark value idx: %i\n", i);
      break;
    }
    if (ADD) {
      output[indices[i]] = add(output[indices[i]], values[i]);
    } else {
      output[indices[i]] = values[i];
    }
//    if (lt((T)1e-6, abs(values[i])))
//      idx++;

  }
  //  printf("Non zero in decompress: %i\n", idx);
}

template <typename T> int get_non_zeros(T* a, int num_elems) { return 0; }

template <> int get_non_zeros<float>(float* a, int num_elems) {
  int num = 0;
  for (int i = 0; i < num_elems; i++) {
    if (1e-9 < fabs(a[i]))
      num++;
  }
  return num;
}

template <typename T>
void CUDA_topk_compress(unsigned char* input_data, unsigned char* output_data,
                        unsigned char* utility_buf,
                        unsigned char* feedback_data, int num_elems,
                        int num_result, cudaStream_t stream) {
  T* input = (T*)input_data;
  unsigned int* meta = (unsigned int*)output_data;
  T* output = (T*)(meta + num_result);
  //  printf("Num elems: %i num result: %i \n", num_elems, num_result);
//  topk_find_threshold(input, utility_buf, num_elems, num_result, stream);
  //  topk_find_threshold_thrust(input, utility_buf, num_elems, num_result,
  //  stream);
  topk_find_threshold_sort(input, utility_buf, num_elems, num_result, stream);

  int num_threads = std::min(MAX_THREADS_PER_BLOCK, num_result);
  int num_blocks = BLOCKS_PER_GRID(num_result, num_threads);
  my_memset<<<num_blocks, num_threads, 0, stream>>>(meta, num_result, (unsigned int)MARK_VALUE);
  CUDA_CHECK(cudaGetLastError());
  num_threads = std::min(MAX_THREADS_PER_BLOCK, num_elems);
  num_blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  if (feedback_data != nullptr) {
    topk_compress<T, true><<<num_blocks, num_threads, 0, stream>>>(
        input, meta, output, utility_buf, (T*)feedback_data, num_elems,
        num_result);
  } else {
    topk_compress<T, false><<<num_blocks, num_threads, 0, stream>>>(
        input, meta, output, utility_buf, nullptr, num_elems, num_result);
  }
  CUDA_CHECK(cudaGetLastError());
  //  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T, bool ADD>
void CUDA_topk_decompress(unsigned char* input_data, unsigned char* output_data,
                          int num_elems, int num_result, cudaStream_t stream) {
  unsigned int* meta = (unsigned int*)input_data;
  T* values = (T*)(meta + num_result);
  T* output = (T*)output_data;
  int num_threads = std::min(num_elems, MAX_THREADS_PER_BLOCK);
  int num_blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  if (!ADD) {
    my_memset<<<num_blocks, num_threads, 0, stream>>>(output, num_elems,
                                                      (T)0.0);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  num_threads = std::min(num_result, MAX_THREADS_PER_BLOCK);
  num_blocks = BLOCKS_PER_GRID(num_result, num_threads);
//  num_threads = 1;
//  num_blocks = 1;
  topk_decompress<T, ADD><<<num_blocks, num_threads, 0, stream>>>(
      meta, values, output, num_result);
//  T* output_host = new T[num_elems];
//  cudaMemcpyAsync((void*)output_host, (void*)output, num_elems * sizeof(T),
//                  cudaMemcpyDeviceToHost, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
//  unsigned int non_zeros = get_non_zeros(output_host, num_elems);
//  printf("Result non zeros: %i(expected %i) out of %i, %f\n", non_zeros,
//         num_result, num_elems, 1.0 * non_zeros / num_elems);
}

template void CUDA_topk_compress<float>(unsigned char* input_data,
                                        unsigned char* output_data,
                                        unsigned char* utility_buf,
                                        unsigned char* feedback_data,
                                        int num_elems, int num_result,
                                        cudaStream_t stream);

template void CUDA_topk_compress<Half>(unsigned char* input_data,
                                       unsigned char* output_data,
                                       unsigned char* utility_buf,
                                       unsigned char* feedback_data,
                                       int num_elems, int num_result,
                                       cudaStream_t stream);

template void CUDA_topk_decompress<float, true>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                int num_elems, int num_result,
                                                cudaStream_t stream);

template void CUDA_topk_decompress<float, false>(unsigned char* input_data,
                                                 unsigned char* output_data,
                                                 int num_elems, int num_result,
                                                 cudaStream_t stream);

template void CUDA_topk_decompress<Half, true>(unsigned char* input_data,
                                               unsigned char* output_data,
                                               int num_elems, int num_result,
                                               cudaStream_t stream);

template void CUDA_topk_decompress<Half, false>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                int num_elems, int num_result,
                                                cudaStream_t stream);

} // namespace gpu
} // namespace common
} // namespace horovod