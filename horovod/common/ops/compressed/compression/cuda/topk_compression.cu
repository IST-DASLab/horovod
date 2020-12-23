#include "cuda_compression_functions.h"
#include "cuda_fp16_util.h"

namespace horovod {
namespace common {
namespace cuda {

template <typename T>
__global__ void find_stats(T* input, T* stats, int num_elems) {
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  unsigned int stride = block_size * gridDim.x;
  const int shared_size = 2 * MAX_THREADS_PER_BLOCK;
  __shared__ T sdata[shared_size];
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
      local_sum = add(local_sum, div_int(sdata[tid], num_elems));
      local_sum_sq =
          add(local_sum_sq, div_int(sdata[tid + block_size], num_elems));
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
  //  while (y > 0) {
  //    if (y & 1) {
  //      result *= x;
  //    }
  //    y >>= 1;
  //    x *= x;
  //  }
  for (int i = 0; i < y; i++) {
    result *= x;
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
  //  double y = quantile - 0.5;
  //  double x = 0.0;
  //  if (fabs(y) < 0.42) {
  //    double r = y * y;
  //    printf("Nom correct: %f, denom correct: %f \n", y * (((a[3] * r + a[2])
  //    * r + a[1]) * r + a[0]), ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) *
  //    r + 1.0)); x = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]) /
  //           ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0);
  //  } else {
  //    double r = quantile;
  //    if (y > 0) {
  //      r = 1 - quantile;
  //    }
  //    r = log(-log(r));
  //    x =
  //        c[0] +
  //        r * (c[1] +
  //             r * (c[2] +
  //                  r * (c[3] +
  //                       r * (c[4] +
  //                            r * (c[5] + r * (c[6] + r * (c[7] + r *
  //                            c[8])))))));
  //    if (y < 0) {
  //      x = -x;
  //    }
  //  }
  //  return x;
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
  //  printf("Sum %f, Sum sq: %f, numel %i \n", stats[0], stats[1], num_elems);
  T mean = stats[0]; // div_int(, num_elems);
  T std = stats[1];  // div_int(, num_elems);
  std = sqrt(sub(std, mul(mean, mean)));
  //  printf("Mean %f, std: %f \n", mean, std);
  double quantile = 1.0 - (num_result * 1.0 / num_elems);
  T probit = float2type<T>((float)inv_cdf(quantile));
  memset((void*)stats, 0, sizeof(T) * 2);
  //  printf("Quantile %f Probit: %f\n", (float) quantile, probit);
  stats[0] = add(mean, mul(probit, std));
  //  printf("Threshold %f\n", stats[0]);
}

template <typename T>
void topk_find_threshold(T* input, unsigned char* utility_buf, int num_elems,
                         int num_result, cudaStream_t stream) {
  int num_threads = std::min(MAX_THREADS_PER_BLOCK, num_elems);
  int blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  //  int blocks = 1;
  int shared_mem = 2 * num_threads * sizeof(T);
  T* stats = (T*)utility_buf;
  my_memset<<<1, 1, 0, stream>>>(stats, 2, (T)0.0);
  //  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
  //  printf("Num threads %i num blocks %i\n", num_threads, blocks);
  find_stats<<<blocks, num_threads, shared_mem, stream>>>(input, stats,
                                                          num_elems);
  CUDA_CHECK(cudaGetLastError());
  //  CUDA_CHECK(cudaStreamSynchronize(stream));
  find_normal_quantile<<<1, 1, 0, stream>>>(stats, num_elems, num_result);
  //  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, bool EF>
__global__ void topk_compress(T* input, unsigned int* indices, T* values,
                              unsigned char* utility_buf, T* feedback,
                              int num_elem, int num_result) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  T threshold = *((T*)utility_buf);
  unsigned int* index_p = (unsigned int*)(utility_buf + sizeof(T));
  unsigned int idx;
  T value;
  //  printf("Compress indices: ");
  for (unsigned int i = tid; i < num_elem; i += stride) {
    value = input[i];
    if (lt(threshold, abs(value))) {
      idx = atomicAdd(index_p, 1);
      if (idx < num_result) {
        indices[idx] = i;
        //        printf("%i:%i:%u:%f ", i, idx, indices[idx], abs(value));
        values[idx] = value;
      }
    } else if (EF) {
      feedback[i] = value;
    }
  }
  //  printf("\n");
}

template <typename T, bool ADD>
__global__ void topk_decompress(unsigned int* indices, T* values, T* output,
                                int num_result) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  //  printf("Indices: ");
  for (unsigned int i = tid; i < num_result; i += stride) {
    if (indices[i] == UINT_MAX)
      break;
    //    printf("%i:%u ", i, indices[i]);
    if (ADD) {
      output[indices[i]] = add(output[indices[i]], values[i]);
    } else {
      output[indices[i]] = values[i];
    }
  }
  //  printf("\n");
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
  //  CUDA_CHECK(cudaStreamSynchronize(stream));
  topk_find_threshold(input, utility_buf, num_elems, num_result, stream);

  int num_threads = std::min(MAX_THREADS_PER_BLOCK, num_result);
  int num_blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  my_memset<<<num_blocks, num_threads, 0, stream>>>(meta, num_result, UINT_MAX);
  CUDA_CHECK(cudaGetLastError());
  //  CUDA_CHECK(cudaStreamSynchronize(stream));
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
  int num_threads = std::min(num_result, MAX_THREADS_PER_BLOCK);
  int num_blocks = BLOCKS_PER_GRID(num_result, num_threads);
  unsigned int* meta = (unsigned int*)input_data;
  T* values = (T*)(meta + num_result);
  T* output = (T*)output_data;
  topk_decompress<T, ADD><<<num_blocks, num_threads, 0, stream>>>(
      meta, values, output, num_result);
  //  printf("Num result: %i, num elem %i\n", num_result, num_elems);
  //  topk_decompress<T, ADD>
  //      <<<1, 1, 0, stream>>>(meta, values, output, num_result);
  //  CUDA_CHECK(cudaStreamSynchronize(stream));
}

size_t CUDA_get_topk_utility_buf_size() { return sizeof(float) * 2; }

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

} // namespace cuda
} // namespace common
} // namespace horovod