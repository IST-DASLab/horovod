#include "cuda_arithmetic_functions.h"
#include <cuda_fp16.h>

namespace horovod {
namespace common {
namespace cuda {

template <typename T>
__global__ void _add(int64_t n, const T* x, const T* y, T* sum) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    sum[i] = x[i] + y[i];
  }
}

template <>
__global__ void _add<__half>(int64_t n, const __half* x, const __half* y,
                             __half* sum) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    sum[i] = __hadd(x[i], y[i]);
  }
}

template <typename T>
void CUDA_add(int64_t n, const T* x, T* y, T* sum, cudaStream_t stream) {
  int blocks = BLOCKS_PER_GRID(n);
  int num_threads = MAX_THREADS_PER_BLOCK;
  _add<T><<<blocks, num_threads, 0, stream>>>(n, x, y, sum);
  CUDA_CHECK(cudaGetLastError());
}

template void CUDA_add<float>(int64_t n, const float* x, float* y, float* sum,
                              cudaStream_t stream);
template void CUDA_add<Half>(int64_t n, const Half* x, Half* y, Half* sum,
                             cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace horovod
