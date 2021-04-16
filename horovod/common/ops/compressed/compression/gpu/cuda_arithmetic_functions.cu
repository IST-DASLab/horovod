#include "cuda_arithmetic_functions.h"
#include <cuda_fp16.h>

namespace horovod {
namespace common {
namespace gpu {

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
void CUDA_add(int n, const T* x, T* y, T* sum,
              cudaStream_t stream) {
  int num_threads = min(n, MAX_THREADS_PER_BLOCK);
  int blocks = BLOCKS_PER_GRID(n, num_threads);
  _add<T><<<blocks, num_threads, 0, stream>>>(n, x, y, sum);
  CUDA_CHECK(cudaGetLastError());
}

template void CUDA_add<float>(int n, const float* x, float* y, float* sum,
                              cudaStream_t stream);
template void CUDA_add<Half>(int n, const Half* x, Half* y, Half* sum,
                             cudaStream_t stream);

} // namespace gpu
} // namespace common
} // namespace horovod
