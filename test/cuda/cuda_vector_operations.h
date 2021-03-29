#ifndef HOROVOD_CUDA_VECTOR_OPERATIONS_H
#define HOROVOD_CUDA_VECTOR_OPERATIONS_H
#include <cuda_fp16.h>

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


template <typename T>
void CUDA_add(int n, const T* x, T* y, T* sum,
                            cudaStream_t stream) {
    int blocks = BLOCKS_PER_GRID(n);
    int num_threads = MAX_THREADS_PER_BLOCK;
    _add<T><<<blocks, num_threads, 0, stream>>>(n, x, y, sum);
    CUDACHECK(cudaGetLastError());
}


#endif // HOROVOD_CUDA_VECTOR_OPERATIONS_H
