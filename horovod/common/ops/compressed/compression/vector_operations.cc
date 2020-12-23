#include "vector_operations.h"
#include "cuda/cuda_arithmetic_functions.h"
#include <omp.h>
#include <stdexcept>

namespace horovod {
namespace common {
void CPUSummator::Add(float* x, float* y, float* sum, int num_elems) {
#pragma omp parallel for simd num_threads(num_threads_)
  for (int i = 0; i < num_elems; i++) {
    sum[i] = x[i] + y[i];
  }
}

void CPUSummator::Add(Half* x, Half* y, Half* sum, int num_elems) {
  throw std::logic_error("CPU summation doesn't support half type");
}

void GPUSummator::Add(float* x, float* y, float* sum, int num_elems) {
  cuda::CUDA_add<float>(
      num_elems, x, y, sum,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

void GPUSummator::Add(Half* x, Half* y, Half* sum, int num_elems) {
  cuda::CUDA_add<Half>(
      num_elems, x, y, sum,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

} // namespace common
} // namespace horovod
