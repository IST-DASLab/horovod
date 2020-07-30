#include "vector_operations.h"
#include <omp.h>
#include <stdexcept>

namespace horovod {
namespace common {

void CPUSummator::Add(float* x, float* y, float* sum, int64_t num_elems) {
#pragma omp parallel for simd num_threads(num_threads_)
  for (int i = 0; i < num_elems; i++) {
    sum[i] = x[i] + y[i];
  }
}

void CPUSummator::Add(Half* x, Half* y, Half* sum, int64_t num_elems) {
  throw std::logic_error("CPU summation doesn't support half type");
}

void GPUSummator::Add(float* x, float* y, float* sum, int64_t num_elems) {
  CUDA_add_fp32(num_elems, x, y, sum,
           gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

void GPUSummator::Add(Half* x, Half* y, Half* sum, int64_t num_elems) {
  CUDA_add_fp16(num_elems, x, y, sum,
           gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

void GPUSummator::Finalize() {
  gpu_context_->StreamSynchronize(
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}


void Summator::Finalize() {}

} // namespace common
} // namespace horovod
