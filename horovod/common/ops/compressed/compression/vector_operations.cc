#include "vector_operations.h"
#include "cuda/cuda_functions.h"
#include <omp.h>

namespace horovod {
namespace common {

void CPUSummator::Add(float* x, float* y, float* sum, int64_t num_elems) {
  int np;
#pragma omp parallel for simd num_threads(num_threads_)
  for (int i = 0; i < num_elems; i++) {
    sum[i] = x[i] + y[i];
  }
}

void GPUSummator::Add(float* x, float* y, float* sum, int64_t num_elems) {
  CUDA_add(num_elems, x, y, sum,
           gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

void GPUSummator::Finalize() {
  gpu_context_->StreamSynchronize(
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

void Summator::Add(float* x, TensorTableEntry& entry, int64_t num_elems) {
  device_ = entry.device;
  auto data = ((float*)entry.tensor->data());
  Add(x, data, data, num_elems);
}

void Summator::Finalize() {}

void Summator::Add(float* x,
                   std::vector<horovod::common::TensorTableEntry>& entries,
                   int64_t fusion_offset, int64_t global_offset,
                   int64_t num_elems, bool original) {
  device_ = entries[0].device;

  if (entries.size() == 1) {
    auto input_data =
        ((float*)entries[0].tensor->data()) + fusion_offset + global_offset;
    if (!original)
      input_data =
          ((float*)entries[0].output->data()) + fusion_offset + global_offset;
    auto output_data =
        ((float*)entries[0].output->data()) + fusion_offset + global_offset;

    Add(x, input_data, output_data, num_elems);
    Finalize();
    return;
  }

  int64_t offset_cumm = 0;
  int64_t nelem = 0;
  int64_t buffer_offset = 0;
  for (auto& entry : entries) {
    nelem = entry.output->shape().num_elements();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }

    if (offset_cumm - fusion_offset >= num_elems) {
      break;
    }
    buffer_offset = 0;
    if (offset_cumm < fusion_offset) {
      // If the first part of param group is placed in previous slice
      // depending on reduction algorithm.
      nelem = offset_cumm + nelem - fusion_offset;
      buffer_offset = entry.tensor->shape().num_elements() - nelem;
    }

    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + num_elems - std::max(offset_cumm, fusion_offset);
    }

    auto input_data = ((float*)entry.tensor->data()) + buffer_offset;
    if (!original)
      input_data = ((float*)entry.output->data()) + buffer_offset;
    auto output_data = ((float*)entry.output->data()) + buffer_offset;
    Add(x, input_data, output_data, nelem);
    offset_cumm += entry.tensor->shape().num_elements();
    x += nelem;
  }
  //  Finalize();
}

} // namespace common
} // namespace horovod
