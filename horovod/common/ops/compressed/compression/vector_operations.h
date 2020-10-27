#ifndef HOROVOD_VECTOR_OPERATIONS_H
#define HOROVOD_VECTOR_OPERATIONS_H

#include "../../gpu_operations.h"
#include "cuda/cuda_arithmetic_functions.h"

namespace horovod {
namespace common {

class Summator {
public:
  Summator() = default;
  virtual ~Summator() = default;

  template<typename T>
  void Add(T* x, TensorTableEntry& entry, int64_t num_elems) {
    device_ = entry.device;
    auto data = ((T*)entry.tensor->data());
    Add(x, data, data, num_elems);
  }

  // @original parameter stands for where take the values from entry: original
  // tensor or output.
  template<typename T>
  void Add(T* x, std::vector<TensorTableEntry>& entries,
           int64_t fusion_offset, int64_t global_offset, int64_t num_elems,
           bool original) {
    device_ = entries[0].device;

    if (entries.size() == 1) {
      auto input_data =
          ((T*)entries[0].tensor->data()) + fusion_offset + global_offset;
      if (!original)
        input_data =
            ((T*)entries[0].output->data()) + fusion_offset + global_offset;
      auto output_data =
          ((T*)entries[0].output->data()) + fusion_offset + global_offset;

      Add(x, input_data, output_data, num_elems);
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

      auto input_data = ((T*)entry.tensor->data()) + buffer_offset;
      if (!original)
        input_data = ((T*)entry.output->data()) + buffer_offset;
      auto output_data = ((T*)entry.output->data()) + buffer_offset;
      Add(x, input_data, output_data, nelem);
      offset_cumm += entry.tensor->shape().num_elements();
      x += nelem;
    }
  }

  virtual void Add(float* x, float* y, float* sum, int64_t num_elems) = 0;
  virtual void Add(Half* x, Half* y, Half* sum, int64_t num_elems) = 0;
  virtual void Finalize();

protected:
  int device_;
};

class GPUSummator : public Summator {
public:
  GPUSummator(HorovodGlobalState* global_state, GPUContext* gpu_context)
      : gpu_context_(gpu_context), global_state_(global_state) {}
  void Add(float* x, float* y, float* sum, int64_t num_elems);
  void Add(Half* x, Half* y, Half* sum, int64_t num_elems);
  void Finalize() override;

private:
  GPUContext* gpu_context_;
  HorovodGlobalState* global_state_;
};

class CPUSummator : public Summator {
public:
  CPUSummator() = default;
  // sum[i] = y[i] + x[i]
  void Add(float* x, float* y, float* sum, int64_t num_elems);
  void Add(Half* x, Half* y, Half* sum, int64_t num_elems);
  int num_threads_ = 4;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_VECTOR_OPERATIONS_H
