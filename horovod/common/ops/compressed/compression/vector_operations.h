#ifndef HOROVOD_VECTOR_OPERATIONS_H
#define HOROVOD_VECTOR_OPERATIONS_H

#include "../../gpu_operations.h"

namespace horovod {
namespace common {

class Summator {
public:
  Summator() = default;
  virtual ~Summator() = default;

  void Add(float* x, TensorTableEntry& entry, int64_t num_elems);

  // @inplace stands for using output tensor from entry as input or original
  // one. entry.output = x + (entry.output|entry.tensor) if inplace set true use
  // output.
  void Add(float* x, std::vector<TensorTableEntry>& entries,
           int64_t fusion_offset, int64_t global_offset, int64_t num_elems,
           bool inplace);
  virtual void Add(float* x, float* y, float* sum, int64_t num_elems) = 0;
  virtual void Finalize();

protected:
  int device_;
};

class GPUSummator : public Summator {
public:
  GPUSummator(HorovodGlobalState* global_state, GPUContext* gpu_context)
      : gpu_context_(gpu_context), global_state_(global_state) {}
  void Add(float* x, float* y, float* sum, int64_t num_elems);
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
  int num_threads_ = 4;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_VECTOR_OPERATIONS_H
