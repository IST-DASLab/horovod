#ifndef HOROVOD_VECTOR_OPERATIONS_H
#define HOROVOD_VECTOR_OPERATIONS_H

#include "../../gpu_operations.h"
#include "gpu/cuda_arithmetic_functions.h"

namespace horovod {
namespace common {

class Summator {
public:
  Summator() = default;
  virtual ~Summator() = default;

  template<typename T>
  void Add(T* x, TensorTableEntry& entry, int num_elems) {
    device_ = entry.device;
    auto data = ((T*)entry.output->data());
    Add(data, x, data, num_elems);
  }

  virtual void Add(float* x, float* y, float* sum, int num_elems) = 0;
  virtual void Add(Half* x, Half* y, Half* sum, int num_elems) = 0;

protected:
  int device_;
};

class GPUSummator : public Summator {
public:
  GPUSummator(HorovodGlobalState* global_state, GPUContext* gpu_context)
      : gpu_context_(gpu_context), global_state_(global_state) {}
  void Add(float* x, float* y, float* sum, int num_elems);
  void Add(Half* x, Half* y, Half* sum, int num_elems);

private:
  GPUContext* gpu_context_;
  HorovodGlobalState* global_state_;
};

class CPUSummator : public Summator {
public:
  CPUSummator() = default;
  // sum[i] = y[i] + x[i]
  void Add(float* x, float* y, float* sum, int num_elems);
  void Add(Half* x, Half* y, Half* sum, int num_elems);
  int num_threads_ = 4;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_VECTOR_OPERATIONS_H
