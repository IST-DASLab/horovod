#ifndef HOROVOD_CUDA_VECTOR_OPERATIONS_H
#define HOROVOD_CUDA_ARITHMETIC_FUNCTIONS_H
#include "gpu_def.h"

namespace horovod {
namespace common {
namespace gpu {

template <typename T>
void CUDA_add(int n, const T* x, T* y, T* sum, cudaStream_t stream);

} // namespace gpu
} // namespace common
} // namespace horovod
#endif // HOROVOD_CUDA_VECTOR_OPERATIONS_H
