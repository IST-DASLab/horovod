#ifndef HOROVOD_CUDA_VECTOR_OPERATIONS_H
#define HOROVOD_CUDA_ARITHMETIC_FUNCTIONS_H
#include "cuda_def.h"

namespace horovod {
namespace common {
namespace cuda {

template <typename T>
void CUDA_add(int64_t n, const T* x, T* y, T* sum, cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace horovod
#endif // HOROVOD_CUDA_VECTOR_OPERATIONS_H
