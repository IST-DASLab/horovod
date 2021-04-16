#ifndef HOROVOD_TEST_CUDA_DEF_H
#define HOROVOD_TEST_CUDA_DEF_H
#include <stdexcept>
#include <stdint.h>
#include <string>
#include "../../common.h"

#if HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#elif HAVE_ROCM
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#define Half __half
//#define GPURandState int
#define GPURandState horovod::common::gpu::xorshift128p_state
//#define GPURandState curandState

namespace horovod {
namespace common {
namespace gpu {

struct xorshift128p_state {
  uint64_t a, b;
};

const float EPS = 1e-10;
const int PACK_SIZE = 8;
const int MAX_THREADS_PER_BLOCK = 1024;
const int THREADS_PER_BLOCK_DECOMPRESS = MAX_THREADS_PER_BLOCK;
const int THREADS_PER_BLOCK_COMPRESS = 128;
const int MAX_NUMBER_OF_BLOCKS = 65535;
const int WARP_SIZE = 32;

constexpr int MIN(int a, int b) { return (a > b) ? b : a; }

constexpr int BLOCKS_PER_GRID(int num_elems, int threads_per_block) {
  return MIN((num_elems + (threads_per_block - 1)) / threads_per_block,
             MAX_NUMBER_OF_BLOCKS);
}
#if HAVE_CUDA
using gpuStream_t = cudaStream_t;
#elif HAVE_ROCM
using gpuStream_t = hipStream_t;
#endif

} // namespace gpu
} // namespace common
} // namespace horovod

#endif // HOROVOD_TEST_CUDA_DEF_H
