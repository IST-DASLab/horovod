#ifndef HOROVOD_TEST_CUDA_DEF_H
#define HOROVOD_TEST_CUDA_DEF_H
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <stdint.h>
#include <string>

#define Half __half
//#define CurandState int
#define CurandState horovod::common::cuda::xorshift128p_state
//#define CurandState curandState

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t cuda_result = condition;                                       \
    if (cuda_result != cudaSuccess) {                                          \
      printf("%s on line %i in %s returned: %s(code:%i)\n", #condition,        \
             __LINE__, __FILE__, cudaGetErrorString(cuda_result),              \
             cuda_result);                                                     \
      throw std::runtime_error(                                                \
          std::string(#condition) + " on line " + std::to_string(__LINE__) +   \
          " returned: " + cudaGetErrorString(cuda_result));                    \
    }                                                                          \
  } while (0)

namespace horovod {
namespace common {
namespace cuda {

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

typedef union {
  float4 vec;
  float a[4];
} F4;

typedef union {
  uchar2 vec;
  unsigned char a[2];
} U2;

typedef union {
  uchar3 vec;
  unsigned char a[3];
} U3;

typedef union {
  uchar4 vec;
  unsigned char a[4];
} U4;

} // namespace cuda
} // namespace common
} // namespace horovod

#endif // HOROVOD_TEST_CUDA_DEF_H
