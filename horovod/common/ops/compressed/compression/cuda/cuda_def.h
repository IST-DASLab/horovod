#ifndef HOROVOD_TEST_CUDA_DEF_H
#define HOROVOD_TEST_CUDA_DEF_H
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string>
#include <stdexcept>

#define Half __half
//#define CurandState int
#define CurandState horovod::common::cuda::xorshift128p_state

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t cuda_result = condition;                                       \
    if (cuda_result != cudaSuccess) {                                          \
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


constexpr int BLOCKS_PER_GRID(int num_elems) {
  return MIN((num_elems + (MAX_THREADS_PER_BLOCK - 1)) / MAX_THREADS_PER_BLOCK,
             MAX_NUMBER_OF_BLOCKS);
}

typedef union {
  float4 vec;
  float a[4];
} F4;

typedef union {
  uchar4 vec;
  unsigned char a[4];
} U4;


} // namespace cuda
} // namespace common
} // namespace horovod


#endif // HOROVOD_TEST_CUDA_DEF_H
