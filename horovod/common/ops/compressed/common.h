#ifndef COMPRESSED_COMMON_H
#define COMPRESSED_COMMON_H

#if HAVE_CUDA
#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t cuda_result = condition;                                       \
    if (cuda_result != cudaSuccess) {                                          \
      fprintf(stderr, "%s on line %i in %s returned: %s(code:%i)\n",           \
              #condition, __LINE__, __FILE__, cudaGetErrorString(cuda_result), \
              cuda_result);                                                    \
      throw std::runtime_error(                                                \
          std::string(#condition) + " on line " + std::to_string(__LINE__) +   \
          " returned: " + cudaGetErrorString(cuda_result));                    \
    }                                                                          \
  } while (0)
#elif HAVE_ROCM
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
              error, __FILE__, __LINE__);                                      \
      throw std::runtime_error(std::string(#cmd) + " on line " +               \
                               std::to_string(__LINE__) +                      \
                               " returned: " + hipGetErrorString(error));      \
    }                                                                          \
  } while (0)
#endif

namespace horovod {
namespace common {
enum CommunicatorType {
  MPI,
  NCCL,
  SHM,
  P2P
};

enum ReductionType {
  AllGather,
  ScatterAllgather,
  Ring,
  PS,
  Tree,
  NoneReduction
};

enum CompressionType {
  MaxMin,
  Uni,
  Exp,
  TopK,
  NoneCompression
};

enum NormType { L2, Linf };
enum LevelsType { Pos, Wide };

enum CompressionMode {
  NonFused,
  PerEntryFused,
  Fused
};

enum CompressFunc { MaxMinWide, NormWide, NormPos };

const int BUFFER_THRESHOLD = 1000;
const float QUANTIZE_MULTIPLIER = 0.5;

} // namespace common
} // namespace horovod

#endif // COMPRESSED_COMMON_H
