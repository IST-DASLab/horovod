#ifndef HOROVOD_TEST_CUDA_DEF_H
#define HOROVOD_TEST_CUDA_DEF_H
#include <stdint.h>
#include "common.h"

#define Half __half

enum NormType { L2, Linf };
enum LevelsType { Pos, Wide };
struct xorshift128p_state {
  uint64_t a, b;
};

//#define CurandState int
#define CurandState xorshift128p_state

const float EPS = 1e-10;
const int PACK_SIZE = 8;
const int MAX_THREADS_PER_BLOCK = 32;
const int MAX_NUMBER_OF_BLOCKS = 65535;
const int WARP_SIZE = 32;

static int blocks_per_grid = 0;
static void set_blocks(int blocks) { blocks_per_grid = blocks; }

constexpr int MIN(int a, int b) { return (a > b) ? b : a; }

#define BLOCKS_PER_GRID(num_elems)                                             \
  MIN((blocks_per_grid == 0)                                                   \
          ? (num_elems + (MAX_THREADS_PER_BLOCK - 1)) / MAX_THREADS_PER_BLOCK  \
          : blocks_per_grid,                                                   \
      MAX_NUMBER_OF_BLOCKS)

#endif // HOROVOD_TEST_CUDA_DEF_H
