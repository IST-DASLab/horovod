#ifndef HOROVOD_CUDA_RAND_H
#define HOROVOD_CUDA_RAND_H

#include "gpu_def.h"
#include "rand_util.h"
#include <climits>

namespace horovod {
namespace common {
namespace gpu {


__global__ void _init_curand(unsigned int seed, GPURandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
//    unsigned char z[4];
//    for (int i = 0; i < 4; i++)
//      z[i] = (seed + index) % 128;
//    states[index] = toInt(z);
  states[index] = xorshift128_init(seed * index);
//  curand_init(seed, index, 1000, &states[index]);
}


__device__ float GetRand(GPURandState* state_p) {
//  return curand_uniform(state_p);
  return ((float)xorshift128p(state_p)) / UINT64_MAX;
//    return HybridTaus(state_p);
}

void CUDA_init_curand(GPURandState* states, int num_elems, unsigned int seed,
                      cudaStream_t stream) {
  _init_curand<<<BLOCKS_PER_GRID(num_elems, THREADS_PER_BLOCK_COMPRESS), THREADS_PER_BLOCK_COMPRESS, 0,
                 stream>>>(seed, states);
}

} // namespace gpu
} // namespace common
} // namespace horovod
#endif // HOROVOD_CUDA_RAND_H