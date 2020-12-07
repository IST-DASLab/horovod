#include <climits>
#include "cuda_def.h"

namespace horovod {
namespace common {
namespace cuda {

__device__ int toInt(unsigned char* z) {
  return ((unsigned int)z[0] & 0xFF) << 24 | ((unsigned int)z[1] & 0xFF) << 16 |
         ((unsigned int)z[2] & 0xFF) << 8 | ((unsigned int)z[3] & 0xFF);
}

inline __device__ void TausStep(unsigned char& z, unsigned int S1,
                                unsigned int S2, int S3, unsigned M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  z = (((z & M) << S3) ^ b);
}

inline __device__ void LCGStep(unsigned char& z, unsigned A, unsigned C) {
  z = (A * z + C);
}

inline __device__ float HybridTaus(int* state_p) {
  unsigned char z[4];
  int state = *state_p;
  // present int as char array.
  z[0] = (state >> 24) & 0xFF;
  z[1] = (state >> 16) & 0xFF;
  z[2] = (state >> 8) & 0xFF;
  z[3] = state & 0xFF;
  TausStep(z[0], 13, 19, 12, 4294967294UL);
  TausStep(z[1], 2, 25, 4, 4294967288UL);
  TausStep(z[2], 3, 11, 17, 4294967280UL);
  LCGStep(z[3], 1664525, 1013904223UL);
  *state_p = toInt(z);
  return (z[0] ^ z[1] ^ z[2] ^ z[3]) / 256.0;
}

inline __device__ uint64_t splitmix64(uint64_t* seed) {
  uint64_t result = *seed;

  *seed = result + 0x9E3779B97f4A7C15;
  result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
  result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
  return result ^ (result >> 31);
}

inline __device__ xorshift128p_state xorshift128_init(uint64_t seed) {
  xorshift128p_state result;
  uint64_t tmp = splitmix64(&seed);
  result.a = tmp;
  tmp = splitmix64(&seed);
  result.b = tmp;
  return result;
}

__global__ void _init_curand(unsigned int seed, CurandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
//    unsigned char z[4];
//    for (int i = 0; i < 4; i++)
//      z[i] = (seed + index) % 128;
//    states[index] = toInt(z);
  states[index] = xorshift128_init(seed * index);
//  curand_init(seed, index, 0, &states[index]);
}

inline __device__ float xorshift128p(xorshift128p_state* state) {
  uint64_t t = state->a;
  uint64_t s = state->b;
  state->a = s;
  t ^= t << 23;       // a
  t ^= t >> 17;       // b
  t ^= s ^ (s >> 26); // c
  state->b = t;
  return (t + s) * 1.0;
}

__device__ float GetRand(CurandState* state_p) {
//  return curand_uniform(state_p);
  return ((float)xorshift128p(state_p)) / UINT64_MAX;
//    return HybridTaus(state_p);
}

void CUDA_init_curand(CurandState* states, int num_elems, unsigned int seed,
                      cudaStream_t stream) {
  _init_curand<<<BLOCKS_PER_GRID(num_elems, THREADS_PER_BLOCK_COMPRESS), THREADS_PER_BLOCK_COMPRESS, 0,
                 stream>>>(seed, states);
}

int CUDA_get_curand_array_size(int num_elems) {
  return BLOCKS_PER_GRID(num_elems, THREADS_PER_BLOCK_COMPRESS) * THREADS_PER_BLOCK_COMPRESS *
         sizeof(CurandState);
}

} // namespace cuda
} // namespace common
} // namespace horovod