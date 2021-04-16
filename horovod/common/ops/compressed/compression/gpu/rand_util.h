#ifndef GPU_RAND_UTIL_H_
#define GPU_RAND_UTIL_H_

namespace horovod {
namespace common {
namespace gpu {

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

} // namespace gpu
} // namespace common
} // namespace horovod
#endif // GPU_RAND_UTIL_H_
