#include "cuda_functions.h"
#include <cstdlib>
#include <iostream>

#define maxThreadsPerBlock 1024


__device__
int toInt(unsigned char* z) {
  return ((unsigned int) z[0] & 0xFF) << 24 |
         ((unsigned int) z[1] & 0xFF) << 16 |
         ((unsigned int) z[2] & 0xFF) << 8 |
         ((unsigned int) z[3] & 0xFF);
}

__global__ void _init_curand(unsigned int seed, CurandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  /* we have to initialize the state */
  curand_init(seed,  /* the seed can be the same for each core, here we pass the
                        time in from the CPU */
              index, /* the sequence number should be different for each core
                        (unless you want all cores to get the same sequence of
                        numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for
                    each call, can be 0 */
              &states[index]);

//  unsigned char z[4];
//  for  (int i = 0; i < 4; i++)
//    z[i] = 128 + index % 128;
//  states[index] = toInt(z);
}

__global__ void _add(int n, float* x, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

__global__ void _find_max_and_min_bucket_seq(float* x, float* maxandmin, int n,
                                             const int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (int i = index; i < (n + bucket_size - 1) / bucket_size; i += stride) {
    float mmin = x[i * bucket_size];
    float mmax = x[i * bucket_size];
    for (int j = i * bucket_size; j < fminf((i + 1) * bucket_size, n); j++) {
      mmin = fminf(mmin, x[j]);
      mmax = fmaxf(mmax, x[j]);
    }
    maxandmin[2 * i] = mmax;
    maxandmin[2 * i + 1] = mmin;
  }
}

__device__
void TausStep(unsigned char &z, unsigned int S1, unsigned int S2, int S3, unsigned M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  z = (((z & M) << S3) ^ b);
}

__device__
void LCGStep(unsigned char &z, unsigned A, unsigned C)
{
  z = (A * z + C);
}

__device__
float HybridTaus(int& state) {
  unsigned char z[4];
  // present int as char array.
  z[0] = (state >> 24) & 0xFF;
  z[1] = (state >> 16) & 0xFF;
  z[2] = (state >> 8) & 0xFF;
  z[3] = state & 0xFF;
  TausStep(z[0], 13, 19, 12, 4294967294UL);
  TausStep(z[1], 2, 25, 4, 4294967288UL);
  TausStep(z[2],  3, 11, 17, 4294967280UL);
  LCGStep(z[3], 1664525, 1013904223UL);
  state = toInt(z);
  return (z[0] ^ z[1] ^ z[2] ^ z[3]) / 256.0;
}

__global__ void _quantize_value_bits(unsigned char* x, const float* y,
                                     const float* maxandmin, const int n,
                                     const int bits, const int bucket_size,
                                     CurandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  CurandState local_state;
  local_state = states[index];

  int parts = 8 / bits;
  int divisor = 1 << bits;

  for (int i = index; i < (n + parts - 1) / parts; i += stride) {
    int a = 0;
    for (int j = 0; j < parts && i * parts + j < n; j++) {
      int my_bucket = (i * parts + j) / bucket_size;
      float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) /
                   (divisor - 1);
      float d = (y[i * parts + j] - maxandmin[my_bucket * 2 + 1]) / unit
//               + (curand(&local_state) % 100001) / 100000.0;
                 + curand_uniform(&local_state);
      a += ((int)floor(d)) << (j * bits);
    }
    x[i] = (unsigned char)a;
  }
  states[index] = local_state;
}

__global__ void _dequantize_value_bits(unsigned char* recv, float* maxandmin,
                                       float* x, const int n, const int bits,
                                       const int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  int parts = 8 / bits;
  int divisor = 1 << bits;

  for (int i = index; i < n; i += stride) {
    int my_bucket = i / bucket_size;
    float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) /
                 (divisor - 1);
    x[i] = maxandmin[my_bucket * 2 + 1] +
           ((recv[i / parts] >> ((i % parts) * bits)) & (divisor - 1)) * unit;
  }
}

#define BLOCKS_PER_GRID(n) (n + (maxThreadsPerBlock - 1)) / maxThreadsPerBlock

void GPU_init_curand(CurandState* states, int num_elems, unsigned int seed,
                     cudaStream_t stream) {
  _init_curand<<<BLOCKS_PER_GRID(num_elems), maxThreadsPerBlock, 0, stream>>>(
      seed, states);
}

int GPU_get_curand_array_size(int num_elems) {
  return BLOCKS_PER_GRID(num_elems) * maxThreadsPerBlock * sizeof(CurandState);
}

void GPU_add(int n, float* x, float* y, cudaStream_t stream) {
  _add<<<BLOCKS_PER_GRID(n), maxThreadsPerBlock, 0, stream>>>(n, x, y);
  cudaStreamSynchronize(stream);
}

void GPU_find_max_and_min_bucket(float* x, float* maxandmin, int n,
                                 int bucket_size, cudaStream_t stream) {
  _find_max_and_min_bucket_seq<<<BLOCKS_PER_GRID(n), maxThreadsPerBlock, 0,
                                 stream>>>(x, maxandmin, n, bucket_size);
  //  _find_max_and_min<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x,
  //  maxandmin, n);
  cudaStreamSynchronize(stream);
}

void GPU_quantize_value_bits(unsigned char* x, float* y, float* maxandmin,
                             int n, int bits, int bucket_size,
                             CurandState* states, cudaStream_t stream) {
  _quantize_value_bits<<<BLOCKS_PER_GRID(n), maxThreadsPerBlock, 0, stream>>>(
      x, y, maxandmin, n, bits, bucket_size, states);
  cudaStreamSynchronize(stream);
}

void GPU_dequantize_value_bits(unsigned char* recv, float* maxandmin, float* x,
                               int n, int bits, int bucket_size,
                               cudaStream_t stream) {
  _dequantize_value_bits<<<BLOCKS_PER_GRID(n), maxThreadsPerBlock, 0, stream>>>(
      recv, maxandmin, x, n, bits, bucket_size);
  cudaStreamSynchronize(stream);
}
