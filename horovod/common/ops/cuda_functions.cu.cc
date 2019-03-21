#include "cuda_functions.h"
#include <cstdlib>
#include <iostream>

#define maxThreadsPerBlock 1024

__global__ void _init_curand(unsigned int seed, curandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              index, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[index]);
}

__global__ void _add(int n, float* x, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];              
  }
}

__global__ void _find_max_and_min(float* array, float* maxandmin, int n) {
/*  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index != 0) {
    return;
  }
  for (int i = 0; i < n; i += 512) {
    float maxf = x[i];
    float minf = x[i];
    for (int j = 0; j < 512; j++) {
      if (i + j < n) {
        maxf = fmaxf(maxf, x[i + j]);
        minf = fminf(minf, x[i + j]);
      }
    }
    maxandmin[i / 512 * 2] = maxf;
    maxandmin[i / 512 * 2 + 1] = minf;
  }*/
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  __shared__ float cache1[maxThreadsPerBlock];
  __shared__ float cache2[maxThreadsPerBlock];

  for (int j = index; j < n; j += stride) {
    int my_bucket = j / 512;
    int index_in_bucket = j % 512;
    int offset = (my_bucket&1) ? 512 : 0;

    // reduction
    unsigned int i = 512 / 2;
    while (i != 0) {
      if (index_in_bucket < i) {
        if (i == 512 / 2) { //get data in cache in first loop
          cache1[index_in_bucket + offset] = j + i < n ? array[j] : fmaxf(array[j], array[j + i]);
          cache2[index_in_bucket + offset] = j + i < n ? array[j] : fminf(array[j], array[j + i]);                 
        } else {
          if (index_in_bucket + offset + i < n) {
            cache1[index_in_bucket + offset] = fmaxf(cache1[index_in_bucket + offset], cache1[index_in_bucket + offset + i]);
            cache2[index_in_bucket + offset] = fminf(cache2[index_in_bucket + offset], cache2[index_in_bucket + offset + i]);
          }
        }
      }
      __syncthreads();
      i /= 2;
    }
    if (threadIdx.x == 0) {
      maxandmin[2 * my_bucket] = cache1[0];
      maxandmin[2 * my_bucket + 1] = cache2[0];
    } else if (threadIdx.x == 512) {
      maxandmin[2 * my_bucket] = cache1[512];
      maxandmin[2 * my_bucket + 1] = cache2[512];
    }
  }
}

__global__ void _find_max_and_min_bucket(float* x, float* maxandmin, int n, const int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  __shared__ float cache1[maxThreadsPerBlock];
  __shared__ float cache2[maxThreadsPerBlock];

  for (int j = index; j < n; j += stride) {
    int my_bucket = j / bucket_size;
    int index_in_bucket = j & (bucket_size - 1);
    int offset = (my_bucket & (maxThreadsPerBlock / bucket_size - 1)) * bucket_size;
 
    unsigned int i = bucket_size / 2;
    while (i > 0) {
      if (index_in_bucket < i) {
        if (i == bucket_size / 2) {
          if (j + i < n) {
            cache1[index_in_bucket + offset] = fmaxf(x[j], x[j + i]);
            cache2[index_in_bucket + offset] = fminf(x[j], x[j + i]);
          } else {
            cache1[index_in_bucket + offset] = x[j];
            cache2[index_in_bucket + offset] = x[j];
          }
        } else {
          if (index_in_bucket + offset + i < n) {
            cache1[index_in_bucket + offset] = fmaxf(cache1[index_in_bucket + offset], cache1[index_in_bucket + offset + i]);
            cache2[index_in_bucket + offset] = fminf(cache2[index_in_bucket + offset], cache2[index_in_bucket + offset + i]);
          }
        }
      }
      __syncthreads();
      i /= 2;
    }

    if ((threadIdx.x & (bucket_size - 1)) == 0) {
      maxandmin[2 * my_bucket] = cache1[offset];
      maxandmin[2 * my_bucket + 1] = cache2[offset];
    }
  }
}

__global__ void _quantize_value_bits(unsigned char* x, const float* y, const float* maxandmin, const int n, const int bits, const int bucket_size, curandState* states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  curandState local_state;
  local_state = states[index];

  int parts = 8 / bits;
  int divisor = 1 << bits;

  for (int i = index; i < (n + parts - 1) / parts; i += stride) {
    int a = 0;
    for (int j = 0; j < parts && i * parts + j < n; j++) {
      int my_bucket = (i * parts + j) / bucket_size;
      float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) / (divisor - 1);
      float d = (y[i * parts + j] - maxandmin[my_bucket * 2 + 1]) / unit + (curand(&local_state) % 1000001 / 1000000.0); 
      a += ((int)floor(d)) << (j * bits);
    }
    x[i] = (unsigned char) a;
  }
  states[index] = local_state;       
}

__global__ void _dequantize_value_bits(unsigned char* recv, float* maxandmin, float* x, const int n, const int bits, const int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  int parts = 8 / bits;
  int divisor = 1 << bits;

  for (int i = index; i < n; i += stride) {
    int my_bucket = i / bucket_size;
    float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) / (divisor - 1);
    x[i] = maxandmin[my_bucket * 2 + 1] + ((recv[i / parts] >> ((i % parts) * bits)) & (divisor - 1)) * unit;
  }
}

__global__ void _copy_value(float* x, const float* y, const int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = y[i];
  }
}

__global__ void _print(float* x, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    printf("[");
    for (int i = 0; i < 10; i++) {
      printf("%f,", x[i]);
    }
    printf("]\n");
  }
}

curandState* GPU_init_curand(int n, unsigned int seed, cudaStream_t stream) {
  curandState* states;

  int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
  cudaMalloc(&states, blocksPerGrid * maxThreadsPerBlock * sizeof(curandState));

  _init_curand<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(seed, states);

  return states;    
}

void GPU_add(int n, float* x, float* y, cudaStream_t stream) {
  int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
  _add<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(n, x, y);
  cudaStreamSynchronize(stream);	    
}

void GPU_find_max_and_min_bucket(float* x, float* maxandmin, int n, int bucket_size, cudaStream_t stream) {
  int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
  _find_max_and_min_bucket<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, maxandmin, n, bucket_size);
//  _find_max_and_min<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, maxandmin, n);
  cudaStreamSynchronize(stream);
}

void GPU_quantize_value_bits(unsigned char* x, float* y, float* maxandmin, int n, int bits, int bucket_size, curandState* states, cudaStream_t stream) {
  int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock / (8 / bits));
  _quantize_value_bits<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, y, maxandmin, n, bits, bucket_size, states);
  cudaStreamSynchronize(stream);
}

void GPU_dequantize_value_bits(unsigned char* recv, float* maxandmin, float* x, int n, int bits, int bucket_size, cudaStream_t stream) {
  int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
  _dequantize_value_bits<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(recv, maxandmin, x, n, bits, bucket_size);
  cudaStreamSynchronize(stream);
}

void GPU_copy_value(float* x, float* y, int n, cudaStream_t stream) {
  int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
  _copy_value<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, y, n);
  cudaStreamSynchronize(stream);
}

void GPU_print(float* x, int n, cudaStream_t stream) {
  _print<<<1, 1, 0, stream>>>(x, n);
  cudaStreamSynchronize(stream);
}
