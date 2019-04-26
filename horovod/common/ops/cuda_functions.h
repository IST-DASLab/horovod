#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_


void GPU_init_curand(curandState* states, int num_elems, unsigned int seed, cudaStream_t stream);
int GPU_get_curand_array_size(int num_elems);
void GPU_add(int n, float *x, float *y, cudaStream_t stream);
void GPU_find_max_and_min_bucket(float *array, float *maxandmin, int n, int bucket_size, cudaStream_t stream);
void GPU_quantize_value_bits(unsigned char* x, float* y, float* maxandmin, int n, int bits, int bucket_size, curandState* states, cudaStream_t stream);
void GPU_dequantize_value_bits(unsigned char* recv, float* maxandmin, float* x, int n, int bits, int bucket_size, cudaStream_t stream);
void GPU_copy_value(float* x, float* y, int n, cudaStream_t stream);
void GPU_print(float* x, int n, cudaStream_t stream);


#endif //CUDA_FUNCTIONS_H_