#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

//#define CurandState curandState
//#define CurandState curandStatePhilox4_32_10_t
#define CurandState int

int CUDA_get_curand_array_size(int num_elems);

void CUDA_init_curand(CurandState* states, int num_elems, unsigned int seed,
                      cudaStream_t stream);
void CUDA_add(int n, const float* x, float* y, float* sum, cudaStream_t stream);

void CUDA_quantize_maxmin(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback, int num_elems, int bits,
                          int bucket_size, CurandState* states,
                          cudaStream_t stream);
void CUDA_dequantize_maxmin(unsigned char* input_data,
                            unsigned char* output_data, int num_elems, int bits,
                            int bucket_size, cudaStream_t stream);
void CUDA_quantize_LinfNorm(unsigned char* input_data,
                            unsigned char* output_data, unsigned char* feedback,
                            float* levels, int num_elems, int bits,
                            int bucket_size, CurandState* states,
                            cudaStream_t stream);
void CUDA_dequantize_LinfNorm(unsigned char* input_data,
                              unsigned char* output_data, float* levels,
                              int num_elems, int bits, int bucket_size,
                              cudaStream_t stream);
void CUDA_quantize_L2Norm(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback, float* levels, int num_elems,
                          int bits, int bucket_size, CurandState* states,
                          cudaStream_t stream);
void CUDA_dequantize_L2Norm(unsigned char* input_data,
                            unsigned char* output_data, float* levels,
                            int num_elems, int bits, int bucket_size,
                            cudaStream_t stream);
#endif // CUDA_FUNCTIONS_H_