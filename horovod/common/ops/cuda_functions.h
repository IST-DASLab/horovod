#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

//#define CurandState curandState
#define CurandState curandStatePhilox4_32_10_t
//#define CurandState int

void CUDA_init_curand(CurandState* states, int num_elems, unsigned int seed,
                      cudaStream_t stream);
int CUDA_get_curand_array_size(int num_elems);
// y += x.
void CUDA_add(int n, const float* x, float* y, cudaStream_t stream);
// y -= x
void CUDA_substract(int n, const float* x, float* y, cudaStream_t stream);
void CUDA_find_max_and_min_bucket(const float* x, float* maxandmin, int n,
                                  int bucket_size, cudaStream_t stream);
void CUDA_find_Linf_bucket(const float* x, float* maxs, int n, int bucket_size,
                           cudaStream_t stream);
void CUDA_find_norms_bucket(const float* x, float* max, float* norm, int n,
                            int bucket_size, cudaStream_t stream);
void CUDA_find_L2_and_max_log_bucket(const float* x, float* norm, unsigned char* max_log, float rev_multiplier, int n,
                                     int bucket_size, cudaStream_t stream);

void CUDA_quantize_value_bits(unsigned char* y, const float* x,
                              const float* maxandmin, int n, int bits,
                              int bucket_size, CurandState* states,
                              cudaStream_t stream);
void CUDA_dequantize_value_bits(const unsigned char* y, const float* maxandmin,
                                float* x, int n, int bits, int bucket_size,
                                cudaStream_t stream);
void CUDA_Linf_normalized_quantize_values(unsigned char* y, const float* x,
                                          const float* norms, const float* levels,
                                          int n, int bits, int bucket_size,
                                          CurandState* states, cudaStream_t stream);
void CUDA_Linf_normalized_dequantize_values(const unsigned char* y,
                                            const float* norms, const float* levels,
                                            float* x, int n, int bits,
                                            int bucket_size, cudaStream_t stream);
void CUDA_L2_normalized_quantize_values(unsigned char* y, const float* x,
                                          const float* norms, const unsigned char* max_log, const float* levels,
                                          int n, int bits, int bucket_size,
                                          CurandState* states, cudaStream_t stream);
void CUDA_L2_normalized_dequantize_values(const unsigned char* y,
                                            const float* norms, const unsigned char* max_log, const float* levels,
                                            float* x, int n, int bits,
                                            int bucket_size, cudaStream_t stream);

void GPU_copy_value(float* x, float* y, int n, cudaStream_t stream);
void GPU_print(float* x, int n, cudaStream_t stream);

#endif // CUDA_FUNCTIONS_H_