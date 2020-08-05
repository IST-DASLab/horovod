#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "../../../../common.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define Half __half

struct xorshift128p_state {
  uint64_t a, b;
};

#define CurandState int
//#define CurandState xorshift128p_state

int CUDA_get_curand_array_size(int num_elems);
void CUDA_init_curand(CurandState* states, int num_elems, unsigned int seed,
                      cudaStream_t stream);

void CUDA_add_fp32(int n, const float* x, float* y, float* sum, cudaStream_t stream);
void CUDA_add_fp16(int n, const Half* x, Half* y, Half* sum, cudaStream_t stream);

void CUDA_quantize_maxmin_fp16(unsigned char* input_data,
                               unsigned char* output_data,
                               unsigned char* feedback, int num_elems, int bits,
                               int bucket_size, CurandState* states,
                               cudaStream_t stream);

void CUDA_quantize_maxmin_fp32(unsigned char* input_data,
                               unsigned char* output_data,
                               unsigned char* feedback, int num_elems, int bits,
                               int bucket_size, CurandState* states,
                               cudaStream_t stream);

void CUDA_dequantize_maxmin_fp16(unsigned char* input_data,
                                 unsigned char* output_data, int num_elems,
                                 int bits, int bucket_size, bool add,
                                 cudaStream_t stream);
void CUDA_dequantize_maxmin_fp32(unsigned char* input_data,
                                 unsigned char* output_data, int num_elems,
                                 int bits, int bucket_size, bool add,
                                 cudaStream_t stream);

void CUDA_quantize_Norm_fp32(
    unsigned char* input_data, unsigned char* output_data,
    unsigned char* feedback, float* levels, int num_elems, int bits,
    int bucket_size, CurandState* states, horovod::common::NormType norm_type,
    horovod::common::LevelsType level_type, cudaStream_t stream);

void CUDA_quantize_Norm_fp16(
    unsigned char* input_data, unsigned char* output_data,
    unsigned char* feedback, Half* levels, int num_elems, int bits,
    int bucket_size, CurandState* states, horovod::common::NormType norm_type,
    horovod::common::LevelsType level_type, cudaStream_t stream);
void CUDA_dequantize_Norm_fp32(unsigned char* input_data,
                               unsigned char* output_data, float* levels,
                               int num_elems, int bits, int bucket_size,
                               horovod::common::LevelsType level_type, bool add,
                               cudaStream_t stream);

void CUDA_dequantize_Norm_fp16(unsigned char* input_data,
                               unsigned char* output_data, Half* levels,
                               int num_elems, int bits, int bucket_size,
                               horovod::common::LevelsType level_type, bool add,
                               cudaStream_t stream);

void CUDA_convert_to_halves(float* arr, Half* output, int numel);

#endif // CUDA_FUNCTIONS_H