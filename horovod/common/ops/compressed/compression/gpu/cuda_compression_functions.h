#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_def.h"

namespace horovod {
namespace common {
namespace gpu {

template <typename T>
void CUDA_quantize_maxmin(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback_data, int num_elems, int bits,
                          int bucket_size, GPURandState* states,
                          cudaStream_t stream);

template <typename T, bool ADD>
void CUDA_dequantize_maxmin(unsigned char* input_data,
                            unsigned char* output_data, int num_elems, int bits,
                            int bucket_size, cudaStream_t stream);

template <typename T>
void CUDA_quantize_Norm(unsigned char* input_data, unsigned char* output_data,
                        unsigned char* feedback, T* levels, int num_elems,
                        int bits, int bucket_size, GPURandState* states,
                        NormType norm_type, LevelsType levels_type,
                        cudaStream_t stream);

template <typename T, bool ADD>
void CUDA_dequantize_Norm(unsigned char* input_data, unsigned char* output_data,
                          T* levels, int num_elems, int bits, int bucket_size,
                          LevelsType levels_type, cudaStream_t stream);

void CUDA_convert_to_halves(float* input, Half* output, int numel, cudaStream_t stream);

void CUDA_init_curand(GPURandState* states, int num_elems, unsigned int seed,
                      cudaStream_t stream);

template <typename T>
void CUDA_topk_compress(unsigned char* input_data, unsigned char* output_data,
                        unsigned char* utility_buf, unsigned char* feedback,
                        int num_elems,
                        int num_result, cudaStream_t stream);

template <typename T, bool ADD>
void CUDA_topk_decompress(unsigned char* input_data, unsigned char* output_data,
                          int num_elems, int num_result, cudaStream_t stream);

} // namespace gpu
} // namespace common
} // namespace horovod
#endif // CUDA_FUNCTIONS_H