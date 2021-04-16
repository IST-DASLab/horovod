#ifndef GPU_COMPRESSION_FUNCTIONS
#define GPU_COMPRESSION_FUNCTIONS
#include "gpu_def.h"

#if HAVE_CUDA
#include "cuda_compression_functions.h"
#elif HAVE_ROCM
#include "hip_compression_functions.h"
#endif

namespace horovod {
namespace common {
namespace gpu {

size_t GPU_get_curand_array_size(int num_elems);

template <typename T>
void GPU_quantize_maxmin(unsigned char* input_data, unsigned char* output_data,
                         unsigned char* feedback_data, int num_elems, int bits,
                         int bucket_size, GPURandState* states,
                         gpuStream_t stream);

template <typename T, bool ADD>
void GPU_dequantize_maxmin(unsigned char* input_data,
                           unsigned char* output_data, int num_elems, int bits,
                           int bucket_size, gpuStream_t stream);

template <typename T>
void GPU_quantize_Norm(unsigned char* input_data, unsigned char* output_data,
                       unsigned char* feedback, T* levels, int num_elems,
                       int bits, int bucket_size, GPURandState* states,
                       NormType norm_type, LevelsType levels_type,
                       gpuStream_t stream);

template <typename T, bool ADD>
void GPU_dequantize_Norm(unsigned char* input_data, unsigned char* output_data,
                         T* levels, int num_elems, int bits, int bucket_size,
                         LevelsType levels_type, gpuStream_t stream);

void GPU_convert_to_halves(float* input, Half* output, int numel,
                           gpuStream_t stream);

void GPU_init_curand(GPURandState* states, int num_elems, unsigned int seed,
                     gpuStream_t stream);

size_t GPU_get_topk_utility_buf_size();

template <typename T>
void GPU_topk_compress(unsigned char* input_data, unsigned char* output_data,
                       unsigned char* utility_buf, unsigned char* feedback,
                       int num_elems, int num_result, gpuStream_t stream);
template <typename T, bool ADD>
void GPU_topk_decompress(unsigned char* input_data, unsigned char* output_data,
                         int num_elems, int num_result, gpuStream_t stream);

template <typename T>
void GPU_add(int n, const T* x, T* y, T* sum, gpuStream_t stream);

} // namespace gpu
} // namespace common
} // namespace horovod
#endif // GPU_COMPRESSION_FUNCTIONS
