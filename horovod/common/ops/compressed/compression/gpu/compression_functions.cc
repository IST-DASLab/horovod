#include "compression_functions.h"
#if HAVE_CUDA
#include "cuda_arithmetic_functions.h"
#include "cuda_compression_functions.h"
#elif HAVE_ROCM
#include "hip_compression_functions.h"
#endif

namespace horovod {
namespace common {
namespace gpu {

size_t GPU_get_curand_array_size(int num_elems) {
  return BLOCKS_PER_GRID(num_elems, THREADS_PER_BLOCK_COMPRESS) *
         THREADS_PER_BLOCK_COMPRESS * sizeof(GPURandState);
}

template <typename T>
void GPU_quantize_maxmin(unsigned char* input_data, unsigned char* output_data,
                         unsigned char* feedback_data, int num_elems, int bits,
                         int bucket_size, GPURandState* states,
                         gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_quantize_maxmin<T>(input_data, output_data, feedback_data, num_elems,
                          bits, bucket_size, states, stream);
#elif HAVE_ROCM
  HIP_quantize_maxmin<T>(input_data, output_data, feedback_data, num_elems,
                         bits, bucket_size, states, stream);
#endif
}

template <typename T, bool ADD>
void GPU_dequantize_maxmin(unsigned char* input_data,
                           unsigned char* output_data, int num_elems, int bits,
                           int bucket_size, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_dequantize_maxmin<T, ADD>(input_data, output_data, num_elems, bits,
                                 bucket_size, stream);
#elif HAVE_ROCM
  HIP_dequantize_maxmin<T, ADD>(input_data, output_data, num_elems, bits,
                                bucket_size, stream);
#endif
}

template <typename T>
void GPU_quantize_Norm(unsigned char* input_data, unsigned char* output_data,
                       unsigned char* feedback, T* levels, int num_elems,
                       int bits, int bucket_size, GPURandState* states,
                       NormType norm_type, LevelsType levels_type,
                       gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_quantize_Norm<T>(input_data, output_data, feedback, levels, num_elems,
                        bits, bucket_size, states, norm_type, levels_type,
                        stream);
#elif HAVE_ROCM
#endif
}

template <typename T, bool ADD>
void GPU_dequantize_Norm(unsigned char* input_data, unsigned char* output_data,
                         T* levels, int num_elems, int bits, int bucket_size,
                         LevelsType levels_type, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_dequantize_Norm<T, ADD>(input_data, output_data, levels, num_elems, bits,
                               bucket_size, levels_type,
                               stream);
#elif HAVE_ROCM
#endif
}

void GPU_convert_to_halves(float* input, Half* output, int numel,
                           gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_convert_to_halves(input, output, numel, stream);
#elif HAVE_ROCM
  HIP_convert_to_halves(input, output, numel, stream);
#endif
}

void GPU_init_curand(GPURandState* states, int num_elems, unsigned int seed,
                     gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_init_curand(states, num_elems, seed, stream);
#elif HAVE_ROCM
  HIP_init_curand(states, num_elems, seed, stream);
#endif
}

size_t GPU_get_topk_utility_buf_size() { return 2 * sizeof(float); }

template <typename T>
void GPU_topk_compress(unsigned char* input_data, unsigned char* output_data,
                       unsigned char* utility_buf, unsigned char* feedback,
                       int num_elems, int num_result, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_topk_compress<T>(input_data, output_data, utility_buf, feedback,
                        num_elems, num_result, stream);
#elif HAVE_ROCM
#endif
}

template <typename T, bool ADD>
void GPU_topk_decompress(unsigned char* input_data, unsigned char* output_data,
                         int num_elems, int num_result, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_topk_decompress<T, ADD>(input_data, output_data, num_elems, num_result,
                          stream);
#elif HAVE_ROCM
#endif
}

template <typename T>
void GPU_add(int n, const T* x, T* y, T* sum, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_add<T>(n, x, y, sum, stream);
#elif HAVE_ROCM
  HIP_add<T>(n, x, y, sum, stream);
#endif
}


template void GPU_quantize_maxmin<float>(unsigned char* input_data,
                                          unsigned char* output_data,
                                          unsigned char* feedback_data,
                                          int num_elems, int bits,
                                          int bucket_size, GPURandState* states,
                                          gpuStream_t stream);
template void GPU_quantize_maxmin<Half>(unsigned char* input_data,
                                         unsigned char* output_data,
                                         unsigned char* feedback_data,
                                         int num_elems, int bits,
                                         int bucket_size, GPURandState* states,
                                         gpuStream_t stream);

template void GPU_dequantize_maxmin<float, true>(unsigned char* input_data,
                                                  unsigned char* output_data,
                                                  int num_elems, int bits,
                                                  int bucket_size,
                                                  gpuStream_t stream);
template void GPU_dequantize_maxmin<float, false>(unsigned char* input_data,
                                                   unsigned char* output_data,
                                                   int num_elems, int bits,
                                                   int bucket_size,
                                                   gpuStream_t stream);

template void GPU_dequantize_maxmin<Half, true>(unsigned char* input_data,
                                                 unsigned char* output_data,
                                                 int num_elems, int bits,
                                                 int bucket_size,
                                                 gpuStream_t stream);
template void GPU_dequantize_maxmin<Half, false>(unsigned char* input_data,
                                                  unsigned char* output_data,
                                                  int num_elems, int bits,
                                                  int bucket_size,
                                                  gpuStream_t stream);

template void
GPU_quantize_Norm<float>(unsigned char* input_data, unsigned char* output_data,
                          unsigned char* feedback, float* levels, int num_elems,
                          int bits, int bucket_size, GPURandState* states,
                          NormType norm_type, LevelsType levels_type,
                          gpuStream_t stream);

template void GPU_quantize_Norm<Half>(unsigned char* input_data,
                                       unsigned char* output_data,
                                       unsigned char* feedback, Half* levels,
                                       int num_elems, int bits, int bucket_size,
                                       GPURandState* states, NormType norm_type,
                                       LevelsType levels_type,
                                       gpuStream_t stream);
template void GPU_dequantize_Norm<float, true>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                float* levels, int num_elems,
                                                int bits, int bucket_size,
                                                LevelsType levels_type,
                                                gpuStream_t stream);

template void GPU_dequantize_Norm<float, false>(unsigned char* input_data,
                                                 unsigned char* output_data,
                                                 float* levels, int num_elems,
                                                 int bits, int bucket_size,
                                                 LevelsType levels_type,
                                                 gpuStream_t stream);

template void GPU_dequantize_Norm<Half, true>(unsigned char* input_data,
                                               unsigned char* output_data,
                                               Half* levels, int num_elems,
                                               int bits, int bucket_size,
                                               LevelsType levels_type,
                                               gpuStream_t stream);

template void GPU_dequantize_Norm<Half, false>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                Half* levels, int num_elems,
                                                int bits, int bucket_size,
                                                LevelsType levels_type,
                                                gpuStream_t stream);
template void GPU_add<float>(int n, const float* x, float* y, float* sum,
                              gpuStream_t stream);
template void GPU_add<Half>(int n, const Half* x, Half* y, Half* sum,
                             gpuStream_t stream);

template void GPU_topk_compress<float>(unsigned char* input_data,
                                        unsigned char* output_data,
                                        unsigned char* utility_buf,
                                        unsigned char* feedback_data,
                                        int num_elems, int num_result,
                                        gpuStream_t stream);

template void GPU_topk_compress<Half>(unsigned char* input_data,
                                       unsigned char* output_data,
                                       unsigned char* utility_buf,
                                       unsigned char* feedback_data,
                                       int num_elems, int num_result,
                                       gpuStream_t stream);

template void GPU_topk_decompress<float, true>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                int num_elems, int num_result,
                                                gpuStream_t stream);

template void GPU_topk_decompress<float, false>(unsigned char* input_data,
                                                 unsigned char* output_data,
                                                 int num_elems, int num_result,
                                                 gpuStream_t stream);

template void GPU_topk_decompress<Half, true>(unsigned char* input_data,
                                               unsigned char* output_data,
                                               int num_elems, int num_result,
                                               gpuStream_t stream);

template void GPU_topk_decompress<Half, false>(unsigned char* input_data,
                                                unsigned char* output_data,
                                                int num_elems, int num_result,
                                                gpuStream_t stream);

} // namespace gpu
} // namespace common
} // namespace horovod
