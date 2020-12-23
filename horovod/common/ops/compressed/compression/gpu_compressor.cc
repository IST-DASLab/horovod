#include "gpu_compressor.h"
#include "../../../logging.h"
#include "../common.h"
#include "../utils.h"
#include "cuda/cuda_arithmetic_functions.h"
#include "cuda/cuda_compression_functions.h"

namespace horovod {
namespace common {

// =======
// Cuda Compression Context
// ======

Status
GPUCompressionContext::Init(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  device_ = first_entry.device;
  gpu_op_context_.InitGPU(entries);

  int chunk_size =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  int num_elems_in_chunk =
      ceil(((double)chunk_size) / get_sizeof(first_entry.tensor->dtype()));
  size_t curand_array_size =
      cuda::CUDA_get_curand_array_size(num_elems_in_chunk);
  Status status = bufferManager_.InitializeBuffer(
      curand_array_size, device_, first_entry.context,
      global_state_->current_nccl_stream, [&]() {}, [&]() {});
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
  auto buffer =
      bufferManager_.GetBuffer(device_, first_entry.context->framework(),
                               global_state_->current_nccl_stream);
  if (cuda_states_ == nullptr) {
    cuda_states_ = static_cast<CurandState*>(
        const_cast<void*>(buffer->AccessData(first_entry.context)));
    cuda::CUDA_init_curand(
        cuda_states_, num_elems_in_chunk, time(NULL),
        gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  }
  return Status::OK();
}

Status GPUDummyCompressor::Init(const std::vector<TensorTableEntry>& entries) {
  auto status = gpu_compression_context_->Init(entries);
  if (!status.ok())
    return status;
  return Compressor::Init(entries);
}

size_t GPUDummyCompressor::CompressBuffer(
    unsigned char* input_data, unsigned char* output,
    unsigned char* feedback_data, int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg, void* ctx) {
  cudaStream_t* stream = (cudaStream_t*)ctx;
  size_t processed_size = num_elems * get_sizeof(dtype);
  gpu_compression_context_->gpu_context_->MemcpyAsyncD2D(
      (void*)output, (void*)input_data, processed_size, *stream);
  return processed_size;
}

void GPUDummyCompressor::DecompressBuffer(
    unsigned char* input_data, unsigned char* output, int num_elems,
    DataType dtype, bool add, const CompressionModuleConfig& compression_cfg,
    void* ctx) {
  cudaStream_t* stream = (cudaStream_t*)ctx;
  size_t processed_size = num_elems * get_sizeof(dtype);
  if (add) {
    if (dtype == DataType::HOROVOD_FLOAT32)
      cuda::CUDA_add<float>(num_elems, (float*)input_data, (float*)output,
                            (float*)output, *stream);
    else
      cuda::CUDA_add<Half>(num_elems, (Half*)input_data, (Half*)output,
                           (Half*)output, *stream);
  } else {
    gpu_compression_context_->gpu_context_->MemcpyAsyncD2D(
        (void*)output, (void*)input_data, processed_size, *stream);
  }
}

// ================
// Max Min Quantizer
// ================

Status GPUMaxMinQuantizer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto status = gpu_compression_context_->Init(entries);
  if (!status.ok())
    return status;
  return Compressor::Init(entries);
}

size_t GPUMaxMinQuantizer::GetRequiredFreeSize() {
  int chunk_size =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  int num_elems_in_chunk =
      ceil(((double)chunk_size) / get_sizeof(HOROVOD_FLOAT16));
  return cuda::CUDA_get_curand_array_size(num_elems_in_chunk);
}

size_t GPUMaxMinQuantizer::CompressBuffer(
    unsigned char* input, unsigned char* output, unsigned char* feedback,
    int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg, void* ctx) {
  if (num_elems == 0)
    return 0;
  cudaStream_t* stream = (cudaStream_t*)ctx;
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  const bool skip_incomplete = compression_cfg.skip_incomplete_buckets ||
                               num_elems < MIN_SIZE_TO_COMPRESS;
  int num_elems_to_compress = num_elems;
  int residual_elems = 0;
  int compressed_size = 0;
  if (skip_incomplete) {
    num_elems_to_compress = (num_elems / bucket_size) * bucket_size;
    residual_elems = num_elems % bucket_size;
  }
  if (num_elems_to_compress > 0) {
    if (dtype != DataType::HOROVOD_FLOAT16) {
      cuda::CUDA_quantize_maxmin<float>(
          input, output, feedback, num_elems_to_compress, bits, bucket_size,
          gpu_compression_context_->cuda_states_, *stream);
    } else {
      cuda::CUDA_quantize_maxmin<Half>(
          input, output, feedback, num_elems_to_compress, bits, bucket_size,
          gpu_compression_context_->cuda_states_, *stream);
    }
    compressed_size = BufferSize(num_elems_to_compress, dtype, compression_cfg);
  }
  if (skip_incomplete and residual_elems > 0) {
    input += num_elems_to_compress * get_sizeof(dtype);
    output += compressed_size;
    gpu_compression_context_->gpu_context_->MemcpyAsyncD2D(
        (void*)output, (void*)input, residual_elems * get_sizeof(dtype),
        *stream);
    compressed_size += get_sizeof(dtype) * residual_elems;
  }
  return compressed_size;
}

void GPUMaxMinQuantizer::DecompressBuffer(
    unsigned char* input, unsigned char* output, int num_elems,
    DataType dtype, bool add, const CompressionModuleConfig& compression_cfg,
    void* ctx) {
  if (num_elems == 0)
    return;
  cudaStream_t* stream = (cudaStream_t*)ctx;
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  const bool skip_incomplete = compression_cfg.skip_incomplete_buckets ||
                               num_elems < MIN_SIZE_TO_COMPRESS;
  int num_elems_to_decompress = num_elems;
  int residual_elems = 0;
  if (skip_incomplete) {
    num_elems_to_decompress = (num_elems / bucket_size) * bucket_size;
    residual_elems = num_elems % bucket_size;
  }
  if (num_elems_to_decompress > 0) {
    if (add) {
      if (dtype != DataType::HOROVOD_FLOAT16) {
        cuda::CUDA_dequantize_maxmin<float, true>(
            input, output, num_elems_to_decompress, bits, bucket_size, *stream);
      } else {
        cuda::CUDA_dequantize_maxmin<Half, true>(
            input, output, num_elems_to_decompress, bits, bucket_size, *stream);
      }
    } else {
      if (dtype != DataType::HOROVOD_FLOAT16) {
        cuda::CUDA_dequantize_maxmin<float, false>(
            input, output, num_elems_to_decompress, bits, bucket_size, *stream);
      } else {
        cuda::CUDA_dequantize_maxmin<Half, false>(
            input, output, num_elems_to_decompress, bits, bucket_size, *stream);
      }
    }
  }

  if (skip_incomplete and residual_elems > 0) {
    int compressed_size =
        BufferSize(num_elems_to_decompress, dtype, compression_cfg);
    input += compressed_size;
    output += num_elems_to_decompress * get_sizeof(dtype);
    if (add) {
      if (dtype == DataType::HOROVOD_FLOAT32)
        cuda::CUDA_add<float>(residual_elems, (float*)input, (float*)output,
                              (float*)output, *stream);
      else
        cuda::CUDA_add<Half>(residual_elems, (Half*)input, (Half*)output,
                             (Half*)output, *stream);
    } else {
      gpu_compression_context_->gpu_context_->MemcpyAsyncD2D(
          (void*)output, (void*)input, residual_elems * get_sizeof(dtype),
          *stream);
    }
  }
}

inline size_t
GPUMaxMinQuantizer::BufferSize(int num_elems, DataType dtype,
                               const CompressionModuleConfig& compression_cfg) {
  if (num_elems == 0)
    return 0;
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  const bool skip_incomplete = compression_cfg.skip_incomplete_buckets ||
                               num_elems < MIN_SIZE_TO_COMPRESS;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  int residuals = 0;
  if (skip_incomplete) {
    num_buckets = num_elems / bucket_size;
    residuals = num_elems % bucket_size;
    num_elems = num_buckets * bucket_size;
  }
  size_t meta_buffer_size = 2 * num_buckets * get_sizeof(dtype);
  size_t compressed_values_buffer_size = (num_elems * bits + 7) / 8;
  return meta_buffer_size + ALIGNED_SIZE(compressed_values_buffer_size) +
         residuals * get_sizeof(dtype);
}

// ================
// Normalized Quantizers
// ================
size_t GPUNormalizedQuantizer::GetRequiredFreeSize() {
  int chunk_size =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  int num_elems_in_chunk =
      ceil(((double)chunk_size) / get_sizeof(HOROVOD_FLOAT16));
  return cuda::CUDA_get_curand_array_size(num_elems_in_chunk);
}

Status GPUNormalizedQuantizer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto status = gpu_compression_context_->Init(entries);
  if (!status.ok())
    return status;
  for (auto& entry : entries) {
    auto& config = GetModuleConfig(entry.tensor_name);
    if (bits_to_levels_.find(config.quantization_bits) ==
        bits_to_levels_.end()) {
      float* copy_levels;
      Half* copy_levels_fp16;
      int init_num_levels;
      float* host_levels =
          FillLevels(default_config.quantization_bits, init_num_levels,
                     compression_type_, levels_type_);

      gpu_compression_context_->gpu_context_->ErrorCheck(
          "cudaMalloc",
          cudaMalloc((void**)&copy_levels, sizeof(float) * init_num_levels));
      gpu_compression_context_->gpu_context_->ErrorCheck(
          "cudaMalloc", cudaMalloc((void**)&copy_levels_fp16,
                                   sizeof(Half) * init_num_levels));
      gpu_compression_context_->gpu_context_->ErrorCheck(
          "cudaMemcpy",
          cudaMemcpy((void*)copy_levels, (void*)host_levels,
                     sizeof(float) * init_num_levels, cudaMemcpyHostToDevice));
      cuda::CUDA_convert_to_halves(copy_levels, copy_levels_fp16,
                                   init_num_levels);

      bits_to_levels_[config.quantization_bits] = copy_levels;
      bits_to_levels_fp16_[config.quantization_bits] = copy_levels_fp16;
      delete[] host_levels;
    }
  }
  return Compressor::Init(entries);
}

void GPUNormalizedQuantizer::SetQuantizationLevels(float* levels, int bits) {
  int num_levels = 1 << (bits - 1);
  float* copy_levels;
  Half* copy_levels_fp16;

  if (levels_type_ == LevelsType::Wide)
    num_levels *= 2;
  if (bits_to_levels_.find(bits) == bits_to_levels_.end()) {
    cudaMalloc((void**)&copy_levels, sizeof(float) * num_levels);
    cudaMalloc((void**)&copy_levels_fp16, sizeof(Half) * num_levels);
  } else {
    copy_levels = bits_to_levels_[bits];
    copy_levels_fp16 = bits_to_levels_fp16_[bits];
  }
  if (global_state_->controller->GetRank() == 0) {
    std::stringstream message;
    message << "Set levels: [";
    for (int i = 0; i < num_levels; i++) {
      message << " " << levels[i];
    }
    message << " ]" << std::endl;
    LOG(INFO) << message.str();
  }
  cudaMemcpy((void*)copy_levels, (void*)levels, sizeof(float) * num_levels,
             cudaMemcpyHostToDevice);
  cuda::CUDA_convert_to_halves(copy_levels, copy_levels_fp16, num_levels);
  bits_to_levels_[bits] = copy_levels;
  bits_to_levels_fp16_[bits] = copy_levels_fp16;
}

inline size_t GPUNormalizedQuantizer::BufferSize(
    int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg) {
  if (num_elems <= 0)
    return 0;
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  const bool skip_incomplete = compression_cfg.skip_incomplete_buckets ||
                               num_elems < MIN_SIZE_TO_COMPRESS;
  int residual_elems = 0;
  size_t num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  if (skip_incomplete) {
    num_buckets = num_elems / bucket_size;
    residual_elems = num_elems % bucket_size;
    num_elems = num_buckets * bucket_size;
  }
  size_t compressed_values_buffer_size = (num_elems * bits + 7) / 8;
  size_t meta_buffer_size = get_sizeof(dtype) * num_buckets;
  return ALIGNED_SIZE(compressed_values_buffer_size) + meta_buffer_size +
         residual_elems * get_sizeof(dtype);
}

size_t GPUNormalizedQuantizer::CompressBuffer(
    unsigned char* input, unsigned char* output, unsigned char* feedback,
    int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg, void* ctx) {
  if (num_elems <= 0) {
    return 0;
  }
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  const bool skip_incomplete = compression_cfg.skip_incomplete_buckets ||
                               num_elems < MIN_SIZE_TO_COMPRESS;
  cudaStream_t* stream = (cudaStream_t*)ctx;
  int num_elems_compress = num_elems;
  int residual_elems = 0;
  if (skip_incomplete) {
    num_elems_compress = (num_elems / bucket_size) * bucket_size;
    residual_elems = num_elems % bucket_size;
  }
  int compressed_size = BufferSize(num_elems_compress, dtype, compression_cfg);
  if (num_elems_compress > 0) {
    if (dtype != DataType::HOROVOD_FLOAT16) {
      cuda::CUDA_quantize_Norm<float>(
          input, output, feedback, bits_to_levels_[bits], num_elems_compress,
          bits, bucket_size, gpu_compression_context_->cuda_states_, norm_type_,
          levels_type_, *stream);
    } else {
      cuda::CUDA_quantize_Norm<Half>(input, output, feedback,
                                     bits_to_levels_fp16_[bits],
                                     num_elems_compress, bits, bucket_size,
                                     gpu_compression_context_->cuda_states_,
                                     norm_type_, levels_type_, *stream);
    }
  }
  if (skip_incomplete and residual_elems > 0) {
    input += num_elems_compress * get_sizeof(dtype);
    output += compressed_size;
    gpu_compression_context_->gpu_context_->MemcpyAsyncD2D(
        (void*)output, (void*)input, residual_elems * get_sizeof(dtype),
        *stream);
    compressed_size += get_sizeof(dtype) * residual_elems;
  }
  return compressed_size;
}

void GPUNormalizedQuantizer::DecompressBuffer(
    unsigned char* input, unsigned char* output, int num_elems,
    DataType dtype, bool add, const CompressionModuleConfig& compression_cfg,
    void* ctx) {
  if (num_elems <= 0)
    return;
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  const bool skip_incomplete = compression_cfg.skip_incomplete_buckets ||
                               num_elems < MIN_SIZE_TO_COMPRESS;
  int num_elems_decompress = num_elems;
  int residual_elems = 0;
  if (skip_incomplete) {
    num_elems_decompress = (num_elems / bucket_size) * bucket_size;
    residual_elems = num_elems % bucket_size;
  }
  cudaStream_t* stream = (cudaStream_t*)ctx;
  if (num_elems_decompress > 0) {
    if (add) {
      if (dtype != DataType::HOROVOD_FLOAT16) {
        cuda::CUDA_dequantize_Norm<float, true>(
            input, output, bits_to_levels_[bits], num_elems_decompress, bits,
            bucket_size, levels_type_, *stream);
      } else {
        cuda::CUDA_dequantize_Norm<Half, true>(
            input, output, bits_to_levels_fp16_[bits], num_elems_decompress,
            bits, bucket_size, levels_type_, *stream);
      }
    } else {
      if (dtype != DataType::HOROVOD_FLOAT16) {
        cuda::CUDA_dequantize_Norm<float, false>(
            input, output, bits_to_levels_[bits], num_elems_decompress, bits,
            bucket_size, levels_type_, *stream);
      } else {
        cuda::CUDA_dequantize_Norm<Half, false>(
            input, output, bits_to_levels_fp16_[bits], num_elems_decompress,
            bits, bucket_size, levels_type_, *stream);
      }
    }
  }
  if (skip_incomplete and residual_elems > 0) {
    int compressed_size =
        BufferSize(num_elems_decompress, dtype, compression_cfg);
    input += compressed_size;
    output += num_elems_decompress * get_sizeof(dtype);
    if (add) {
      if (dtype == DataType::HOROVOD_FLOAT32)
        cuda::CUDA_add<float>(residual_elems, (float*)input, (float*)output,
                              (float*)output, *stream);
      else
        cuda::CUDA_add<Half>(residual_elems, (Half*)input, (Half*)output,
                             (Half*)output, *stream);
    } else {
      gpu_compression_context_->gpu_context_->MemcpyAsyncD2D(
          (void*)output, (void*)input, residual_elems * get_sizeof(dtype),
          *stream);
    }
  }
}

// ================
// TopK Compressor
// ================
GPUTopKCompressor::GPUTopKCompressor(GPUContext* gpu_context,
                                     HorovodGlobalState* global_state,
                                     Summator* summator, float topk_ratio)
    : Compressor(global_state, summator), topk_ratio_(topk_ratio) {
  gpu_compression_context_ = std::unique_ptr<GPUCompressionContext>(
      new GPUCompressionContext(gpu_context, global_state));
}

Status GPUTopKCompressor::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  size_t required_size = cuda::CUDA_get_topk_utility_buf_size();
  if (utility_buf_ == nullptr) {
    gpu_compression_context_->gpu_context_->ErrorCheck(
        "cudaMalloc", cudaMalloc((void**)&utility_buf_, required_size));
  }
  return Compressor::Init(entries);
}

size_t GPUTopKCompressor::GetRequiredFreeSize() {
  // TODO: For precise topK algorithm we need more memory.
  size_t result = cuda::CUDA_get_topk_utility_buf_size();
  return result;
}

size_t GPUTopKCompressor::BufferSize(
    int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg) {
  int num_result = ceil(num_elems * topk_ratio_);
  return num_result * (sizeof(unsigned int) + get_sizeof(dtype));
}

size_t GPUTopKCompressor::CompressBuffer(
    unsigned char* input, unsigned char* output, unsigned char* feedback,
    int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg, void* ctx) {
  if (num_elems <= 0)
    return 0;
  cudaStream_t* stream = (cudaStream_t*)ctx;
  int num_result = ceil(num_elems * topk_ratio_);
//  int num_result = 10;

  if (dtype == DataType::HOROVOD_FLOAT32) {
    cuda::CUDA_topk_compress<float>(input, output, utility_buf_, feedback, num_elems,
                                    num_result, *stream);
  } else if (dtype == DataType::HOROVOD_FLOAT16) {
    cuda::CUDA_topk_compress<Half>(input, output, utility_buf_, feedback, num_elems,
                                   num_result, *stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(*stream));
  return BufferSize(num_elems, dtype, compression_cfg);
}

void GPUTopKCompressor::DecompressBuffer(
    unsigned char* input, unsigned char* output, int num_elems, DataType dtype,
    bool add, const CompressionModuleConfig& compression_cfg, void* ctx) {
  if (num_elems <= 0)
    return;
  cudaStream_t* stream = (cudaStream_t*)ctx;
  int num_result = ceil(num_elems * topk_ratio_);
//  int num_result = 10;
  if (add) {
    if (dtype != DataType::HOROVOD_FLOAT16) {
      cuda::CUDA_topk_decompress<float, true>(input, output, num_elems,
                                              num_result, *stream);
    } else {
      cuda::CUDA_topk_decompress<Half, true>(input, output, num_elems,
                                             num_result, *stream);
    }
  } else {
    if (dtype != DataType::HOROVOD_FLOAT16) {
      cuda::CUDA_topk_decompress<float, false>(input, output, num_elems,
                                               num_result, *stream);
    } else {
      cuda::CUDA_topk_decompress<Half, false>(input, output, num_elems,
                                              num_result, *stream);
    }
  }
}

} // namespace common
} // namespace horovod
