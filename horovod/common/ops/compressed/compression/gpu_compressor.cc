#include "gpu_compressor.h"
#include "../../../logging.h"
#include "../common.h"
#include "../utils.h"

namespace horovod {
namespace common {

// =======
// Cuda Compression Context
// ======

// TODO: Add compressions/decompressions in multiple streams.
void GPUCompressionContext::Finalize() {
  gpu_context_->StreamSynchronize(*stream_);
}

Status
GPUCompressionContext::Init(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  device_ = first_entry.device;
  gpu_op_context_.InitGPU(entries);
  stream_ = &gpu_context_->streams[global_state_->current_nccl_stream][device_];

  int chunk_size =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  int num_elems_in_chunk = ceil(((double)chunk_size) / sizeof(float));
  int64_t curand_array_size = CUDA_get_curand_array_size(num_elems_in_chunk);
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
    CUDA_init_curand(
        cuda_states_, num_elems_in_chunk, time(NULL), *stream_);
  }
  return Status::OK();
}

Status GPUDummyCompressor::Init(const std::vector<TensorTableEntry>& entries) {
  return gpu_compression_context_->Init(entries);
}

int64_t GPUDummyCompressor::Compress(unsigned char* input_data,
                                     unsigned char* output,
                                     unsigned char* feedback_data,
                                     int64_t num_elems, DataType dtype) {
  int64_t processed_size = num_elems * get_sizeof(dtype);

  gpu_compression_context_->gpu_context_->MemcpyAsyncD2D(
      (void*)output, (void*)input_data, processed_size,
      *gpu_compression_context_->stream_);
  return processed_size;
}

void GPUDummyCompressor::Decompress(unsigned char* input_data,
                                    unsigned char* output, int64_t num_elems,
                                    DataType dtype, bool add) {
  int64_t processed_size = num_elems * get_sizeof(dtype);
  auto stream_p = gpu_compression_context_->stream_;
  if (add) {
    if (dtype == DataType::HOROVOD_FLOAT32)
      CUDA_add_fp32(num_elems, (float*)input_data, (float*)output, (float*)output,
               *stream_p);
    else
      CUDA_add_fp16(num_elems, (Half*)input_data, (Half*)output, (Half*)output,
               *stream_p);
  } else {
    gpu_compression_context_->gpu_context_->MemcpyAsyncD2D(
        (void*)output, (void*)input_data, processed_size, *stream_p);
  }
}

void GPUDummyCompressor::Finalize() { gpu_compression_context_->Finalize(); }
// ================
// Max Min Quantizer
// ================

Status GPUMaxMinQuantizer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  return gpu_compression_context_->Init(entries);
}

int64_t GPUMaxMinQuantizer::Compress(unsigned char* input,
                                     unsigned char* output,
                                     unsigned char* feedback, int64_t num_elems,
                                     DataType dtype) {
  if (dtype != DataType::HOROVOD_FLOAT16) {
    CUDA_quantize_maxmin_fp32(
        input, output, feedback, num_elems, bits_, bucket_size_,
        gpu_compression_context_->cuda_states_,
        *gpu_compression_context_->stream_);
  } else {
    CUDA_quantize_maxmin_fp16(
        input, output, feedback, num_elems, bits_, bucket_size_,
        gpu_compression_context_->cuda_states_,
        *gpu_compression_context_->stream_);
  }
  return BufferSize(num_elems, dtype);
}

void GPUMaxMinQuantizer::Decompress(unsigned char* input, unsigned char* output,
                                    int64_t num_elems, DataType dtype,
                                    bool add) {
  if (dtype != DataType::HOROVOD_FLOAT16) {
    CUDA_dequantize_maxmin_fp32(
        input, output, num_elems, bits_, bucket_size_, add,
        *gpu_compression_context_->stream_);
  } else {
    CUDA_dequantize_maxmin_fp16(
        input, output, num_elems, bits_, bucket_size_, add,
        *gpu_compression_context_->stream_);
  }
}

inline int64_t GPUMaxMinQuantizer::BufferSize(int num_elems, DataType dtype) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = 2 * num_buckets * get_sizeof(dtype);
  int64_t compressed_values_buffer_size = (num_elems * bits_ + 7) / 8;
  return meta_buffer_size +
         round_to(compressed_values_buffer_size, ALIGNMENT_UNIT);
}

void GPUMaxMinQuantizer::Finalize() { gpu_compression_context_->Finalize(); }

// ================
// Normalized Quantizers
// ================

Status GPUNormalizedQuantizer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  gpu_compression_context_->Init(entries);
  if (levels_ == nullptr) {
    int init_num_levels;
    float* host_levels =
        FillLevels(bits_, init_num_levels, compression_type_, levels_type_);
    gpu_compression_context_->gpu_context_->ErrorCheck(
        "cudaMalloc",
        cudaMalloc((void**)&levels_, sizeof(float) * init_num_levels));
    gpu_compression_context_->gpu_context_->ErrorCheck(
        "cudaMalloc",
        cudaMalloc((void**)&levels_fp16_, sizeof(float) * init_num_levels));
    gpu_compression_context_->gpu_context_->ErrorCheck(
        "cudaMemcpy",
        cudaMemcpy((void*)levels_, (void*)host_levels,
                   sizeof(float) * init_num_levels, cudaMemcpyHostToDevice));
    CUDA_convert_to_halves(levels_, levels_fp16_, init_num_levels);
    delete[] host_levels;
  }
  return Status::OK();
}

void GPUNormalizedQuantizer::SetQuantizationLevels(float* levels) {
  int num_levels = 1 << (bits_ - 1);
  if (levels_type_ == LevelsType::Wide)
    num_levels *= 2;
  if (levels_ == nullptr) {
    cudaMalloc((void**)&levels_, sizeof(float) * num_levels);
    cudaMalloc((void**)&levels_fp16_, sizeof(Half) * num_levels);
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
  cudaMemcpy((void*)levels_, (void*)levels, sizeof(float) * num_levels,
             cudaMemcpyHostToDevice);
  CUDA_convert_to_halves(levels_, levels_fp16_, num_levels);
}

void GPUNormalizedQuantizer::Finalize() {
  gpu_compression_context_->Finalize();
}

inline int64_t GPUNormalizedQuantizer::BufferSize(int num_elems,
                                                  DataType dtype) {
  int64_t compressed_values_buffer_size = (num_elems * bits_ + 7) / 8;
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = get_sizeof(dtype) * num_buckets;
  return round_to(compressed_values_buffer_size, ALIGNMENT_UNIT) +
         meta_buffer_size;
}

int64_t GPUNormalizedQuantizer::Compress(unsigned char* input,
                                         unsigned char* output,
                                         unsigned char* feedback,
                                         int64_t num_elems, DataType dtype) {
  if (dtype != DataType::HOROVOD_FLOAT16) {
    CUDA_quantize_Norm_fp32(
        input, output, feedback, levels_, num_elems, bits_, bucket_size_,
        gpu_compression_context_->cuda_states_, norm_type_, levels_type_,
        *gpu_compression_context_->stream_);
  } else {
    CUDA_quantize_Norm_fp16(
        input, output, feedback, levels_fp16_, num_elems, bits_, bucket_size_,
        gpu_compression_context_->cuda_states_, norm_type_, levels_type_,
        *gpu_compression_context_->stream_);
  }
  return BufferSize(num_elems, dtype);
}

void GPUNormalizedQuantizer::Decompress(unsigned char* input,
                                        unsigned char* output,
                                        int64_t num_elems, DataType dtype,
                                        bool add) {
  if (dtype != DataType::HOROVOD_FLOAT16) {
    CUDA_dequantize_Norm_fp32(
        input, output, levels_, num_elems, bits_, bucket_size_, levels_type_,
        add,
        *gpu_compression_context_->stream_);
  } else {
    CUDA_dequantize_Norm_fp16(
        input, output, levels_fp16_, num_elems, bits_, bucket_size_,
        levels_type_, add,
        *gpu_compression_context_->stream_);
  }
}

} // namespace common
} // namespace horovod
