#include "gpu_compressor.h"
#include "../../../logging.h"

namespace horovod {
namespace common {

Status GPUDummyCompressor::Init(const std::vector<TensorTableEntry>& entries) {
  gpu_op_context_.InitGPU(entries);

  device_ = entries[0].device;
  return Status::OK();
}

int64_t GPUDummyCompressor::Compress(unsigned char* input_data,
                                     unsigned char* output,
                                     unsigned char* feedback_data,
                                     int64_t num_elems) {
  int64_t processed_size = num_elems * sizeof(float);

  gpu_context_->MemcpyAsyncD2D(
      (void*)output, (void*)input_data, processed_size,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  return processed_size;
}

void GPUDummyCompressor::Decompress(unsigned char* input_data,
                                    unsigned char* output, int64_t num_elems) {
  int64_t processed_size = num_elems * sizeof(float);
  gpu_context_->MemcpyAsyncD2D(
      (void*)output, (void*)input_data, processed_size,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

void GPUDummyCompressor::Finalize() {
  gpu_context_->StreamSynchronize(
      gpu_context_
          ->streams[global_state_->current_nccl_stream][device_]);
}

Status GPUCompressionContext::Init(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  device_ = first_entry.device;
  gpu_op_context_.InitGPU(entries);

  int num_elems_in_chunk = ceil(((double)chunk_size_) / sizeof(float));
  int64_t curand_array_size = CUDA_get_curand_array_size(num_elems_in_chunk);
  Status status = bufferManager_.InitializeBuffer(
      curand_array_size, device_, first_entry.context, current_nccl_stream_,
      [&]() {}, [&]() {});
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }

  auto buffer = bufferManager_.GetBuffer(
      device_, first_entry.context->framework(), current_nccl_stream_);

  cuda_states_ = static_cast<CurandState*>(
      const_cast<void*>(buffer->AccessData(first_entry.context)));
  CUDA_init_curand(cuda_states_, num_elems_in_chunk, time(NULL),
                   gpu_context_->streams[current_nccl_stream_][device_]);
  return Status::OK();
}

// ================
// Max Min Quantizer
// ================

GPUMaxMinQuantizer::GPUMaxMinQuantizer(
    horovod::common::GPUContext* gpu_context,
    horovod::common::HorovodGlobalState* global_state, int quantization_bits)
    : MaxMinQuantizer(global_state, quantization_bits),
      GPUCompressionContext(gpu_context, global_state) {}

Status GPUMaxMinQuantizer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  Status status = GPUCompressionContext::Init(entries);
  if (!status.ok()) {
    return status;
  }
  return Status::OK();
}

int64_t GPUMaxMinQuantizer::Compress(unsigned char* input,
                                     unsigned char* output,
                                     unsigned char* feedback,
                                     int64_t num_elems) {
  auto start = clock_::now();
  CUDA_quantize_maxmin(
      input, output, feedback, num_elems, bits_, bucket_size_, cuda_states_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
  return BufferSize(num_elems);
}

void GPUMaxMinQuantizer::Decompress(unsigned char* input, unsigned char* output,
                                    int64_t num_elems) {
  auto start = clock_::now();
  CUDA_dequantize_maxmin(
      input, output, num_elems, bits_, bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
}

inline int64_t GPUMaxMinQuantizer::BufferSize(int num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = 2 * num_buckets * sizeof(float);
  int64_t compressed_values_buffer_size = (num_elems * bits_ + 7) / 8;
  return meta_buffer_size +
         round_to(compressed_values_buffer_size, ALIGNMENT_UNIT);
}

void GPUMaxMinQuantizer::Finalize() {
  gpu_context_->StreamSynchronize(
      gpu_context_
          ->streams[global_state_->current_nccl_stream][device_]);
}
// ================
// Normalized Quantizers
// ================

Status GPUNormalizedQuantizer::Init(
    GPUContext* gpu_context,
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  if (levels_ == nullptr) {
    float* host_levels;
    int num_levels = 1 << (bits_ - 1);
    if (multiplier_ < 0.0) {
      host_levels = new float[num_levels];

      host_levels[0] = 1.0;
      float value = 1.0 / (num_levels - 1);
      for (int i = 1; i <= num_levels - 1; i++) {
        host_levels[i] = (num_levels - i - 1) * value;
      }
    } else {
      // Exponential levels
      num_levels *= 4;
      host_levels = new float[num_levels];
      host_levels[0] = 1.0;
      float level_v = multiplier_;
      for (int i = 1; i < num_levels - 1; i++) {
        host_levels[i] = level_v;
        level_v *= multiplier_;
      }
    }
    gpu_context->ErrorCheck(
        "cudaMalloc", cudaMalloc((void**)&levels_, sizeof(float) * num_levels));
    gpu_context->ErrorCheck("cudaMemcpy",
                            cudaMemcpy((void*)levels_, (void*)host_levels,
                                       sizeof(float) * num_levels,
                                       cudaMemcpyHostToDevice));
    delete[] host_levels;
  }
  return Status::OK();
}

void GPUNormalizedQuantizer::SetQuantizationLevels(float* levels) {
  int num_levels = 1 << (bits_ - 1);
  if (levels_ == nullptr) {
      cudaMalloc((void**)&levels_, sizeof(float) * num_levels);
  }
  if (global_state_->controller->GetRank() == 0) {
    std::cout << "Set levels: [";
    for (int i = 0; i < num_levels; i++){
      std::cout << " " << levels[i];
    }
    std::cout << " ]" << std::endl;
  }
  cudaMemcpy((void*)levels_, (void*)levels,
             sizeof(float) * num_levels,
             cudaMemcpyHostToDevice);
}

GPUNormLinfQuantizer::GPUNormLinfQuantizer(GPUContext* gpu_context,
                                           HorovodGlobalState* global_state,
                                           int quantization_bits,
                                           float multiplier)
    : GPUNormalizedQuantizer(global_state, quantization_bits, multiplier),
      GPUCompressionContext(gpu_context, global_state) {}

int64_t GPUNormLinfQuantizer::Compress(unsigned char* input,
                                       unsigned char* output,
                                       unsigned char* feedback,
                                       int64_t num_elems) {
  auto start = clock_::now();
  CUDA_quantize_LinfNorm(
      input, output, feedback, levels_, num_elems, bits_, bucket_size_,
      cuda_states_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
  return BufferSize(num_elems);
}

void GPUNormLinfQuantizer::Decompress(unsigned char* input,
                                      unsigned char* output,
                                      int64_t num_elems) {
  auto start = clock_::now();
  CUDA_dequantize_LinfNorm(
      input, output, levels_, num_elems, bits_, bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
}

Status
GPUNormLinfQuantizer::Init(const std::vector<TensorTableEntry>& entries) {
  Status status = GPUCompressionContext::Init(entries);
  if (!status.ok()) {
    return status;
  }
  status = GPUNormalizedQuantizer::Init(gpu_context_, entries);
  return status;
}

inline int64_t GPUNormLinfQuantizer::BufferSize(int num_elems) {
  int64_t compressed_values_buffer_size = (num_elems * bits_ + 7) / 8;
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = sizeof(float) * num_buckets;
  return round_to(compressed_values_buffer_size, ALIGNMENT_UNIT) +
         meta_buffer_size;
}

void GPUNormLinfQuantizer::Finalize() {
  gpu_context_->StreamSynchronize(
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

GPUNormL2Quantizer::GPUNormL2Quantizer(GPUContext* gpu_context,
                                       HorovodGlobalState* global_state,
                                       int quantization_bits, float multiplier)
    : GPUNormalizedQuantizer(global_state, quantization_bits, multiplier),
      GPUCompressionContext(gpu_context, global_state) {
  if (multiplier != 0.5)
    throw std::logic_error(
        "CPUNormL2Quantizer: Multipliers other than 0.5 are not supported yet");
}

int64_t GPUNormL2Quantizer::Compress(unsigned char* input,
                                     unsigned char* output,
                                     unsigned char* feedback,
                                     int64_t num_elems) {
  auto start = clock_::now();
  CUDA_quantize_L2Norm(
      input, output, feedback, levels_, num_elems, bits_, bucket_size_,
      cuda_states_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
  return BufferSize(num_elems);
}

void GPUNormL2Quantizer::Decompress(unsigned char* input, unsigned char* output,
                                    int64_t num_elems) {
  auto start = clock_::now();
  CUDA_dequantize_L2Norm(
      input, output, levels_, num_elems, bits_, bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
}

Status GPUNormL2Quantizer::Init(const std::vector<TensorTableEntry>& entries) {
  Status status = GPUCompressionContext::Init(entries);
  if (!status.ok()) {
    return status;
  }
  status = GPUNormalizedQuantizer::Init(gpu_context_, entries);
  return status;
}

inline int64_t GPUNormL2Quantizer::BufferSize(int num_elems) {
  int64_t compressed_values_buffer_size = (num_elems * bits_ + 7) / 8;
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to allocate max_log + norm
  int64_t meta_buffer_size = (sizeof(char) + sizeof(float)) * num_buckets;
  return round_to(compressed_values_buffer_size + meta_buffer_size,
                  ALIGNMENT_UNIT);
}

void GPUNormL2Quantizer::Finalize() {
  gpu_context_->StreamSynchronize(
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
}

} // namespace common
} // namespace horovod
