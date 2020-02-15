#include "compressor.h"
#include <assert.h>
#include <cstring>

#include "../../../logging.h"
#include "../../../utils/env_parser.h"
#include "../utils.h"

namespace horovod {
namespace common {

Compressor::Compressor(HorovodGlobalState* global_state)
    : global_state_(global_state) {
  bucket_size_ = GetIntEnvOrDefault(HOROVOD_COMPRESSION_BUCKET_SIZE,
                                    COMPRESSION_BUCKET_SIZE);
}

int64_t Compressor::BufferSize(
    int chunk_num_elems,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int64_t fusion_offset, int64_t global_offset) {
  if (entries.size() == 1) {
    // there is no need in bucket preparation
    // Call ordinary Compression
    return BufferSize(chunk_num_elems);
  }

  int64_t offset_cumm = 0;
  int64_t nelem = 0;
  int64_t sum_result = 0;
  for (auto& entry : entries) {
    nelem = entry.tensor->shape().num_elements();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }

    if (offset_cumm - fusion_offset >= chunk_num_elems) {
      break;
    }

    if (offset_cumm < fusion_offset) {
      // If the first part of param group is placed in previous slice
      // depending on reduction algorithm.
      nelem = offset_cumm + nelem - fusion_offset;
    }

    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
              std::max(offset_cumm, fusion_offset);
    }
    sum_result += BufferSize(nelem);
    offset_cumm += entry.tensor->shape().num_elements();
  }
  return sum_result;
}

int64_t Compressor::Compress(
    unsigned char* input_data, unsigned char* output,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int64_t fusion_offset, int64_t global_offset, int64_t chunk_num_elems) {
  if (entries.size() == 1) {
    // there is no need in bucket preparation
    // Call ordinary Compression
    return Compress(input_data, output, chunk_num_elems);
  }
  int64_t offset_cumm = 0;
  int64_t nelem = 0;
  int64_t buffer_offset = 0;
  int64_t total_compressed_size = 0;
  int compressed_size;
  for (auto& entry : entries) {
    nelem = entry.tensor->shape().num_elements();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }

    if (offset_cumm - fusion_offset >= chunk_num_elems) {
      break;
    }

    if (offset_cumm < fusion_offset) {
      // If the first part of param group is placed in previous slice
      // depending on reduction algorithm.
      nelem = offset_cumm + nelem - fusion_offset;
    }

    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
              std::max(offset_cumm, fusion_offset);
    }

    buffer_offset = std::max(offset_cumm - fusion_offset, 0l);

    compressed_size = round_to(
        Compress(input_data + buffer_offset * sizeof(float), output, nelem),
        ALIGNMENT_UNIT);
    offset_cumm += entry.tensor->shape().num_elements();
    output += compressed_size;
    total_compressed_size += compressed_size;
  }
  return total_compressed_size;
}

void Compressor::Decompress(
    unsigned char* input_data, unsigned char* output,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int64_t fusion_offset, int64_t global_offset, int64_t chunk_num_elems) {
  if (entries.size() == 1) {
    // there is no need in bucket preparation
    // Call ordinary Compression
    Decompress(input_data, output, chunk_num_elems);
    return;
  }

  int64_t offset_cumm = 0;
  int64_t nelem = 0;
  int64_t buffer_offset = 0;
  int64_t cumm_decompressed = 0;
  for (auto& entry : entries) {
    nelem = entry.tensor->shape().num_elements();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }
    if (offset_cumm - fusion_offset >= chunk_num_elems)
      break;

    if (offset_cumm < fusion_offset) {
      // If the first part of param group is placed in previous slice
      // depending on reduction algorithm.
      nelem = offset_cumm + nelem - fusion_offset;
    }
    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
              std::max(offset_cumm, fusion_offset);
    }
    buffer_offset = std::max(offset_cumm - fusion_offset, 0l);
    output = output + buffer_offset * sizeof(float);
    Decompress(input_data + cumm_decompressed, output, nelem);
    cumm_decompressed += BufferSize(nelem);
    offset_cumm += entry.tensor->shape().num_elements();
  }
}
double Compressor::getMetaInfoTime() const { return meta_info_time_; }
double Compressor::getCompressionTime() const { return compression_time_; }

// ================
// Dummy Compressors
// ================
CPUDummyCompressor::CPUDummyCompressor(
    horovod::common::HorovodGlobalState* global_state)
    : Compressor(global_state) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "CPUDummyCompressor";
  }
}

int64_t CPUDummyCompressor::Compress(unsigned char* input_data,
                                     unsigned char* output, int64_t num_elems) {
  int64_t processed_size = num_elems * sizeof(float);
  std::memcpy(output, input_data, processed_size);
  return processed_size;
}

void CPUDummyCompressor::Decompress(unsigned char* input_data,
                                    unsigned char* output, int64_t num_elems) {
  int64_t processed_size = num_elems * sizeof(float);
  std::memcpy(output, input_data, processed_size);
}

GPUDummyCompressor::GPUDummyCompressor(GPUContext* gpu_context,
                                       HorovodGlobalState* global_state)
    : GPUCompressor(gpu_context), CPUDummyCompressor(global_state) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "GPUDummyCompressor";
  }
}

int64_t GPUDummyCompressor::Compress(unsigned char* input_data,
                                     unsigned char* output, int64_t num_elems) {
  int64_t processed_size = num_elems * sizeof(float);
  cudaMemcpyAsync((void*)output, (void*)input_data, processed_size,
                  cudaMemcpyDeviceToDevice);
  return processed_size;
}

void GPUDummyCompressor::Decompress(unsigned char* input_data,
                                    unsigned char* output, int64_t num_elems) {
  int64_t processed_size = num_elems * sizeof(float);
  cudaMemcpyAsync((void*)output, (void*)input_data, processed_size,
                  cudaMemcpyDeviceToDevice);
}

// ================
// Quantizers
// ================

Quantizer::Quantizer(horovod::common::HorovodGlobalState* global_state,
                     int quantization_bits)
    : Compressor(global_state) {
  bits_ = quantization_bits;
}

Status GPUQuantizer::Init(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  device_ = first_entry.device;

  int64_t chunk_size =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  int num_elems_in_chunk = ceil(((double)chunk_size) / sizeof(float));
  int64_t curand_array_size = CUDA_get_curand_array_size(num_elems_in_chunk);
  Status status = bufferManager_.InitializeBuffer(
      curand_array_size, first_entry.device, first_entry.context,
      global_state_->current_nccl_stream, [&]() {}, [&]() {});
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }

  auto buffer = bufferManager_.GetBuffer(first_entry.device,
                                         first_entry.context->framework(),
                                         global_state_->current_nccl_stream);

  cuda_states_ = static_cast<CurandState*>(
      const_cast<void*>(buffer->AccessData(first_entry.context)));
  CUDA_init_curand(
      cuda_states_, num_elems_in_chunk, time(NULL),
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  return Status::OK();
}

// ================
// Max Min Quantizer
// ================

GPUMaxMinQuantizer::GPUMaxMinQuantizer(
    horovod::common::GPUContext* gpu_context,
    horovod::common::HorovodGlobalState* global_state, int quantization_bits)
    : GPUQuantizer(gpu_context, global_state, quantization_bits) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "GPUMaxMinQuantizer";
  }
}

int64_t GPUMaxMinQuantizer::BufferSize(int num_elems) {
  int64_t compressed_values_buffer_size =
      round_to(ceil(1.0 * num_elems * bits_ / 8), ALIGNMENT_UNIT);
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Max min buffer to allocate
  int64_t meta_buffer_size = 2 * sizeof(float) * num_buckets;
  return compressed_values_buffer_size + meta_buffer_size;
}

int64_t GPUMaxMinQuantizer::Compress(unsigned char* input_data,
                                     unsigned char* output, int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Maxmin buffer to use.
  int64_t meta_buffer_size = 2 * num_buckets * sizeof(float);
  auto start = clock_::now();
  unsigned char* meta_buffer = output;
  CUDA_find_max_and_min_bucket(
      (float*)input_data, (float*)meta_buffer, num_elems, bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  meta_info_time_ += time_since(start);
  start = clock_::now();
  CUDA_quantize_maxmin(
      meta_buffer + meta_buffer_size, (float*)input_data, (float*)meta_buffer,
      num_elems, bits_, bucket_size_, cuda_states_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
  int entries_per_byte = 8 / bits_;
  int64_t total_size =
      meta_buffer_size +
      round_to((num_elems + entries_per_byte - 1) / entries_per_byte,
               ALIGNMENT_UNIT);
  return total_size;
}

void GPUMaxMinQuantizer::Decompress(unsigned char* input_data,
                                    unsigned char* output, int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Maxmin buffer to use.
  int64_t meta_buffer_size = 2 * num_buckets * sizeof(float);
  auto start = clock_::now();
  CUDA_dequantize_maxmin(
      input_data + meta_buffer_size, (float*)input_data, (float*)output,
      num_elems, bits_, bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
}

// ================
// Normalized Quantizers
// ================

Status GPUNormalizedQuantizer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  Status status = GPUQuantizer::Init(entries);
  if (!status.ok()) {
    return status;
  }
  if (levels_ == nullptr) {
    int num_levels = 1 << (bits_ - 1);
    cudaMalloc((void**)&levels_, num_levels * sizeof(float));
    // TODO: Handle errors
    float* host_memory_levels = new float[num_levels];
    host_memory_levels[0] = 0.0;
    host_memory_levels[num_levels - 1] = 1.0;
    if (multiplier_ < 0.0) {
      // Uniform levels
      float value = 1.0 / (num_levels - 1);
      for (int i = 0; i < num_levels - 1; i++) {
        host_memory_levels[i] = i * value;
      }
    } else {
      // Exponential levels
      float level_v = multiplier_;
      for (int i = 1; i < num_levels - 1; i++) {
        host_memory_levels[num_levels - 1 - i] = level_v;
        level_v *= multiplier_;
      }
    }
    if (global_state_->controller->GetLocalRank() == 0) {
      LOG(DEBUG) << host_memory_levels << " ";
    }
    cudaMemcpy((void*)levels_, (void*)host_memory_levels,
               num_levels * sizeof(float), cudaMemcpyHostToDevice);
    delete[] host_memory_levels;
  }
}

GPUNormLinfQuantizer::GPUNormLinfQuantizer(GPUContext* gpu_context,
                                           HorovodGlobalState* global_state,
                                           int quantization_bits,
                                           float multiplier)
    : GPUNormalizedQuantizer(gpu_context, global_state, quantization_bits,
                             multiplier) {
  if (global_state_->controller->GetRank() == 0) {
    if (multiplier < 0.0)
      LOG(INFO) << "GPUNormLinfQuantizer::Uni";
    else
      LOG(INFO) << "GPUNormLinfQuantizer::Exp";
  }
}

int64_t GPUNormLinfQuantizer::BufferSize(int num_elems) {
  int64_t compressed_values_buffer_size =
      round_to(ceil(1.0 * num_elems * bits_ / 8), ALIGNMENT_UNIT);
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size =
      round_to(sizeof(float) * num_buckets, ALIGNMENT_UNIT);
  return compressed_values_buffer_size + meta_buffer_size;
}

int64_t GPUNormLinfQuantizer::Compress(unsigned char* input_data,
                                       unsigned char* output,
                                       int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  int64_t meta_buffer_size =
      round_to(sizeof(float) * num_buckets, ALIGNMENT_UNIT);

  int64_t total_size = meta_buffer_size;
  unsigned char* meta_buffer = output;
  unsigned char* data = meta_buffer + meta_buffer_size;
  auto start = clock_::now();
  CUDA_find_Linf_bucket(
      (float*)input_data, (float*)meta_buffer, num_elems, bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  meta_info_time_ += time_since(start);

  start = clock_::now();
  CUDA_Linf_normalized_quantize_values(
      data, (float*)input_data, (float*)meta_buffer, levels_, num_elems, bits_,
      bucket_size_, cuda_states_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);

  int entries_per_byte = 8 / bits_;
  total_size += round_to((num_elems + entries_per_byte - 1) / entries_per_byte,
                         ALIGNMENT_UNIT);
  return total_size;
}

void GPUNormLinfQuantizer::Decompress(unsigned char* input,
                                      unsigned char* output,
                                      int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  int64_t meta_buffer_size =
      round_to(sizeof(float) * num_buckets, 2 * sizeof(float));
  auto start = clock_::now();
  CUDA_Linf_normalized_dequantize_values(
      input + meta_buffer_size, (float*)input, levels_, (float*)output,
      num_elems, bits_, bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
}

GPUNormL2Quantizer::GPUNormL2Quantizer(GPUContext* gpu_context,
                                       HorovodGlobalState* global_state,
                                       int quantization_bits, float multiplier)
    : GPUNormalizedQuantizer(gpu_context, global_state, quantization_bits,
                             multiplier) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "GPUNormL2Quantizer::Exp";
  }
}

int64_t GPUNormL2Quantizer::BufferSize(int num_elems) {
  int64_t compressed_values_buffer_size =
      round_to(ceil(1.0 * num_elems * bits_ / 8), ALIGNMENT_UNIT);
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to allocate max_log + norm
  int64_t norm_buffer_size =
      round_to(sizeof(float) * num_buckets, ALIGNMENT_UNIT);
  int64_t meta_buffer_size =
      round_to(sizeof(char) * num_buckets, ALIGNMENT_UNIT) + norm_buffer_size;
  return compressed_values_buffer_size + meta_buffer_size;
}

int64_t GPUNormL2Quantizer::Compress(unsigned char* input_data,
                                     unsigned char* output, int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  int64_t norm_buffer_size =
      round_to(sizeof(float) * num_buckets, ALIGNMENT_UNIT);
  int64_t meta_buffer_size =
      round_to(sizeof(char) * num_buckets, ALIGNMENT_UNIT) + norm_buffer_size;
  int64_t total_size = meta_buffer_size;

  unsigned char* meta_buffer = output;
  unsigned char* compressed_data = meta_buffer + meta_buffer_size;
  auto start = clock_::now();
  float* norms = (float*)meta_buffer;
  unsigned char* max_logs = (unsigned char*)meta_buffer + norm_buffer_size;
  CUDA_find_L2_and_max_log_bucket(
      (float*)input_data, norms, max_logs, 1.0 / multiplier_, num_elems,
      bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);

  meta_info_time_ += time_since(start);
  start = clock_::now();
  CUDA_L2_normalized_quantize_values(
      compressed_data, (float*)input_data, norms, max_logs, levels_, num_elems,
      bits_, bucket_size_, cuda_states_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
  int entries_per_byte = 8 / bits_;
  total_size += round_to((num_elems + entries_per_byte - 1) / entries_per_byte,
                         ALIGNMENT_UNIT);
  return total_size;
}

void GPUNormL2Quantizer::Decompress(unsigned char* input, unsigned char* output,
                                    int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  int64_t norm_buffer_size =
      round_to(sizeof(float) * num_buckets, ALIGNMENT_UNIT);
  int64_t meta_buffer_size =
      round_to(sizeof(char) * num_buckets, ALIGNMENT_UNIT) + norm_buffer_size;

  float* norms = (float*)input;
  unsigned char* max_logs = input + norm_buffer_size;
  auto start = clock_::now();
  CUDA_L2_normalized_dequantize_values(
      input + meta_buffer_size, norms, max_logs, levels_, (float*)output,
      num_elems, bits_, bucket_size_,
      gpu_context_->streams[global_state_->current_nccl_stream][device_]);
  compression_time_ += time_since(start);
}

} // namespace common
} // namespace horovod