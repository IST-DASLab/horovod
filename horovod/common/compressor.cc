#include "compressor.h"
#include "utils.h"
#include <assert.h>
#include <cstring>

namespace horovod {
namespace common {
Compressor::Compressor(horovod::common::CUDAContext* cuda_context)
    : cuda_context_(cuda_context) {
  auto env_str = getenv(HOROVOD_COMPRESSION_BUCKET_SIZE);
  if (env_str == nullptr)
    bucket_size_ = COMPRESSION_BUCKET_SIZE;
  else
    bucket_size_ = std::stol(std::string(env_str));
}

Status Quantizer::Init(HorovodGlobalState* globalState, int num_elements,
                       std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  bits_ = globalState->quantization_bits;

  num_elements_ = num_elements;
  device_ = first_entry.device;
  int64_t chunk_size = num_elements * sizeof(float);
  if (chunk_size < prev_chunk_size_) {
    return Status::OK();
  }
  chunk_size = fmaxf(num_elements * sizeof(float),
                     globalState->param_manager.TensorFusionThresholdBytes());

  int num_elems_in_chunk = ceil(((double)chunk_size) / sizeof(float));
  size_t curand_array_size = CUDA_get_curand_array_size(num_elems_in_chunk);

  Status status =
      bufferManager_.InitializeBuffer(curand_array_size, first_entry.device,
                                      first_entry.context, [&]() {}, [&]() {});
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }

  auto& buffer = bufferManager_.GetBuffer(first_entry.device,
                                          first_entry.context->framework());

  cuda_states_ = static_cast<CurandState*>(
      const_cast<void*>(buffer->AccessData(first_entry.context)));
  CUDA_init_curand(cuda_states_, num_elems_in_chunk, time(NULL),
                   cuda_context_->streams[device_]);
  return Status::OK();
}

Status
MaxMinQuantizer::Init(horovod::common::HorovodGlobalState* globalState,
                      int num_elements,
                      std::vector<horovod::common::TensorTableEntry>& entries) {
  Status status = Quantizer::Init(globalState, num_elements, entries);
  if (!status.ok()) {
    return status;
  }
  int64_t chunk_size = num_elements * sizeof(float);
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Maxmin buffer to use.
  // TODO: reimplement allocator/buffer manager in order to use buffers smarter.
  meta_buffer_size_ = 2 * num_buckets * sizeof(float);
  if (chunk_size < prev_chunk_size_) {
    return Status::OK();
  }
  // TODO: Deallocate prev chunk.
  // Chunk size that must be allocated.
  chunk_size = fmaxf(num_elements * sizeof(float),
                     globalState->param_manager.TensorFusionThresholdBytes());
  int64_t compressed_values_buffer_size =
      round_to(ceil(1.0 * chunk_size * globalState->quantization_bits /
                    (sizeof(float) * 8)),
               2 * sizeof(float));
  prev_chunk_size_ = chunk_size;

  // Max min buffer to allocate
  num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = 2 * sizeof(float) * num_buckets;
  allocate_buffer_size_ = compressed_values_buffer_size + meta_buffer_size;
  allocate_buffer_size_ = round_to(allocate_buffer_size_, 2 * sizeof(float));
  return Status::OK();
}

int64_t MaxMinQuantizer::Compress(void* input_data, void* compressed_data) {
  int data_size = meta_buffer_size_;
  int64_t start = now();
  CUDA_find_max_and_min_bucket((float*)input_data, (float*)compressed_data,
                               num_elements_, bucket_size_,
                               cuda_context_->streams[device_]);
  meta_info_time += now() - start;
  start = now();
  HERE CUDA_quantize_value_bits(
      (unsigned char*)(compressed_data) + meta_buffer_size_, (float*)input_data,
      (float*)compressed_data, num_elements_, bits_, bucket_size_, cuda_states_,
      cuda_context_->streams[device_]);
  compression_time += now() - start;
  int entries_per_byte = 8 / bits_;
  data_size += (num_elements_ + entries_per_byte - 1) / entries_per_byte;
  return round_to(data_size, 2 * sizeof(float));
}

void MaxMinQuantizer::Decompress(void* input_data, void* decompressed_data) {
  CUDA_dequantize_value_bits((unsigned char*)(input_data) + meta_buffer_size_,
                             (float*)input_data, (float*)decompressed_data,
                             num_elements_, bits_, bucket_size_,
                             cuda_context_->streams[device_]);
}


NormalizedQuantizer *CreateNormalized(CUDAContext *cuda_context) {
    std::cout << "NormalizedQuantizer" << std::endl;

    auto env_str = std::getenv(HOROVOD_QUANTIZATION_TYPE);
    if (env_str == nullptr || env_str[0] == 'u') {
        // Uniform quantization is reasonable to use only with Linf norm.
        std::cout << "Uniform quantization" << std::endl;
        return new NormLinfQuantizer(cuda_context);
    } else {
        using NType = NormalizedQuantizer::NormalizationType;
        auto norm_type = NType::Linf;
        if (strncmp(env_str, "eLi", 3) == 0) {
            std::cout << "Linf normalization ";
        } else if (strncmp(env_str, "eL2", 3) == 0) {
            std::cout << "L2 normalization ";
            norm_type = NType::L2;
        } else {
            std::cerr << "Wrong env var " << HOROVOD_QUANTIZATION_TYPE << " " << env_str << std::endl;
            exit(0);
        }
        float multiplier = QUANTIZE_MULTIPLIER;
        if (env_str[3] != '\0') {
            multiplier = std::strtof(env_str + 3, nullptr);
            if (multiplier == 0.0)
                multiplier = QUANTIZE_MULTIPLIER;
        }
        std::cout << "with multiplier " << multiplier << std::endl;
        if (norm_type == NType::Linf) {
            return new NormLinfQuantizer(cuda_context, multiplier);
        } else {
            return new NormL2Quantizer(cuda_context, multiplier);
        }
    }
}

Status NormLinfQuantizer::Init(
    horovod::common::HorovodGlobalState* globalState, int num_elements,
    std::vector<horovod::common::TensorTableEntry>& entries) {
  Status status = Quantizer::Init(globalState, num_elements, entries);
  if (!status.ok()) {
    return status;
  }
  int64_t chunk_size = num_elements * sizeof(float);
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  meta_buffer_size_ = round_to(sizeof(float) * num_buckets, 2 * sizeof(float));
  if (chunk_size < prev_chunk_size_) {
    return Status::OK();
  }

  chunk_size = fmaxf(num_elements * sizeof(float),
                     globalState->param_manager.TensorFusionThresholdBytes());
  int64_t compressed_values_buffer_size =
      round_to(ceil(1.0 * chunk_size * globalState->quantization_bits /
                    (sizeof(float) * 8)),
               2 * sizeof(float));
  num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to allocate max_log + norm
  int64_t meta_buffer_size =
          round_to(sizeof(float) * num_buckets, 2 * sizeof(float));
  allocate_buffer_size_ = compressed_values_buffer_size + meta_buffer_size;
  // Allocate auxiliary buffer for saving max values per bucket.
  prev_chunk_size_ = chunk_size;

  if (levels_ == nullptr) {
    int num_levels = 1 << (bits_ - 1);
    cudaMalloc((void**)&levels_, num_levels * sizeof(float));
    // TODO: Handle errors
    float* host_memory_levels = new float[num_levels];
    host_memory_levels[0] = 0.0;
    if (multiplier_ < 0.0) {
      float value = 1.0 / num_levels;
      for (int i = 0; i <= num_levels - 1; i++) {
        host_memory_levels[i] = i * value;
      }
    } else {
      float level_v = multiplier_;
      for (int i = 0; i < num_levels - 1; i++) {
        host_memory_levels[num_levels - 1 - i] = level_v;
        level_v *= multiplier_;
      }
    }
    cudaMemcpy((void*)levels_, (void*)host_memory_levels,
               num_levels * sizeof(float), cudaMemcpyHostToDevice);
    delete []host_memory_levels;
  }
  return Status::OK();
}


int64_t NormLinfQuantizer::Compress(void* input_data, void* compressed_data) {
  int data_size = meta_buffer_size_;
  void* meta_buffer = compressed_data;
  unsigned char* data = (unsigned char*)compressed_data + meta_buffer_size_;
  int64_t start = now();
  CUDA_find_Linf_bucket((float *) input_data, (float *) meta_buffer,
                      num_elements_, bucket_size_,
                      cuda_context_->streams[device_]);
//    CUDA_find_norms_bucket((float*)input_data, (float*)meta_buffer, max_buffer_,
//                           num_elements_, bucket_size_,
//                           cuda_context_->streams[device_]);

  meta_info_time += now() - start;
  start = now();
  HERE
  CUDA_Linf_normalized_quantize_values(
          data, (float *) input_data, (float *) meta_buffer, levels_, num_elements_,
          bits_, bucket_size_, cuda_states_, cuda_context_->streams[device_]);
  compression_time += now() - start;
  HERE
  int entries_per_byte = 8 / bits_;
  data_size += (num_elements_ + entries_per_byte - 1) / entries_per_byte;
  return round_to(data_size, 2 * sizeof(float));
}


void NormLinfQuantizer::Decompress(void* compressed_data,
                                     void* decompressed_data) {
  CUDA_Linf_normalized_dequantize_values(
          (unsigned char *) compressed_data + meta_buffer_size_,
          (float *) compressed_data, levels_, (float *) decompressed_data,
          num_elements_, bits_, bucket_size_, cuda_context_->streams[device_]);
}

Status NormL2Quantizer::Init(
        horovod::common::HorovodGlobalState* globalState, int num_elements,
        std::vector<horovod::common::TensorTableEntry>& entries) {
  Status status = Quantizer::Init(globalState, num_elements, entries);
  if (!status.ok()) {
    return status;
  }
  HERE
  int64_t chunk_size = num_elements * sizeof(float);
  int64_t num_buckets =
          (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  norm_buffer_size_ = round_to(sizeof(float) * num_buckets, 2 * sizeof(float));
  meta_buffer_size_ = round_to(sizeof(char) * num_buckets, 2 * sizeof(float)) + norm_buffer_size_;
  if (chunk_size < prev_chunk_size_) {
    return Status::OK();
  }
  HERE
  chunk_size = fmaxf(num_elements * sizeof(float),
                     globalState->param_manager.TensorFusionThresholdBytes());
  int64_t compressed_values_buffer_size =
          round_to(ceil(1.0 * chunk_size * globalState->quantization_bits / (sizeof(float) * 8)),2 * sizeof(float));
  num_buckets =
          (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to allocate max_log + norm
  int64_t meta_buffer_size =
          round_to(sizeof(float) * num_buckets, 2 * sizeof(float)) + round_to(sizeof(char) * num_buckets,
                                                                              2 * sizeof(float));
  allocate_buffer_size_ = compressed_values_buffer_size + meta_buffer_size;
  // Allocate auxiliary buffer for saving max values per bucket.
  prev_chunk_size_ = chunk_size;

  if (levels_ == nullptr) {
    int num_levels = 1 << (bits_ - 1);
    // In case of L2 normalization we can do following heuristic.
    // We store save min power of multiplier that is more than evey normalized value in bucket
    // and quantize starting with this power not with 1. With that we can achieve better accuracy, e.g. there is no need
    // to quantize starting 0.5 point because it's likely no values lie in range [0.5, 1.0].
    // We precompute 4 * levels in advance.
    num_levels *= 4;
    cudaMalloc((void**)&levels_, num_levels * sizeof(float));
    // TODO: Handle errors
    float* host_memory_levels = new float[num_levels];
    float level_v = multiplier_;
    for (int i = 0; i < num_levels - 1; i++) {
      host_memory_levels[i] = level_v;
      level_v *= multiplier_;
    }
    cudaMemcpy((void*)levels_, (void*)host_memory_levels,
               num_levels * sizeof(float), cudaMemcpyHostToDevice);
    delete []host_memory_levels;
  }
  HERE
  return Status::OK();
}

int64_t NormL2Quantizer::Compress(void* input_data, void* output) {
  int data_size = meta_buffer_size_;
  void* meta_buffer = output;
  unsigned char* compressed_data = (unsigned char*)meta_buffer + meta_buffer_size_;
  int64_t start = now();
  float *norms = (float *) meta_buffer;
  unsigned char* max_logs = (unsigned char *) meta_buffer + norm_buffer_size_;
  CUDA_find_L2_and_max_log_bucket((float *) input_data, norms, max_logs,
                        1.0 / multiplier_, num_elements_, bucket_size_,
                        cuda_context_->streams[device_]);

  meta_info_time += now() - start;
  start = now();
  HERE
  CUDA_L2_normalized_quantize_values(
          compressed_data, (float *) input_data, norms, max_logs, levels_, num_elements_,
          bits_, bucket_size_, cuda_states_, cuda_context_->streams[device_]);
  compression_time += now() - start;
  HERE
  int entries_per_byte = 8 / bits_;
  data_size += (num_elements_ + entries_per_byte - 1) / entries_per_byte;
  return round_to(data_size, 2 * sizeof(float));
}

void NormL2Quantizer::Decompress(void* compressed_data,
                                   void* decompressed_data) {
  float *norms = (float *)compressed_data;
  unsigned char *max_logs = (unsigned char *)compressed_data + norm_buffer_size_;
  CUDA_L2_normalized_dequantize_values(
          (unsigned char *) compressed_data + meta_buffer_size_,
          norms, max_logs, levels_, (float *) decompressed_data,
          num_elements_, bits_, bucket_size_, cuda_context_->streams[device_]);
}


Status
TopKcompressor::Init(horovod::common::HorovodGlobalState* globalState,
                     int num_elements,
                     std::vector<horovod::common::TensorTableEntry>& entries) {
  auto topk = std::getenv(HOROVOD_TOPK);
  if (topk != nullptr) {
    taken_amount = std::strtof(topk, nullptr);
  } else {
    taken_amount = DEFAULT_TOPK;
  }
}

} // namespace common
} // namespace horovod