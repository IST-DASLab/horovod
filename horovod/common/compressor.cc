#include "compressor.h"
#include "utils.h"
#include <assert.h>
#include <cstring>
#include "logging.h"

namespace horovod {
namespace common {

Compressor::Compressor(){
  auto env_str = getenv(HOROVOD_COMPRESSION_BUCKET_SIZE);
  if (env_str == nullptr)
    bucket_size_ = COMPRESSION_BUCKET_SIZE;
  else
    bucket_size_ = std::stol(std::string(env_str));
}

int64_t DummyCompressor::Compress(unsigned char* input_data, void** output_p,
                                  int64_t num_elems) {
  HERE
  *((unsigned char **)output_p) = input_data;
  return num_elems * sizeof(float);
}

void DummyCompressor::Decompress(unsigned char* input, void** output_p,
                                 int64_t num_elems) {
  HERE
  *((unsigned char **)output_p) = input;
}

CUDAQuantizer::CUDAQuantizer(
    CUDAContext* cuda_context,
    HorovodGlobalState* global_state):CUDACompressor(cuda_context) {
  bits_ = global_state->quantization_bits;
}

Status CUDAQuantizer::Init(HorovodGlobalState* global_state,
                           const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  device_ = first_entry.device;

  int64_t chunk_size = global_state->param_manager.TensorFusionThresholdBytes();
  int num_elems_in_chunk = ceil(((double)chunk_size) / sizeof(float));
  curand_array_size = CUDA_get_curand_array_size(num_elems_in_chunk);
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

int64_t CUDAMaxMinQuantizer::BufferSize(int chunk_size) {
  int64_t compressed_values_buffer_size =
      ceil(1.0 * chunk_size * bits_ /
           (sizeof(float) * 8));
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Max min buffer to allocate
  int64_t meta_buffer_size = 2 * sizeof(float) * num_buckets;
  return compressed_values_buffer_size + meta_buffer_size;
}

int64_t CUDAMaxMinQuantizer::Compress(unsigned char* input_data,
                                      void** output_p, int64_t num_elems) {
  int64_t chunk_size = num_elems * sizeof(float);
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Maxmin buffer to use.
  int64_t meta_buffer_size = 2 * num_buckets * sizeof(float);
  int64_t start = now();
  unsigned char* meta_buffer = *((unsigned char **)output_p);
  CUDA_find_max_and_min_bucket((float*)input_data, (float*)meta_buffer,
                               num_elems, bucket_size_,
                               cuda_context_->streams[device_]);
  meta_info_time += now() - start;
  start = now();
  HERE
  CUDA_quantize_value_bits(
      meta_buffer + meta_buffer_size, (float*)input_data,
      (float*)meta_buffer, num_elems, bits_, bucket_size_, cuda_states_,
      cuda_context_->streams[device_]);
  compression_time += now() - start;
  int entries_per_byte = 8 / bits_;
  int64_t total_size = meta_buffer_size + (num_elems + entries_per_byte - 1) / entries_per_byte;
  return total_size;
}

void CUDAMaxMinQuantizer::Decompress(unsigned char* input_data, void** output_p,
                                     int64_t num_elems) {
  unsigned char* output = *((unsigned char **)output_p);

  int64_t chunk_size = num_elems * sizeof(float);
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Maxmin buffer to use.
  int64_t meta_buffer_size = 2 * num_buckets * sizeof(float);

  CUDA_dequantize_value_bits(input_data + meta_buffer_size,
                             (float*)input_data, (float*)output,
                             num_elems, bits_, bucket_size_,
                             cuda_context_->streams[device_]);
}


CUDANormalizedQuantizer *CreateCUDANormalized(CUDAContext* cuda_context, HorovodGlobalState* global_state) {
    LOG(INFO) << "NormalizedQuantizer";

    auto env_str = std::getenv(HOROVOD_QUANTIZATION_TYPE);
    if (env_str == nullptr || env_str[0] == 'u') {
        // Uniform quantization is reasonable to use only with Linf norm.
        LOG(INFO) << "Uniform quantization";
        return new CUDANormLinfQuantizer(cuda_context, global_state);
    } else {
        using NType = CUDANormalizedQuantizer::NormalizationType;
        auto norm_type = NType::Linf;
        if (strncmp(env_str, "eLi", 3) == 0) {
            LOG(INFO) << "Linf normalization ";
        } else if (strncmp(env_str, "eL2", 3) == 0) {
            LOG(INFO) << "L2 normalization ";
            norm_type = NType::L2;
        } else {
            LOG(ERROR) << "Wrong env var " << HOROVOD_QUANTIZATION_TYPE << " " << env_str << std::endl;
            exit(0);
        }
        float multiplier = QUANTIZE_MULTIPLIER;
        if (env_str[3] != '\0') {
            multiplier = std::strtof(env_str + 3, nullptr);
            if (multiplier == 0.0)
                multiplier = QUANTIZE_MULTIPLIER;
        }
        LOG(INFO) << "with multiplier " << multiplier;
        if (norm_type == NType::Linf) {
            return new CUDANormLinfQuantizer(cuda_context, global_state, multiplier);
        } else {
            return new CUDANormL2Quantizer(cuda_context, global_state, multiplier);
        }
    }
}

int64_t CUDANormLinfQuantizer::BufferSize(int chunk_size) {
  int64_t compressed_values_buffer_size =
      round_to(ceil(1.0 * chunk_size * bits_ /
                    (sizeof(float) * 8)),
               2 * sizeof(float));
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size =
      round_to(sizeof(float) * num_buckets, 2 * sizeof(float));
  return compressed_values_buffer_size + meta_buffer_size;
}

Status CUDANormLinfQuantizer::Init(HorovodGlobalState* globalState,
                            const std::vector<TensorTableEntry>& entries) {
  Status status = CUDAQuantizer::Init(globalState, entries);
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
      float value = 1.0 / (num_levels - 1);
      for (int i = 0; i < num_levels - 1; i++) {
        host_memory_levels[i] = i * value;
      }
    } else {
      float level_v = multiplier_;
      for (int i = 1; i < num_levels - 1; i++) {
        host_memory_levels[num_levels - 1 - i] = level_v;
        level_v *= multiplier_;
      }
    }
    if (globalState->rank == 0) {
      for (int i = 0; i < num_levels; i++) {
        std::cout << host_memory_levels[i] << " ";
      }
      std::cout << std::endl;
    }
    cudaMemcpy((void*)levels_, (void*)host_memory_levels,
               num_levels * sizeof(float), cudaMemcpyHostToDevice);
    delete []host_memory_levels;
  }
  return Status::OK();
}

int64_t CUDANormLinfQuantizer::Compress(unsigned char* input_data,
                                        void** output_p, int64_t num_elems) {
  int64_t chunk_size = num_elems * sizeof(float);
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  int64_t meta_buffer_size = round_to(sizeof(float) * num_buckets, 2 * sizeof(float));

  int64_t total_size = meta_buffer_size;
  unsigned char* meta_buffer = *((unsigned char **) output_p);
  unsigned char* data = meta_buffer + meta_buffer_size;
  int64_t start = now();
  CUDA_find_Linf_bucket((float *) input_data, (float *) meta_buffer,
                      num_elems, bucket_size_,
                      cuda_context_->streams[device_]);
//    CUDA_find_norms_bucket((float*)input_data, (float*)meta_buffer, max_buffer_,
//                           num_elements_, bucket_size_,
//                           cuda_context_->streams[device_]);

  meta_info_time += now() - start;
  start = now();
  HERE
  CUDA_Linf_normalized_quantize_values(
          data, (float *) input_data, (float *) meta_buffer, levels_, num_elems,
          bits_, bucket_size_, cuda_states_, cuda_context_->streams[device_]);
  compression_time += now() - start;
  HERE
  int entries_per_byte = 8 / bits_;
  total_size += (num_elems + entries_per_byte - 1) / entries_per_byte;
  return total_size;
}

void CUDANormLinfQuantizer::Decompress(unsigned char* input, void** output_p,
                                       int64_t num_elems) {
  unsigned char* output = *((unsigned char **) output_p);

  int64_t chunk_size = num_elems * sizeof(float);
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  int64_t meta_buffer_size = round_to(sizeof(float) * num_buckets, 2 * sizeof(float));

  CUDA_Linf_normalized_dequantize_values(
          input + meta_buffer_size,
          (float *) input, levels_, (float *) output,
          num_elems, bits_, bucket_size_, cuda_context_->streams[device_]);
}

int64_t CUDANormL2Quantizer::BufferSize(int chunk_size) {
  int64_t compressed_values_buffer_size =
      ceil(1.0 * chunk_size * bits_ / (sizeof(float) * 8));
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to allocate max_log + norm
  int64_t norm_buffer_size = round_to(sizeof(float) * num_buckets, ALIGNMENT_UNIT);
  int64_t meta_buffer_size = round_to(sizeof(char) * num_buckets, ALIGNMENT_UNIT) + norm_buffer_size;
  return compressed_values_buffer_size + meta_buffer_size;
}

Status CUDANormL2Quantizer::Init(HorovodGlobalState* globalState,
                                 const std::vector<TensorTableEntry>& entries) {
  Status status = CUDAQuantizer::Init(globalState, entries);
  if (!status.ok()) {
    return status;
  }
  HERE
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
    float *host_memory_levels = new float[num_levels];
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

int64_t CUDANormL2Quantizer::Compress(unsigned char* input_data,
                                      void** output_p, int64_t num_elems) {
  int64_t chunk_size = num_elems * sizeof(float);
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  int64_t norm_buffer_size = round_to(sizeof(float) * num_buckets, ALIGNMENT_UNIT);
  int64_t meta_buffer_size = round_to(sizeof(char) * num_buckets, ALIGNMENT_UNIT) + norm_buffer_size;
  int64_t total_size = meta_buffer_size;

  unsigned char* meta_buffer = *((unsigned char **) output_p);
  unsigned char* compressed_data = (unsigned char*) meta_buffer + meta_buffer_size;
  int64_t start = now();
  float* norms = (float *) meta_buffer;
  unsigned char* max_logs = (unsigned char*) meta_buffer + norm_buffer_size;
  CUDA_find_L2_and_max_log_bucket((float *) input_data, norms, max_logs,
                        1.0 / multiplier_, num_elems, bucket_size_,
                        cuda_context_->streams[device_]);

  meta_info_time += now() - start;
  start = now();
  HERE
  CUDA_L2_normalized_quantize_values(
          compressed_data, (float *) input_data, norms, max_logs, levels_, num_elems,
          bits_, bucket_size_, cuda_states_, cuda_context_->streams[device_]);
  compression_time += now() - start;
  HERE
  int entries_per_byte = 8 / bits_;
  total_size += (num_elems + entries_per_byte - 1) / entries_per_byte;
  return total_size;
}

void CUDANormL2Quantizer::Decompress(unsigned char* input, void** output_p,
                                     int64_t num_elems) {
  unsigned char* output = *((unsigned char **) output_p);

  int64_t chunk_size = num_elems * sizeof(float);
  int64_t num_buckets =
      (chunk_size + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to use.
  int64_t norm_buffer_size = round_to(sizeof(float) * num_buckets, ALIGNMENT_UNIT);
  int64_t meta_buffer_size = round_to(sizeof(char) * num_buckets, ALIGNMENT_UNIT) + norm_buffer_size;

  float *norms = (float *)input;
  unsigned char *max_logs = input + norm_buffer_size;
  CUDA_L2_normalized_dequantize_values(
          input + meta_buffer_size,
          norms, max_logs, levels_, (float *) output,
          num_elems, bits_, bucket_size_, cuda_context_->streams[device_]);
}

} // namespace common
} // namespace horovod