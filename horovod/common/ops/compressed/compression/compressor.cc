#include "compressor.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <yaml.h>

#include "../../../logging.h"
#include "../../../utils/env_parser.h"
#include "../common.h"
#include "../utils.h"

#include "../reducers/reducer.h"

namespace horovod {
namespace common {

const int PACK_SIZE = 8;
const float EPS = 1e-6;

void PackBucket(float* input, unsigned char* output, float* feedback,
                int num_elems, int bits,
                std::function<unsigned char(float, float*)> encode);
void UnpackBucket(unsigned char* input, float* output, int num_elems, int bits,
                  bool add, std::function<float(unsigned char)> decode);

void Compressor::SetQuantizationLevels(float* levels, int bits) {
  std::cout << "I don't set anything" << std::endl;
}

size_t Compressor::GetRequiredFreeSize() { return 0; }

Compressor::Compressor(HorovodGlobalState* global_state, Summator* summator)
    : global_state_(global_state), initialized_(false),
      error_feedback_(summator) {
  default_config.bucket_size = GetIntEnvOrDefault(
      HOROVOD_COMPRESSION_BUCKET_SIZE, COMPRESSION_BUCKET_SIZE);
  default_config.skip_incomplete_buckets = false;
  SetBoolFromEnv(HOROVOD_COMPRESSION_SKIP_INCOMPLETE_BUCKETS,
                 default_config.skip_incomplete_buckets, true);
  compression_mode_ =
      GetEnumEnvOrDefault(HOROVOD_COMPRESSION_MODE, CompressionMode::NonFused);
  const char* config_filename = std::getenv(HOROVOD_COMPRESSION_CONFIG_FILE);
  if (config_filename != nullptr) {
    ParseYaml(config_filename);
  }
}

Status Compressor::Init(const std::vector<TensorTableEntry>& entries) {
  error_feedback_.Init(entries);
  if (compression_mode_ == CompressionMode::Fused and
      error_feedback_.isEnabled() and fused_feedback_buf == nullptr) {
    CUDA_CHECK(cudaMalloc(
        &fused_feedback_buf,
        global_state_->parameter_manager.TensorFusionThresholdBytes()));
    cudaMemset(fused_feedback_buf, 0,
               global_state_->parameter_manager.TensorFusionThresholdBytes());
  }
  initialized_ = true;
  return Status::OK();
}

CompressionModuleConfig& Compressor::GetModuleConfig(const std::string& name) {
  for (auto& module : modules_configs) {
    if (name.find(module.first) != std::string::npos) {
      auto& config = module.second;
      config.quantization_bits = (config.quantization_bits > 0)
                                     ? config.quantization_bits
                                     : default_config.quantization_bits;
      config.bucket_size = (config.bucket_size > 0)
                               ? config.bucket_size
                               : default_config.bucket_size;
      config.skip_incomplete_buckets = default_config.skip_incomplete_buckets;
      return config;
    }
  }
  return default_config;
}

void Compressor::GetSizesAndOffsets(
    int num_elements, int world_size,
    const std::vector<TensorTableEntry>& entries, std::vector<int>& offsets,
    std::vector<int>& sizes) {
  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
  int offset = 0;
  for (int rank = 0; rank < world_size; rank++) {
    sizes.push_back(num_elems_per_node + ((rank < residue) ? 1 : 0));
    offsets.push_back(offset);
    offset += sizes.back();
  }
}

size_t Compressor::BufferSize(
    int chunk_num_elems,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int fusion_offset) {
  auto dtype = entries[0].tensor->dtype();
  if (compression_mode_ == CompressionMode::Fused) {
    return BufferSize(chunk_num_elems, entries[0].tensor->dtype(),
                      default_config);
  }
  int offset_cumm = 0;
  int nelem = 0;
  size_t sum_result = 0;
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
    sum_result += BufferSize(nelem, dtype, GetModuleConfig(entry.tensor_name));
    offset_cumm += entry.tensor->shape().num_elements();
  }
  return sum_result;
}

size_t Compressor::Compress(
    unsigned char* input_data, unsigned char* output,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int fusion_offset, int global_offset, int chunk_num_elems, bool disable_error_feedback,
    void* ctx) {
  auto dtype = entries[0].tensor->dtype();
  if (compression_mode_ == CompressionMode::Fused) {
    unsigned char* feedback_data = nullptr;
    if (!disable_error_feedback && error_feedback_.isEnabled()) {
      if (entries.size() == 1) {
        feedback_data = error_feedback_.GetData(entries[0]) +
            (fusion_offset + global_offset) * get_sizeof(dtype);
      } else {
        feedback_data = fused_feedback_buf;
      }
    }
    auto result = CompressBuffer(input_data + fusion_offset * get_sizeof(dtype),
                                 output, feedback_data, chunk_num_elems, dtype,
                                 default_config, ctx);
    if (!disable_error_feedback && error_feedback_.isEnabled() &&
        entries.size() > 1) {
      error_feedback_.CopyToErrorFeedback(feedback_data, entries,
                                          chunk_num_elems, fusion_offset, ctx);
    }
    return result;
  } else if (compression_mode_ == CompressionMode::PerEntryFused or
             input_data != nullptr) {
    return CompressPerEntry(input_data + fusion_offset * get_sizeof(dtype),
                            output, entries, fusion_offset, chunk_num_elems,
                            disable_error_feedback, ctx);
  } else {
    return CompressFromEntries(output, entries, fusion_offset, chunk_num_elems,
                               disable_error_feedback, ctx);
  }
}

void Compressor::Decompress(unsigned char* input_data, unsigned char* output,
                            const std::vector<TensorTableEntry>& entries,
                            int fusion_offset, int chunk_num_elems, bool add,
                            void* ctx) {
  auto dtype = entries[0].tensor->dtype();
  if (compression_mode_ == CompressionMode::Fused) {
    DecompressBuffer(input_data, output + fusion_offset * get_sizeof(dtype),
                     chunk_num_elems, dtype, add, default_config, ctx);
  } else if (compression_mode_ == CompressionMode::PerEntryFused or
             output != nullptr) {
    DecompressPerEntry(input_data, output + fusion_offset * get_sizeof(dtype),
                       entries, fusion_offset, chunk_num_elems, add, ctx);
  } else {
    DecompressIntoEntries(input_data, entries, fusion_offset, chunk_num_elems,
                          add, ctx);
  }
}

size_t Compressor::CompressPerEntry(
    unsigned char* input_data, unsigned char* output,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int fusion_offset, int chunk_num_elems, bool disable_error_feedback,
    void* ctx) {
  size_t total_compressed_size = 0;
  auto dtype = entries[0].tensor->dtype();
  int offset_cumm = 0;
  int nelem = 0;
  int buffer_offset = 0, entry_offset = 0;
  size_t compressed_size;
  for (auto& entry : entries) {
    nelem = entry.tensor->shape().num_elements();
    entry_offset = 0;
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
      entry_offset = entry.tensor->shape().num_elements() - nelem;
    }

    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
              std::max(offset_cumm, fusion_offset);
    }

    buffer_offset = std::max(offset_cumm - fusion_offset, 0);
    auto offset = buffer_offset * get_sizeof(dtype);
    unsigned char* feedback_data = nullptr;
    if (!disable_error_feedback && error_feedback_.isEnabled())
      feedback_data =
          error_feedback_.GetData(entry) + entry_offset * get_sizeof(dtype);
    compressed_size =
        CompressBuffer(input_data + offset, output, feedback_data, nelem, dtype,
                       GetModuleConfig(entry.tensor_name), ctx);
    offset_cumm += entry.tensor->shape().num_elements();
    output += compressed_size;
    total_compressed_size += compressed_size;
  }
  return total_compressed_size;
}

void Compressor::DecompressPerEntry(
    unsigned char* input_data, unsigned char* output_data,
    const std::vector<TensorTableEntry>& entries, int fusion_offset,
    int chunk_num_elems, bool add, void* ctx) {
  auto dtype = entries[0].tensor->dtype();
  int offset_cumm = 0;
  int nelem = 0;
  int buffer_offset = 0;
  size_t cumm_decompressed = 0;
  unsigned char* output;
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
    auto& module_config = GetModuleConfig(entry.tensor_name);
    buffer_offset = std::max(offset_cumm - fusion_offset, 0);
    output = output_data + buffer_offset * get_sizeof(dtype);
    DecompressBuffer(input_data + cumm_decompressed, output, nelem, dtype, add,
                     module_config, ctx);
    cumm_decompressed += BufferSize(nelem, dtype, module_config);
    offset_cumm += entry.tensor->shape().num_elements();
  }
}

size_t Compressor::CompressFromEntries(
    unsigned char* output,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int fusion_offset, int chunk_num_elems, bool disable_error_feedback,
    void* ctx) {
  size_t total_compressed_size = 0;
  auto dtype = entries[0].tensor->dtype();

  int offset_cumm = 0;
  int nelem = 0;
  int buffer_offset = 0;
  size_t compressed_size;
  for (auto& entry : entries) {
    nelem = entry.tensor->shape().num_elements();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }

    if (offset_cumm - fusion_offset >= chunk_num_elems) {
      break;
    }
    buffer_offset = 0;
    if (offset_cumm < fusion_offset) {
      // If the first part of the entry is placed in the previous slice.
      nelem = offset_cumm + nelem - fusion_offset;
      buffer_offset = entry.tensor->shape().num_elements() - nelem;
    }

    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if entry doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
              std::max(offset_cumm, fusion_offset);
    }
    auto offset = buffer_offset * get_sizeof(dtype);
    auto tensor_data = ((unsigned char*)entry.output->data()) + offset;
    unsigned char* feedback_data = nullptr;

    if (!disable_error_feedback && error_feedback_.isEnabled())
      feedback_data = error_feedback_.GetData(entry) + offset;
    compressed_size =
        CompressBuffer(tensor_data, output, feedback_data, nelem, dtype,
                       GetModuleConfig(entry.tensor_name), ctx);
    offset_cumm += entry.tensor->shape().num_elements();
    output += compressed_size;
    total_compressed_size += compressed_size;
  }
  return total_compressed_size;
}

void Compressor::DecompressIntoEntries(
    unsigned char* input_data,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int fusion_offset, int chunk_num_elems, bool add, void* ctx) {
  auto dtype = entries[0].output->dtype();
  int offset_cumm = 0;
  int nelem = 0;
  int buffer_offset = 0;
  size_t cumm_decompressed = 0;

  for (auto& entry : entries) {
    nelem = entry.tensor->shape().num_elements();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }
    if (offset_cumm - fusion_offset >= chunk_num_elems)
      break;
    buffer_offset = 0;
    if (offset_cumm < fusion_offset) {
      // If the first part of param group is placed in previous slice
      // depending on reduction algorithm.
      nelem = offset_cumm + nelem - fusion_offset;
      buffer_offset = entry.tensor->shape().num_elements() - nelem;
    }
    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
              std::max(offset_cumm, fusion_offset);
    }
    auto output = ((unsigned char*)entry.output->data()) +
                  buffer_offset * get_sizeof(dtype);
    const auto& module_config = GetModuleConfig(entry.tensor_name);
    DecompressBuffer(input_data + cumm_decompressed, output, nelem, dtype, add,
                     module_config, ctx);
    cumm_decompressed += BufferSize(nelem, dtype, module_config);
    offset_cumm += entry.tensor->shape().num_elements();
  }
}

void Compressor::ApplyErrorFeedback(std::vector<TensorTableEntry>& entries) {
  error_feedback_.Apply(entries);
}

// ================
// Dummy Compressor
// ================
size_t
DummyCompressor::BufferSize(int num_elems, horovod::common::DataType dtype,
                            const CompressionModuleConfig& compression_cfg) {
  return num_elems * get_sizeof(dtype);
}

size_t CPUDummyCompressor::CompressBuffer(
    unsigned char* input_data, unsigned char* output,
    unsigned char* feedback_data, int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg, void* ctx) {
  assert(dtype == DataType::HOROVOD_FLOAT32);
  size_t processed_size = num_elems * sizeof(float);
  std::memcpy(output, input_data, processed_size);
  return processed_size;
}

void CPUDummyCompressor::DecompressBuffer(
    unsigned char* input_data, unsigned char* output, int num_elems,
    DataType dtype, bool add, const CompressionModuleConfig& compression_cfg,
    void* ctx) {
  assert(dtype == DataType::HOROVOD_FLOAT32);
  if (add) {
    float* output_f = (float*)output;
    float* input_f = (float*)input_data;
    for (int i = 0; i < num_elems; i++)
      output_f[i] += input_f[i];
  } else {
    size_t processed_size = num_elems * sizeof(float);
    std::memcpy(output, input_data, processed_size);
  }
}

// ================
// Quantizers
// ================
Quantizer::Quantizer(horovod::common::HorovodGlobalState* global_state,
                     Summator* summator, int quantization_bits)
    : Compressor(global_state, summator) {
  default_config.quantization_bits = quantization_bits;
}

void Quantizer::GetSizesAndOffsets(int num_elements, int world_size,
                                   const std::vector<TensorTableEntry>& entries,
                                   std::vector<int>& offsets,
                                   std::vector<int>& sizes) {
  if (default_config.quantization_bits == 32) {
    Compressor::GetSizesAndOffsets(num_elements, world_size, entries, offsets,
                                   sizes);
    return;
  }
  int offset = 0;
  int num_per_node;
  auto it = entries.begin();
  int entry_offset = 0;
  int n_elem = std::min((int)it->tensor->shape().num_elements(), num_elements);
  int cur_size = 0;
  for (int rank = 0; rank < world_size; rank++) {
    num_per_node = num_elements / (world_size - rank);
    cur_size = 0;
    while (cur_size < num_per_node) {
      if (n_elem <= num_per_node - cur_size) {
        cur_size += n_elem;
        it++;
        if (it == entries.end())
          break;
        n_elem =
            std::min((int)it->tensor->shape().num_elements(), num_elements);
      } else {
        int aligned =
            std::min((int)round_to(num_per_node - cur_size, 4), n_elem);
        cur_size += aligned;
        n_elem -= aligned;
      }
    }
    num_elements -= cur_size;
    sizes.push_back(cur_size);
    offsets.push_back(offset);
    offset += cur_size;
  }
}
// ================
// Max Min Quantizer
// ================
inline size_t
CPUMaxMinQuantizer::BufferSize(int num_elems, DataType dtype,
                               const CompressionModuleConfig& compression_cfg) {
  assert(dtype == DataType::HOROVOD_FLOAT32);
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  size_t compressed_values_buffer_size = (num_elems * bits + 7) / 8;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  // Max min buffer to allocate
  size_t meta_buffer_size = 2 * sizeof(float) * num_buckets;
  return compressed_values_buffer_size + meta_buffer_size;
}

size_t CPUMaxMinQuantizer::CompressBuffer(
    unsigned char* input_data, unsigned char* output,
    unsigned char* feedback_data, int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg, void* ctx) {
  assert(dtype == DataType::HOROVOD_FLOAT32);
  int bits = compression_cfg.quantization_bits;
  int bucket_size = compression_cfg.bucket_size;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  size_t meta_buffer_size = 2 * sizeof(float) * num_buckets;
  size_t compressed_values_buffer_size = (num_elems * bits + 7) / 8;

  auto meta_info_buffer = (float*)output;
  output += meta_buffer_size;
  for (int bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    CompressBucket(
        input_data, meta_info_buffer, output, feedback_data, compression_cfg,
        std::min(bucket_size, num_elems - bucket_no * bucket_size), bucket_no);
  }
  return compressed_values_buffer_size + meta_buffer_size;
}

void CPUMaxMinQuantizer::DecompressBuffer(
    unsigned char* input_data, unsigned char* output, int num_elems,
    DataType dtype, bool add, const CompressionModuleConfig& compression_cfg,
    void* ctx) {
  assert(dtype == DataType::HOROVOD_FLOAT32);
  int bits = compression_cfg.quantization_bits;
  int bucket_size = compression_cfg.bucket_size;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  int meta_buffer_size = 2 * sizeof(float) * num_buckets;

  auto meta_info_buffer = (float*)input_data;
  input_data += meta_buffer_size;
  for (int bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    DecompressBucket(input_data, meta_info_buffer, output, compression_cfg,
                     std::min(bucket_size, num_elems - bucket_no * bucket_size),
                     bucket_no, add);
  }
}

void CPUMaxMinQuantizer::CompressBucket(
    unsigned char* input_data, float* meta_info_buffer, unsigned char* output,
    unsigned char* feedback_data,
    const CompressionModuleConfig& compression_cfg, int num_elems,
    int bucket_no) {
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  float* input = ((float*)input_data) + bucket_no * bucket_size;
  float* feedback = nullptr;
  if (feedback_data)
    feedback = ((float*)feedback_data) + bucket_no * bucket_size;
  // number of bits in char.
  output =
      output + (bucket_size * bucket_no * bits + PACK_SIZE - 1) / PACK_SIZE;

  float max = input[0];
  float min = input[0];
  for (int i = 1; i < num_elems; i++) {
    max = std::max(max, input[i]);
    min = std::min(min, input[i]);
  }

  meta_info_buffer[2 * bucket_no] = max;
  meta_info_buffer[2 * bucket_no + 1] = min;
  int divisor = (1 << bits) - 1;
  float unit = (max - min) / divisor;
  std::function<unsigned char(float, float*)> encode =
      std::bind(&CPUMaxMinQuantizer::EncodeValue, this, std::placeholders::_1,
                std::placeholders::_2, min, unit);
  PackBucket(input, output, feedback, num_elems, bits, encode);
}

unsigned char CPUMaxMinQuantizer::EncodeValue(float v, float* feedback,
                                              float min, float unit) {
  float rand = randomizer.GetRand();
  float d = ((v - min) / unit) + rand;
  unsigned char level = (unsigned char)floor(d);
  // update error feedback
  if (feedback)
    *feedback = v - (min + level * unit);
  return level;
}

void CPUMaxMinQuantizer::DecompressBucket(
    unsigned char* input_data, float* meta_info_buffer,
    unsigned char* output_data, const CompressionModuleConfig& compression_cfg,
    int num_elems, int bucket_no, bool add) {
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  float* output = (float*)output_data + bucket_no * bucket_size;
  // number of bits in char.
  const int divisor = (1 << bits) - 1;
  float max = meta_info_buffer[2 * bucket_no];
  float min = meta_info_buffer[2 * bucket_no + 1];
  float unit = (max - min) / divisor;
  input_data =
      input_data + (bucket_size * bucket_no * bits + PACK_SIZE - 1) / PACK_SIZE;
  std::function<float(unsigned char)> decode = std::bind(
      &CPUMaxMinQuantizer::DecodeValue, this, std::placeholders::_1, min, unit);

  UnpackBucket(input_data, output, num_elems, bits, add, decode);
}

float CPUMaxMinQuantizer::DecodeValue(unsigned char input, float min,
                                      float unit) {
  return min + unit * input;
}

// ================
// Normalized Quantizers
// ================
Status CPUNormalizedQuantizer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  for (auto& entry : entries) {
    auto& config = GetModuleConfig(entry.tensor_name);
    if (bits_to_levels_.find(config.quantization_bits) ==
        bits_to_levels_.end()) {
      int num_levels;
      bits_to_levels_[config.quantization_bits] =
          FillLevels(config.quantization_bits, num_levels, compression_type_,
                     levels_type_);
    }
  }
  return Compressor::Init(entries);
}

inline size_t CPUNormalizedQuantizer::BufferSize(
    int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg) {
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  size_t compressed_values_buffer_size = (num_elems * bits + 7) / 8;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  size_t meta_buffer_size = get_sizeof(dtype) * num_buckets;
  return compressed_values_buffer_size + meta_buffer_size;
}

size_t CPUNormalizedQuantizer::CompressBuffer(
    unsigned char* input_data, unsigned char* output,
    unsigned char* feedback_data, int num_elems, DataType dtype,
    const CompressionModuleConfig& compression_cfg, void* ctx) {
  assert(dtype == DataType::HOROVOD_FLOAT32);
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  size_t meta_buffer_size = sizeof(float) * num_buckets;
  size_t compressed_values_buffer_size = (num_elems * bits + 7) / 8;

  auto meta_info_buffer = (float*)output;
  output += meta_buffer_size;
  for (int bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    CompressBucket(
        input_data, meta_info_buffer, output, feedback_data, compression_cfg,
        std::min(bucket_size, num_elems - bucket_no * bucket_size), bucket_no);
  }
  return compressed_values_buffer_size + meta_buffer_size;
}

void CPUNormalizedQuantizer::DecompressBuffer(
    unsigned char* input_data, unsigned char* output, int num_elems,
    DataType dtype, bool add, const CompressionModuleConfig& compression_cfg,
    void* ctx) {
  assert(dtype == DataType::HOROVOD_FLOAT32);
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  size_t meta_buffer_size = sizeof(float) * num_buckets;

  auto meta_info_buffer = (float*)input_data;
  input_data += meta_buffer_size;
  for (int bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    DecompressBucket(input_data, meta_info_buffer, output, compression_cfg,
                     std::min(bucket_size, num_elems - bucket_no * bucket_size),
                     bucket_no, add);
  }
}

unsigned char CPUNormalizedQuantizer::EncodeValue(float v, float* feedback,
                                                  float norm, int bits) {
  float d = v;
  unsigned char num_levels = 1 << (bits - 1);
  char sign = (d < -EPS);
  d /= norm;
  if (levels_type_ == LevelsType::Wide) {
    num_levels *= 2;
    sign = 0;
  } else {
    d = fabs(d);
  }
  float* levels = bits_to_levels_[bits];
  float rand = randomizer.GetRand();
  unsigned char level_idx = 0;
  // levels are going 1.0 q_n q_{n-1} ... 0.0(or -1.0)
  while (level_idx + 1 < num_levels) {
    if (d - levels[level_idx + 1] > EPS) {
      if (d + (levels[level_idx] - levels[level_idx + 1]) * rand -
              levels[level_idx] <
          -EPS) {
        level_idx++;
      }
      break;
    }
    level_idx++;
  }

  // update error feedback
  if (feedback) {
    float recovered_v = norm * (sign ? -1.0 : 1.0);
    if (bits > 1)
      recovered_v *= levels[level_idx];
    *feedback = v - recovered_v;
  }
  level_idx |= (sign << (bits - 1));
  return level_idx;
}

float CPUNormalizedQuantizer::DecodeValue(unsigned char input, float norm,
                                          int bits) {
  unsigned int num_levels;
  char sign;
  float* levels = bits_to_levels_[bits];
  if (levels_type_ == LevelsType::Wide and bits > 1) {
    num_levels = 1 << bits;
    sign = 1;
  } else {
    num_levels = 1 << (bits - 1);
    sign = (input & num_levels) ? -1 : 1;
    input &= num_levels - 1;
  }
  float decode_value = norm * sign;

  if (bits > 1) {
    decode_value *= levels[input];
  }
  return decode_value;
}

void CPUNormalizedQuantizer::CompressBucket(
    unsigned char* input_data, float* meta_info_buffer, unsigned char* output,
    unsigned char* feedback_data,
    const CompressionModuleConfig& compression_cfg, int num_elems,
    int bucket_no) {
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  float* input = ((float*)input_data) + bucket_size * bucket_no;
  float* feedback = nullptr;
  if (feedback_data)
    feedback = ((float*)feedback_data) + bucket_no * bucket_size;
  output =
      output + (bucket_size * bucket_no * bits + PACK_SIZE - 1) / PACK_SIZE;

  float norm = 0;
  if (norm_type_ == NormType::Linf) {
    for (int i = 0; i < num_elems; i++) {
      norm = std::max((float)fabs(input[i]), norm);
    }
  } else {
    for (int i = 0; i < num_elems; i++) {
      norm += input[i] * input[i];
    }
    norm = std::sqrt(norm);
  }
  if (norm < EPS)
    norm += EPS;
  meta_info_buffer[bucket_no] = norm;

  std::function<unsigned char(float, float*)> encode =
      std::bind(&CPUNormalizedQuantizer::EncodeValue, this,
                std::placeholders::_1, std::placeholders::_2, norm, bits);
  PackBucket(input, output, feedback, num_elems, bits, encode);
}

void CPUNormalizedQuantizer::DecompressBucket(
    unsigned char* input_data, float* meta_info_buffer,
    unsigned char* output_data, const CompressionModuleConfig& compression_cfg,
    int num_elems, int bucket_no, bool add) {
  const int bits = compression_cfg.quantization_bits;
  const int bucket_size = compression_cfg.bucket_size;
  float* output = (float*)output_data + bucket_no * bucket_size;
  // number of bits in char.
  const float norm = meta_info_buffer[bucket_no];
  input_data =
      input_data + (bucket_size * bucket_no * bits + PACK_SIZE - 1) / PACK_SIZE;
  std::function<float(unsigned char)> decode =
      std::bind(&CPUNormalizedQuantizer::DecodeValue, this,
                std::placeholders::_1, norm, bits);
  UnpackBucket(input_data, output, num_elems, bits, add, decode);
}

// Utils
void PackBucket(float* input, unsigned char* output, float* feedback,
                int num_elems, int bits,
                std::function<unsigned char(float, float*)> encode) {
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  float* feedback_ = nullptr;
  for (int i = 0; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i++) {
    uint64_t value = 0;
    for (unsigned int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems;
         j++) {
      int idx = i * PACK_SIZE + j;
      if (feedback)
        feedback_ = feedback + idx;
      uint64_t encoded = encode(input[idx], feedback_);
      value += (encoded << (j * bits));
    }
    for (int j = 0; j < bits && i * bits + j < num_char; j++) {
      output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;
    }
  }
}

void UnpackBucket(unsigned char* input, float* output, int num_elems, int bits,
                  bool add, std::function<float(unsigned char)> decode) {
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  unsigned int divisor = 1 << bits;
  for (int i = 0; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i++) {
    uint64_t value = 0;
    for (int j = 0; j < bits && i * bits + j < num_char; j++) {
      value |= ((uint64_t)input[i * bits + j]) << (j * PACK_SIZE);
    }
    for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
      unsigned char encoded_value = (value >> (j * bits)) & (divisor - 1);
      if (add)
        output[i * PACK_SIZE + j] += decode(encoded_value);
      else
        output[i * PACK_SIZE + j] = decode(encoded_value);
    }
  }
}

struct CompressConfigParser {
  void ParseYaml(FILE* fh, Compressor::map_compresion_configs& modules_configs,
                 Compressor::set_ignore_modules& ignore_modules) {
    yaml_parser_t parser;
    yaml_event_t event;
    if (!yaml_parser_initialize(&parser)) {
      LOG(WARNING) << "Failed to initialize yaml parser";
      return;
    }
    yaml_parser_set_input_file(&parser, fh);
    do {
      if (!yaml_parser_parse(&parser, &event)) {
        LOG(ERROR) << "Compressor: yaml Parser error " << parser.error;
        exit(1);
      }
      if (event.type == YAML_SCALAR_EVENT) {
        auto value =
            std::string(reinterpret_cast<char*>(event.data.scalar.value));
        if (value == "ignore_modules") {
          ParseIgnoreModules(&parser, &event, ignore_modules);
        } else if (value == "modules") {
          ParseModules(&parser, &event, modules_configs);
        }
      }
    } while (event.type != YAML_STREAM_END_EVENT);
    yaml_event_delete(&event);
    yaml_parser_delete(&parser);
  }

private:
  void ParseIgnoreModules(yaml_parser_t* parser_p, yaml_event_t* event_p,
                          Compressor::set_ignore_modules& modules) {
    do {
      if (!yaml_parser_parse(parser_p, event_p)) {
        LOG(ERROR) << "Compressor: yaml parsing error " << parser_p->error;
        exit(1);
      }
      switch (event_p->type) {
      case YAML_SEQUENCE_START_EVENT:
      case YAML_SEQUENCE_END_EVENT:
        break;
      case YAML_SCALAR_EVENT: {
        std::string value(reinterpret_cast<char*>(event_p->data.scalar.value));
        modules.insert(value);
        break;
      }
      default:
        LOG(ERROR) << "Compressor: yaml parsing unexpected token";
        break;
      }
    } while (event_p->type != YAML_SEQUENCE_END_EVENT);
  }
  void ParseModules(yaml_parser_t* parser_p, yaml_event_t* event_p,
                    Compressor::map_compresion_configs& modules) {
    CompressionModuleConfig config;
    std::string name;
    do {
      if (!yaml_parser_parse(parser_p, event_p)) {
        LOG(ERROR) << "Compressor: yaml parsing error " << parser_p->error;
        exit(1);
      }
      switch (event_p->type) {
      case YAML_SEQUENCE_START_EVENT:
      case YAML_SEQUENCE_END_EVENT:
        break;
      case YAML_MAPPING_START_EVENT:
        config = {};
        name = "";
        break;
      case YAML_MAPPING_END_EVENT:
        assert(name != "");
        modules.emplace(name, config);
        break;
      case YAML_SCALAR_EVENT: {
        std::string value(reinterpret_cast<char*>(event_p->data.scalar.value));
        if (value == "name") {
          ParseString(parser_p, event_p, name);
        } else if (value == "quantization_bits") {
          int value;
          ParseInt(parser_p, event_p, value);
          config.quantization_bits = value;
        } else if (value == "bucket_size") {
          int value;
          ParseInt(parser_p, event_p, value);
          config.bucket_size = value;
        }
        break;
      }
      default:
        LOG(ERROR) << "Compressor: yaml parsing unexpected token";
        break;
      }
    } while (event_p->type != YAML_SEQUENCE_END_EVENT);
  }

  void ParseString(yaml_parser_t* parser_p, yaml_event_t* event_p,
                   std::string& value) {
    if (!yaml_parser_parse(parser_p, event_p)) {
      LOG(ERROR) << "Compressor: yaml Parsing error " << parser_p->error;
      exit(1);
    }
    assert(event_p->type == YAML_SCALAR_EVENT);
    value = std::string(reinterpret_cast<char*>(event_p->data.scalar.value));
  }

  void ParseInt(yaml_parser_t* parser_p, yaml_event_t* event_p, int& value) {
    if (!yaml_parser_parse(parser_p, event_p)) {
      LOG(ERROR) << "Compressor: Parsing Int error " << parser_p->error;
      exit(1);
    }
    assert(event_p->type == YAML_SCALAR_EVENT);
    value = std::atoi(reinterpret_cast<char*>(event_p->data.scalar.value));
  }
};

void Compressor::ParseYaml(const char* file) {
  FILE* fh = fopen(file, "r");
  if (fh == nullptr) {
    LOG(WARNING) << "Compression config file not found";
    return;
  }
  CompressConfigParser parser;
  parser.ParseYaml(fh, modules_configs, ignore_modules);
  fclose(fh);
}

} // namespace common
} // namespace horovod