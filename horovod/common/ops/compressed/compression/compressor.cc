#include "compressor.h"
#include <algorithm>
#include <cstring>
#include <functional>

#include "../../../logging.h"
#include "../../../utils/env_parser.h"
#include "../utils.h"

namespace horovod {
namespace common {

const int PACK_SIZE = 8;
const float EPS = 1e-6;

void PackBucket(float* input, unsigned char* output, float* feedback,
                int num_elems, int bits,
                std::function<unsigned char(float, float*)> encode);
void UnpackBucket(unsigned char* input, float* output, int num_elems, int bits,
                  std::function<float(unsigned char)> decode);

void Compressor::SetQuantizationLevels(float* levels) {
}

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
    ErrorFeedback& error_feedback, int64_t fusion_offset, int64_t global_offset,
    int64_t chunk_num_elems, bool disable_error_feedback) {
  int64_t total_compressed_size = 0;
  if (entries.size() == 1) {
    unsigned char* feedback_data = nullptr;
    if (!disable_error_feedback && error_feedback.isEnabled())
      feedback_data = error_feedback.GetData(entries[0]) +
                      (global_offset + fusion_offset) * sizeof(float);

    total_compressed_size =  Compress(input_data, output, feedback_data, chunk_num_elems);
  } else {
    int64_t offset_cumm = 0;
    int64_t nelem = 0;
    int64_t buffer_offset = 0;
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
      auto offset = buffer_offset * sizeof(float);
      unsigned char* feedback_data = nullptr;
      if (!disable_error_feedback && error_feedback.isEnabled())
        feedback_data = error_feedback.GetData(entry) + offset;
      compressed_size =
          Compress(input_data + offset, feedback_data, output, nelem);
      offset_cumm += entry.tensor->shape().num_elements();
      output += compressed_size;
      total_compressed_size += compressed_size;
    }
  }
//  Finalize();
  return total_compressed_size;
}

void Compressor::Decompress(unsigned char* input_data, unsigned char* output_data,
                            const std::vector<TensorTableEntry>& entries,
                            int64_t fusion_offset, int64_t chunk_num_elems) {
  if (entries.size() == 1) {
    Decompress(input_data, output_data, chunk_num_elems);
  } else {
    int64_t offset_cumm = 0;
    int64_t nelem = 0;
    int64_t buffer_offset = 0;
    int64_t cumm_decompressed = 0;
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
      buffer_offset = std::max(offset_cumm - fusion_offset, 0l);
      output = output_data + buffer_offset * sizeof(float);
      Decompress(input_data + cumm_decompressed, output, nelem);
      cumm_decompressed += BufferSize(nelem);
      offset_cumm += entry.tensor->shape().num_elements();
    }
  }
//  Finalize();
}

int64_t Compressor::Compress(
    unsigned char* output,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    ErrorFeedback& error_feedback, int64_t fusion_offset, int64_t global_offset,
    int64_t chunk_num_elems, bool original, bool disable_error_feedback) {
  auto get_tensor_data =
      [original](const horovod::common::TensorTableEntry& entry) {
        if (original)
          return (unsigned char*)entry.tensor->data();
        else
          return (unsigned char*)entry.output->data();
      };
  int64_t total_compressed_size = 0;

  if (entries.size() == 1) {
    auto offset = (fusion_offset + global_offset) * sizeof(float);
    unsigned char* feedback_data = nullptr;
    if (!disable_error_feedback && error_feedback.isEnabled())
      feedback_data = error_feedback.GetData(entries[0]) + offset;
    total_compressed_size = Compress(get_tensor_data(entries[0]) + offset, output, feedback_data,
                    chunk_num_elems);
  } else {

    int64_t offset_cumm = 0;
    int64_t nelem = 0;
    int64_t buffer_offset = 0;
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
      auto offset = buffer_offset * sizeof(float);
      auto tensor_data = get_tensor_data(entry) + offset;
      unsigned char* feedback_data = nullptr;
      if (!disable_error_feedback && error_feedback.isEnabled())
        feedback_data = error_feedback.GetData(entry) + offset;
      compressed_size = Compress(tensor_data, output, feedback_data, nelem);
      offset_cumm += entry.tensor->shape().num_elements();
      output += compressed_size;
      total_compressed_size += compressed_size;
    }
  }
//  Finalize();
  return total_compressed_size;
}

void Compressor::Decompress(
    unsigned char* input_data,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int64_t fusion_offset, int64_t global_offset, int64_t chunk_num_elems) {
  if (entries.size() == 1) {
    Decompress(input_data,
               ((unsigned char*)entries[0].output->data()) +
                   fusion_offset * sizeof(float) +
                   global_offset * sizeof(float),
               chunk_num_elems);
  } else {
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
                    buffer_offset * sizeof(float);

      Decompress(input_data + cumm_decompressed, output, nelem);
      cumm_decompressed += BufferSize(nelem);
      offset_cumm += entry.tensor->shape().num_elements();
    }
  }
//  Finalize();
}

void Compressor::Finalize() {
}

double Compressor::getMetaInfoTime() const { return meta_info_time_; }

double Compressor::getCompressionTime() const { return compression_time_; }

// ================
// Dummy Compressor
// ================
int64_t CPUDummyCompressor::Compress(unsigned char* input_data,
                                     unsigned char* output,
                                     unsigned char* feedback_data,
                                     int64_t num_elems) {
  int64_t processed_size = num_elems * sizeof(float);
  std::memcpy(output, input_data, processed_size);
  return processed_size;
}

void CPUDummyCompressor::Decompress(unsigned char* input_data,
                                    unsigned char* output, int64_t num_elems) {
  int64_t processed_size = num_elems * sizeof(float);
  std::memcpy(output, input_data, processed_size);
}

// ================
// Quantizers
// ================

Quantizer::Quantizer(horovod::common::HorovodGlobalState* global_state,
                     int quantization_bits)
    : Compressor(global_state), bits_(quantization_bits) {}

// ================
// Max Min Quantizer
// ================
inline int64_t CPUMaxMinQuantizer::BufferSize(int num_elems) {
  int64_t compressed_values_buffer_size =
      (num_elems * bits_ + 7) / 8;
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Max min buffer to allocate
  int64_t meta_buffer_size = 2 * sizeof(float) * num_buckets;
  return compressed_values_buffer_size + meta_buffer_size;
}

int64_t CPUMaxMinQuantizer::Compress(unsigned char* input_data,
                                     unsigned char* output,
                                     unsigned char* feedback_data,
                                     int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = 2 * sizeof(float) * num_buckets;
  int64_t compressed_values_buffer_size =
      (num_elems * bits_ + 7) / 8;

  auto meta_info_buffer = (float*)output;
  output += meta_buffer_size;
  for (int64_t bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    CompressBucket(
        input_data, meta_info_buffer, output, feedback_data,
        std::min((int64_t)bucket_size_, num_elems - bucket_no * bucket_size_),
        bucket_no);
  }
  return compressed_values_buffer_size + meta_buffer_size;
}

void CPUMaxMinQuantizer::Decompress(unsigned char* input_data,
                                    unsigned char* output, int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = 2 * sizeof(float) * num_buckets;

  auto meta_info_buffer = (float*)input_data;
  input_data += meta_buffer_size;
  for (int64_t bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    DecompressBucket(
        input_data, meta_info_buffer, output,
        std::min((int64_t)bucket_size_, num_elems - bucket_no * bucket_size_),
        bucket_no);
  }
}

void CPUMaxMinQuantizer::CompressBucket(unsigned char* input_data,
                                        float* meta_info_buffer,
                                        unsigned char* output,
                                        unsigned char* feedback_data,
                                        int64_t num_elems, int64_t bucket_no) {
  float* input = ((float*)input_data) + bucket_no * bucket_size_;
  float* feedback = nullptr;
  if (feedback_data)
    feedback = ((float*)feedback_data) + bucket_no * bucket_size_;
  // number of bits in char.
  output =
      output + (bucket_size_ * bucket_no * bits_ + PACK_SIZE - 1) / PACK_SIZE;

  float max = input[0];
  float min = input[0];
  auto start = clock_::now();
  for (int i = 1; i < num_elems; i++) {
    max = std::max(max, input[i]);
    min = std::min(min, input[i]);
  }
  meta_info_time_ += time_since(start);

  meta_info_buffer[2 * bucket_no] = max;
  meta_info_buffer[2 * bucket_no + 1] = min;
  int divisor = (1 << bits_) - 1;
  float unit = (max - min) / divisor;
  start = clock_::now();
  std::function<unsigned char(float, float*)> encode =
      std::bind(&CPUMaxMinQuantizer::EncodeValue, this, std::placeholders::_1,
                std::placeholders::_2, min, unit);
  PackBucket(input, output, feedback, num_elems, bits_, encode);
  compression_time_ += time_since(start);
}

unsigned char CPUMaxMinQuantizer::EncodeValue(float v, float* feedback,
                                              float min, float unit) {
  float rand = GetRand();
  float d = ((v - min) / unit) + rand;
  unsigned char level = (unsigned char)floor(d);
//  printf(" unit %f min %f d: %f, rand: %f decoded: %f\n", unit, min, d, rand, (min + level * unit));
  // update error feedback
  if (feedback)
    *feedback = v - (min + level * unit);
  return level;
}

void CPUMaxMinQuantizer::DecompressBucket(unsigned char* input_data,
                                          float* meta_info_buffer,
                                          unsigned char* output_data,
                                          int64_t num_elems,
                                          int64_t bucket_no) {
  float* output = (float*)output_data + bucket_no * bucket_size_;
  // number of bits in char.
  int divisor = (1 << bits_) - 1;
  float max = meta_info_buffer[2 * bucket_no];
  float min = meta_info_buffer[2 * bucket_no + 1];
  float unit = (max - min) / divisor;
  input_data = input_data +
               (bucket_size_ * bucket_no * bits_ + PACK_SIZE - 1) / PACK_SIZE;
  std::function<float(unsigned char)> encode = std::bind(
      &CPUMaxMinQuantizer::DecodeValue, this, std::placeholders::_1, min, unit);
  auto start = clock_::now();

  UnpackBucket(input_data, output, num_elems, bits_, encode);
  compression_time_ += time_since(start);
}

float CPUMaxMinQuantizer::DecodeValue(unsigned char input, float min,
                                      float unit) {
//  printf(" unit %f min %f decoded: %f\n", unit, min, (min + input * unit));
  return min + unit * input;
}

// ================
// Normalized Quantizers
// ================
Status CPUNormalizedQuantizer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  if (levels_ == nullptr) {
    int num_levels = 1 << (bits_ - 1);
    if (multiplier_ < 0.0) {
      // Uniform levels
      levels_ = new float[num_levels];
      levels_[0] = 1.0;
      float value = 1.0 / (num_levels - 1);
      for (int i = 1; i <= num_levels - 1; i++) {
        levels_[i] = (num_levels - i - 1) * value;
      }
    } else {
      // Exponential levels
      // preallocate more levels
      const int coef = 4;
      levels_ = new float[num_levels * coef];
      levels_[0] = 1.0;
      float level_v = multiplier_;
      for (int i = 1; i < coef * num_levels - 1; i++) {
        levels_[i] = level_v;
        level_v *= multiplier_;
      }
    }
  }
  return Status::OK();
}

inline int64_t CPUNormLinfQuantizer::BufferSize(int num_elems) {
  int64_t compressed_values_buffer_size =
  (num_elems * bits_ + 7) / 8;
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size =
      sizeof(float) * num_buckets;
  return compressed_values_buffer_size + meta_buffer_size;
}

int64_t CPUNormLinfQuantizer::Compress(unsigned char* input_data,
                                       unsigned char* output,
                                       unsigned char* feedback_data,
                                       int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = sizeof(float) * num_buckets;
  int64_t compressed_values_buffer_size =
      (num_elems * bits_ + 7) / 8;

  auto meta_info_buffer = (float*)output;
  output += meta_buffer_size;
  for (int64_t bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    CompressBucket(
        input_data, meta_info_buffer, output, feedback_data,
        std::min((int64_t)bucket_size_, num_elems - bucket_no * bucket_size_),
        bucket_no);
  }
  return compressed_values_buffer_size + meta_buffer_size;
}

void CPUNormLinfQuantizer::Decompress(unsigned char* input_data,
                                      unsigned char* output,
                                      int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = sizeof(float) * num_buckets;

  auto meta_info_buffer = (float*)input_data;
  input_data += meta_buffer_size;
  for (int64_t bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    DecompressBucket(
        input_data, meta_info_buffer, output,
        std::min((int64_t)bucket_size_, num_elems - bucket_no * bucket_size_),
        bucket_no);
  }
}

unsigned char CPUNormLinfQuantizer::EncodeValue(float v, float* feedback,
                                                float norm) {
  float d = v;
  char sign = (d < -EPS);
  unsigned char num_levels = 1 << (bits_ - 1);
  d /= norm;
  d = fabs(d);
  float rand = GetRand();
  unsigned char level_idx = 0;
  // levels are going 1.0 q_n q_{n-1} ... 0.0
  while (level_idx + 1 < num_levels) {
    if (d - levels_[level_idx + 1] > EPS) {
      if (d + (levels_[level_idx] - levels_[level_idx + 1]) * rand -
              levels_[level_idx] <
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
    if (bits_ > 1)
      recovered_v *= (level_idx < num_levels - 1) ? levels_[level_idx] : 0.0;
    *feedback = v - recovered_v;
  }
  level_idx |= (sign << (bits_ - 1));
  return level_idx;
}

float CPUNormLinfQuantizer::DecodeValue(unsigned char input, float norm) {
  unsigned int num_levels = 1 << (bits_ - 1);
  char sign = (input & num_levels) ? -1 : 1;
  float decode_value = norm * sign;

  if (bits_ > 1) {
    input &= num_levels - 1;
    decode_value *= (input < num_levels - 1) ? levels_[input] : 0.0;
  }
  return decode_value;
}

void CPUNormLinfQuantizer::CompressBucket(
    unsigned char* input_data, float* meta_info_buffer, unsigned char* output,
    unsigned char* feedback_data, int64_t num_elems, int64_t bucket_no) {
  float* input = ((float*)input_data) + bucket_size_ * bucket_no;
  float* feedback = nullptr;
  if (feedback_data)
    feedback = ((float*)feedback_data) + bucket_no * bucket_size_;
  output =
      output + (bucket_size_ * bucket_no * bits_ + PACK_SIZE - 1) / PACK_SIZE;

  float norm = 0;
  auto start = clock_::now();
  for (int i = 0; i < num_elems; i++) {
    norm = std::max((float)fabs(input[i]), norm);
  }
  if (norm < EPS)
    norm += EPS;
  meta_info_time_ += time_since(start);
  meta_info_buffer[bucket_no] = norm;
  start = clock_::now();

  std::function<unsigned char(float, float*)> encode =
      std::bind(&CPUNormLinfQuantizer::EncodeValue, this, std::placeholders::_1,
                std::placeholders::_2, norm);
  PackBucket(input, output, feedback, num_elems, bits_, encode);
  compression_time_ += time_since(start);
}

void CPUNormLinfQuantizer::DecompressBucket(unsigned char* input_data,
                                            float* meta_info_buffer,
                                            unsigned char* output_data,
                                            int64_t num_elems,
                                            int64_t bucket_no) {
  float* output = (float*)output_data + bucket_no * bucket_size_;
  // number of bits in char.
  float norm = meta_info_buffer[bucket_no];
  input_data = input_data +
               (bucket_size_ * bucket_no * bits_ + PACK_SIZE - 1) / PACK_SIZE;
  auto start = clock_::now();
  std::function<float(unsigned char)> decode = std::bind(
      &CPUNormLinfQuantizer::DecodeValue, this, std::placeholders::_1, norm);
  UnpackBucket(input_data, output, num_elems, bits_, decode);
  compression_time_ += time_since(start);
}

inline int64_t CPUNormL2Quantizer::BufferSize(int num_elems) {
  int64_t compressed_values_buffer_size =
      (num_elems * bits_ + 7) / 8;
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  // Meta data buffer to allocate max_log + norm
  int64_t meta_buffer_size =
      (sizeof(char) + sizeof(float)) * num_buckets;
  return compressed_values_buffer_size + meta_buffer_size;
}

int64_t CPUNormL2Quantizer::Compress(unsigned char* input_data,
                                     unsigned char* output,
                                     unsigned char* feedback_data,
                                     int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = (sizeof(float) + sizeof(char)) * num_buckets;
  int64_t compressed_values_buffer_size =
      (num_elems * bits_ + 7) / 8;

  auto meta_info_buffer = (float*)output;
  auto max_log_buffer = (unsigned char*)(meta_info_buffer + num_buckets);
  output += meta_buffer_size;
  for (int64_t bucket_no = 0; bucket_no < num_buckets; bucket_no++) {
    CompressBucket(
        input_data, meta_info_buffer, max_log_buffer, output, feedback_data,
        std::min((int64_t)bucket_size_, num_elems - bucket_no * bucket_size_),
        bucket_no);
  }
  return compressed_values_buffer_size + meta_buffer_size;
}

void CPUNormL2Quantizer::Decompress(unsigned char* input_data,
                                    unsigned char* output, int64_t num_elems) {
  int64_t num_buckets = (num_elems + bucket_size_ - 1) / bucket_size_;
  int64_t meta_buffer_size = (sizeof(float) + sizeof(char)) * num_buckets;

  auto meta_info_buffer = (float*)input_data;
  auto max_log_buffer = (unsigned char*)(meta_info_buffer + num_buckets);

  input_data += meta_buffer_size;
  for (int64_t bucket_no = 0; bucket_no < num_buckets; bucket_no++) {

    DecompressBucket(
        input_data, meta_info_buffer, max_log_buffer, output,
        std::min((int64_t)bucket_size_, num_elems - bucket_no * bucket_size_),
        bucket_no);
  }
}

void CPUNormL2Quantizer::CompressBucket(unsigned char* input_data,
                                        float* norm_buffer,
                                        unsigned char* max_log_buffer,
                                        unsigned char* output,
                                        unsigned char* feedback_data,
                                        int64_t num_elems, int64_t bucket_no) {
  float* input = ((float*)input_data) + bucket_size_ * bucket_no;
  float* feedback = nullptr;
  if (feedback_data)
    feedback = ((float*)feedback_data) + bucket_no * bucket_size_;
  output =
      output + (bucket_size_ * bucket_no * bits_ + PACK_SIZE - 1) / PACK_SIZE;

  float norm = 0;
  float max = 0;
  auto start = clock_::now();
  for (int i = 0; i < num_elems; i++) {
    norm += input[i] * input[i];
    max = std::max(max, (float)fabs(input[i]));
  }
  norm = std::sqrt(norm);
  if (norm < EPS)
    norm += EPS;
  max /= norm;
  unsigned char max_log = (unsigned char)(floor(-log2(max)));
//  float m = 1.0;
//  // find log of max so that we could offset search area of the
//  // matching quantization point
//  while (max > EPS && (max - m * multiplier_) < -EPS) {
//    m *= multiplier_;
//    max_log++;
//  }
  meta_info_time_ += time_since(start);
  norm_buffer[bucket_no] = norm;
  max_log_buffer[bucket_no] = max_log;

  start = clock_::now();

  std::function<unsigned char(float, float*)> encode =
      std::bind(&CPUNormL2Quantizer::EncodeValue, this, std::placeholders::_1,
                std::placeholders::_2, norm, max_log);
  PackBucket(input, output, feedback, num_elems, bits_, encode);
  compression_time_ += time_since(start);
}

void CPUNormL2Quantizer::DecompressBucket(unsigned char* input_data,
                                          float* norm_info_buffer,
                                          unsigned char* max_log_buffer,
                                          unsigned char* output_data,
                                          int64_t num_elems,
                                          int64_t bucket_no) {
  float* output = (float*)output_data + bucket_no * bucket_size_;
  // number of bits in char.
  float norm = norm_info_buffer[bucket_no];
  unsigned char max_log = max_log_buffer[bucket_no];
  input_data = input_data +
               (bucket_size_ * bucket_no * bits_ + PACK_SIZE - 1) / PACK_SIZE;
  auto start = clock_::now();
  std::function<float(unsigned char)> decode =
      std::bind(&CPUNormL2Quantizer::DecodeValue, this, std::placeholders::_1,
                norm, max_log);
  UnpackBucket(input_data, output, num_elems, bits_, decode);
  compression_time_ += time_since(start);
}

unsigned char CPUNormL2Quantizer::EncodeValue(float v, float* feedback,
                                              float norm,
                                              unsigned char max_log) {
  int num_levels = 1 << (bits_ - 1);
  char sign = (v < -EPS);
  float d = fabs(v / norm);
  unsigned char level_idx;
  if (d < EPS) {
    level_idx = num_levels - 1;
  } else {
    float level_f = -log2(d);
    int c = (int)(ceil(level_f));
    int f = (int)(floor(level_f));
    level_idx = f - max_log;
    if (level_idx < num_levels - 1 && d + (levels_[f] - levels_[c]) * GetRand() - levels_[f] < -EPS) {
      level_idx++;
    }
    level_idx = std::min(level_idx, (unsigned char)(num_levels - 1));
  }

  if (feedback) {
    float decode_value = norm * (sign ? -1.0 : 1.0);
    if (bits_ > 1)
      decode_value *=
          (level_idx < num_levels - 1) ? levels_[max_log + level_idx] : 0.0;
    *feedback = v - decode_value;
  }
  level_idx |= (sign << (bits_ - 1));
  return level_idx;
}

float CPUNormL2Quantizer::DecodeValue(unsigned char input, float norm,
                                      int max_log) {
  unsigned int num_levels = 1 << (bits_ - 1);
  char sign = (input & num_levels) ? -1 : 1;
  float decode_value = norm * sign;

  if (bits_ > 1) {
    input &= num_levels - 1;
    decode_value *= (input < num_levels - 1) ? levels_[max_log + input] : 0.0;
  }
  return decode_value;
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
//      printf("Encode idx: %i", idx);
      uint64_t encoded = encode(input[idx], feedback_);
      value += (encoded << (j * bits));
    }
    for (int j = 0; j < bits && i * bits + j < num_char; j++) {
      output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;
    }
  }
}

void UnpackBucket(unsigned char* input, float* output, int num_elems, int bits,
                  std::function<float(unsigned char)> decode) {
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  unsigned int divisor = 1 << bits;
  for (int i = 0; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i++) {
    uint64_t value = 0;
    for (int j = 0; j < bits && i * bits + j < num_char; j++) {
      value |= ((uint64_t)input[i * bits + j]) << (j * PACK_SIZE);
    }
    for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
      unsigned char encoded_value = (value >> (j * bits)) & (divisor - 1);
//      printf("Decode idx: %i", i * PACK_SIZE + j);
      output[i * PACK_SIZE + j] = decode(encoded_value);
    }
  }
}

} // namespace common
} // namespace horovod