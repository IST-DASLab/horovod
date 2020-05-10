#ifndef HOROVOD_COMPRESSOR_H
#define HOROVOD_COMPRESSOR_H

#include "../../../common.h"
#include "error_feedback.h"

namespace horovod {
namespace common {
const int COMPRESSION_BUCKET_SIZE = 512;
const int ALIGNMENT_UNIT = 2 * sizeof(float);

class Compressor {
public:
  Compressor(HorovodGlobalState* global_state);
  // Returns size of buffer to allocate for usage in compress (in bytes). We
  // assume that no compression will be done in-place.
  virtual ~Compressor() = default;
  virtual int64_t BufferSize(int num_elems) = 0;
  virtual int64_t BufferSize(int num_elems,
                             const std::vector<TensorTableEntry>& entries,
                             int64_t fusion_offset, int64_t global_offset);
  // Returns size of compressed size (in bytes). And update error_feedback.
  // If error_feedback is nullptr, it's not updated.
  virtual int64_t Compress(unsigned char* input_data, unsigned char* output,
                           unsigned char* feedback_data, int64_t num_elems) = 0;
  virtual void Decompress(unsigned char* input, unsigned char* output,
                          int64_t num_elems) = 0;
  // Compresses input_data into output per entry. Returns size of compressed
  // data.
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   const std::vector<TensorTableEntry>& entries,
                   ErrorFeedback& error_feedback, int64_t fusion_offset,
                   int64_t global_offset, int64_t chunk_num_elems,
                   bool disable_error_feedback=false);
  // Decompresses input_data into output.
  void Decompress(unsigned char* input_data, unsigned char* output,
                  const std::vector<TensorTableEntry>& entries,
                  int64_t fusion_offset, int64_t chunk_num_elems);
  // Compresses entries data into output. Returns size of compressed data.
  // @original parameter stands for where take the values from entry: original
  // tensor or output.
  int64_t Compress(unsigned char* output,
                   const std::vector<TensorTableEntry>& entries,
                   ErrorFeedback& error_feedback, int64_t fusion_offset,
                   int64_t global_offset, int64_t chunk_num_elems,
                   bool original = true, bool disable_error_feedback=false);
  // Decompresses input_data into entries.
  void Decompress(unsigned char* input_data,
                  const std::vector<TensorTableEntry>& entries,
                  int64_t fusion_offset, int64_t global_offset,
                  int64_t chunk_num_elems);

  virtual void Finalize();
  virtual Status Init(const std::vector<TensorTableEntry>& entries) = 0;
  double getMetaInfoTime() const;
  double getCompressionTime() const;
  void SetQuantizationLevels(float* levels);
protected:
  // The size of the bucket.
  int bucket_size_;
  HorovodGlobalState* global_state_;
  double meta_info_time_;
  double compression_time_;
};

class DummyCompressor : public Compressor {
public:
  DummyCompressor(horovod::common::HorovodGlobalState* global_state)
      : Compressor(global_state) {}

  int64_t BufferSize(int num_elems) final { return num_elems * sizeof(float); }

  Status Init(const std::vector<TensorTableEntry>& entries) override {
    return Status::OK();
  }
};

class CPUDummyCompressor : public DummyCompressor {
public:
  CPUDummyCompressor(HorovodGlobalState* global_state)
      : DummyCompressor(global_state) {}

  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   unsigned char* feedback_data, int64_t num_elems) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems) override;
};

class CPURandomizer {
public:
  CPURandomizer() {
    std::random_device rd;
    gen_ = std::mt19937(rd());
    rand_ = std::uniform_real_distribution<float>(0.0, 1.0);
  }
  float GetRand() { return rand_(gen_); }

private:
  std::uniform_real_distribution<float> rand_;
  std::mt19937 gen_;
};

class Quantizer : public Compressor {
public:
  Quantizer(HorovodGlobalState* global_state, int quantization_bits);

  Status Init(const std::vector<TensorTableEntry>& entries) override {
    return Status::OK();
  }

protected:
  int bits_;
};

class MaxMinQuantizer : public Quantizer {
public:
  MaxMinQuantizer(HorovodGlobalState* global_state, int quantization_bits)
      : Quantizer(global_state, quantization_bits) {}
};

class CPUMaxMinQuantizer : public MaxMinQuantizer, public CPURandomizer {
public:
  CPUMaxMinQuantizer(HorovodGlobalState* global_state, int quantization_bits)
      : MaxMinQuantizer(global_state, quantization_bits), CPURandomizer() {}
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   unsigned char* feedback_data, int64_t num_elems) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems) override;
  void CompressBucket(unsigned char* input_data, float* meta_info_buffer,
                      unsigned char* output, unsigned char* feedback_data,
                      int64_t num_elems, int64_t bucket_no);
  void DecompressBucket(unsigned char* input_data, float* meta_info_buffer,
                        unsigned char* output, int64_t num_elems,
                        int64_t bucket_no);
  unsigned char EncodeValue(float v, float* feedback, float min, float unit);
  float DecodeValue(unsigned char input, float max, float min);
  int64_t BufferSize(int num_elems);
};

class NormalizedQuantizer : public Quantizer {
public:
  NormalizedQuantizer(horovod::common::HorovodGlobalState* global_state,
                      int quantization_bits, float multiplier)
      : Quantizer(global_state, quantization_bits), multiplier_(multiplier) {}

protected:
  // Buffer to store static levels. Won't be sent.
  float* levels_ = nullptr;
  // multiplier used in case of Exponential quantization
  float multiplier_;
};

class CPUNormalizedQuantizer : public NormalizedQuantizer {
public:
  CPUNormalizedQuantizer(horovod::common::HorovodGlobalState* global_state,
                         int quantization_bits, float multiplier)
      : NormalizedQuantizer(global_state, quantization_bits, multiplier) {}
  Status
  Init(const std::vector<horovod::common::TensorTableEntry>& entries) override;
};

class CPUNormLinfQuantizer : public CPUNormalizedQuantizer,
                             public CPURandomizer {
public:
  CPUNormLinfQuantizer(horovod::common::HorovodGlobalState* global_state,
                       int quantization_bits, float multiplier)
      : CPUNormalizedQuantizer(global_state, quantization_bits, multiplier),
        CPURandomizer() {}
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   unsigned char* feedback_data, int64_t num_elems) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems) override;

  unsigned char EncodeValue(float input, float* feedback, float norm);

  float DecodeValue(unsigned char input, float norm);

  void CompressBucket(unsigned char* input_data, float* meta_info_buffer,
                      unsigned char* output, unsigned char* feedback_data,
                      int64_t num_elems, int64_t bucket_no);
  void DecompressBucket(unsigned char* input_data, float* meta_info_buffer,
                        unsigned char* output, int64_t num_elems,
                        int64_t bucket_no);
  int64_t BufferSize(int num_elems);
};

class CPUNormL2Quantizer : public CPUNormalizedQuantizer,
                           public CPURandomizer {
public:
  CPUNormL2Quantizer(horovod::common::HorovodGlobalState* global_state,
                     int quantization_bits, float multiplier)
      : CPUNormalizedQuantizer(global_state, quantization_bits, multiplier),
        CPURandomizer() {
    if (multiplier != 0.5)
      throw std::logic_error("CPUNormL2Quantizer: Multipliers other than 0.5 are not supported yet");
  }
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   unsigned char* feedback_data, int64_t num_elems) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems) override;

  void CompressBucket(unsigned char* input_data, float* norm_info_buffer,
                      unsigned char* max_log_buffer, unsigned char* output,
                      unsigned char* feedback_data, int64_t num_elems,
                      int64_t bucket_no);
  void DecompressBucket(unsigned char* input_data, float* norm_info_buffer,
                        unsigned char* max_log_buffer, unsigned char* output,
                        int64_t num_elems, int64_t bucket_no);

  unsigned char EncodeValue(float v, float* feedback, float norm,
                            unsigned char max_log);

  float DecodeValue(unsigned char input, float norm, int max_log);
  int64_t BufferSize(int num_elems) final;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_COMPRESSOR_H
