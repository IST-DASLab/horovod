#ifndef HOROVOD_COMPRESSOR_H
#define HOROVOD_COMPRESSOR_H

#include "../../../common.h"
#include "error_feedback.h"

namespace horovod {
namespace common {
const int COMPRESSION_BUCKET_SIZE = 512;

class Compressor {
public:
  Compressor(HorovodGlobalState* global_state);
  // Returns size of buffer to allocate for usage in compress (in bytes). We
  // assume that no compression will be done in-place.
  virtual ~Compressor() = default;
  virtual int64_t BufferSize(int num_elems, DataType dtype) = 0;
  virtual int64_t BufferSize(int num_elems,
                                const std::vector<TensorTableEntry>& entries,
                             int64_t fusion_offset, int64_t global_offset);
  // Returns size of compressed size (in bytes). And update error_feedback.
  // If error_feedback is nullptr, it's not updated.
  virtual int64_t Compress(unsigned char* input_data, unsigned char* output,
                           unsigned char* feedback_data, int64_t num_elems,
                           DataType dtype, void* ctx) = 0;
  // Decompress data from input to output.
  // If add is True sum decompressed data with output.
  virtual void Decompress(unsigned char* input, unsigned char* output,
                          int64_t num_elems, DataType dtype, bool add, void* ctx) = 0;
  // Compresses input_data into output per entry. Returns size of compressed
  // data.
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   const std::vector<TensorTableEntry>& entries,
                   ErrorFeedback& error_feedback, int64_t fusion_offset,
                   int64_t global_offset, int64_t chunk_num_elems,
                   bool disable_error_feedback, void* ctx);
  // Decompresses input_data into output.
  void Decompress(unsigned char* input_data, unsigned char* output,
                  const std::vector<TensorTableEntry>& entries,
                  int64_t fusion_offset, int64_t chunk_num_elems, bool add,
                  void* ctx);
  // Compresses entries data into output. Returns size of compressed data.
  // @original parameter stands for where take the values from entry: original
  // tensor or output.
  int64_t Compress(unsigned char* output,
                   const std::vector<TensorTableEntry>& entries,
                   ErrorFeedback& error_feedback, int64_t fusion_offset,
                   int64_t global_offset, int64_t chunk_num_elems,
                   bool original, bool disable_error_feedback,
                   void* ctx);
  // Decompresses input_data into entries.
  void Decompress(unsigned char* input_data,
                  const std::vector<TensorTableEntry>& entries,
                  int64_t fusion_offset, int64_t global_offset,
                  int64_t chunk_num_elems, bool add, void* ctx);

  virtual void Finalize();
  virtual Status Init(const std::vector<TensorTableEntry>& entries) = 0;
  virtual void SetQuantizationLevels(float* levels);
protected:
  // The size of the bucket.
  int bucket_size_;
  HorovodGlobalState* global_state_;
};

class DummyCompressor : public Compressor {
public:
  DummyCompressor(horovod::common::HorovodGlobalState* global_state)
      : Compressor(global_state) {}

  int64_t BufferSize(int num_elems, DataType dtype) final;

  Status Init(const std::vector<TensorTableEntry>& entries) override {
    return Status::OK();
  }
};

class CPUDummyCompressor : public DummyCompressor {
public:
  CPUDummyCompressor(HorovodGlobalState* global_state)
      : DummyCompressor(global_state) {}

  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   unsigned char* feedback_data, int64_t num_elems, DataType dtype, void* ctx) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems, DataType dtype, bool add, void* ctx) override;
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

class CPUMaxMinQuantizer : public MaxMinQuantizer {
public:
  CPUMaxMinQuantizer(HorovodGlobalState* global_state, int quantization_bits)
      : MaxMinQuantizer(global_state, quantization_bits) {}
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   unsigned char* feedback_data, int64_t num_elems, DataType dtype, void* ctx) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems, DataType dtype, bool add, void* ctx) override;
  void CompressBucket(unsigned char* input_data, float* meta_info_buffer,
                      unsigned char* output, unsigned char* feedback_data,
                      int64_t num_elems, int64_t bucket_no);
  void DecompressBucket(unsigned char* input_data, float* meta_info_buffer,
                        unsigned char* output, int64_t num_elems,
                        int64_t bucket_no, bool add);
  unsigned char EncodeValue(float v, float* feedback, float min, float unit);
  float DecodeValue(unsigned char input, float max, float min);
  int64_t BufferSize(int num_elems, DataType dtype);

private:
  CPURandomizer randomizer;
};

class NormalizedQuantizer : public Quantizer {
public:
  NormalizedQuantizer(horovod::common::HorovodGlobalState* global_state,
                      int quantization_bits, CompressionType compression_type, NormType norm_type, LevelsType levels_type)
      : Quantizer(global_state, quantization_bits), compression_type_(compression_type), norm_type_(norm_type), levels_type_(levels_type){
  }
protected:
  // Buffer to store static levels. Won't be sent.
  float* levels_ = nullptr;
  CompressionType compression_type_;
  NormType norm_type_;
  LevelsType levels_type_;
  CPURandomizer randomizer;
};

class CPUNormalizedQuantizer : public NormalizedQuantizer {
public:
  CPUNormalizedQuantizer(horovod::common::HorovodGlobalState* global_state,
                         int quantization_bits, CompressionType compression_type, NormType norm_type, LevelsType levels_type)
      : NormalizedQuantizer(global_state, quantization_bits, compression_type, norm_type, levels_type) {}
  Status
  Init(const std::vector<horovod::common::TensorTableEntry>& entries) override;
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   unsigned char* feedback_data, int64_t num_elems, DataType dtype, void* ctx) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems, DataType dtype, bool add, void* ctx) override;

  unsigned char EncodeValue(float input, float* feedback, float norm);

  float DecodeValue(unsigned char input, float norm);

  void CompressBucket(unsigned char* input_data, float* meta_info_buffer,
                      unsigned char* output, unsigned char* feedback_data,
                      int64_t num_elems, int64_t bucket_no);
  void DecompressBucket(unsigned char* input_data, float* meta_info_buffer,
                        unsigned char* output, int64_t num_elems,
                        int64_t bucket_no, bool add);
  int64_t BufferSize(int num_elems, DataType dtype);
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_COMPRESSOR_H
