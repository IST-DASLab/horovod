#ifndef HOROVOD_COMPRESSOR_H
#define HOROVOD_COMPRESSOR_H

#include "../../../common.h"
#include "error_feedback.h"
#include <map>
#include <vector>

namespace horovod {
namespace common {
const int COMPRESSION_BUCKET_SIZE = 512;

struct CompressionModuleConfig {
  int quantization_bits;
  int bucket_size;
  bool skip_incomplete_buckets;
};

class Compressor {
public:
  Compressor(HorovodGlobalState* global_state, Summator* summator);
  // Returns size of buffer to allocate for usage in compress (in bytes). We
  // assume that no compression will be done in-place.
  virtual ~Compressor() = default;
  virtual size_t BufferSize(int num_elems, DataType dtype,
                            const CompressionModuleConfig& compression_cfg) = 0;
  size_t BufferSize(int num_elems, const std::vector<TensorTableEntry>& entries,
                    int fusion_offset);

  size_t Compress(unsigned char* input_data, unsigned char* output,
                  const std::vector<TensorTableEntry>& entries,
                  int fusion_offset, int chunk_num_elems,
                  bool disable_error_feedback, void* ctx);
  void Decompress(unsigned char* input_data, unsigned char* output,
                  const std::vector<TensorTableEntry>& entries,
                  int fusion_offset, int chunk_num_elems, bool add, void* ctx);

  // Returns size of compressed size (in bytes). And update error_feedback.
  // If error_feedback is nullptr, it's not updated.
  virtual size_t
  CompressBuffer(unsigned char* input_data, unsigned char* output,
                 unsigned char* feedback_data, int num_elems, DataType dtype,
                 const CompressionModuleConfig& compression_cfg, void* ctx) = 0;
  // Decompress data from input to output.
  // If add is True sum decompressed data with output.
  virtual void DecompressBuffer(unsigned char* input, unsigned char* output,
                                int num_elems, DataType dtype, bool add,
                                const CompressionModuleConfig& compression_cfg,
                                void* ctx) = 0;
  virtual Status Init(const std::vector<TensorTableEntry>& entries);
  virtual void SetQuantizationLevels(float* levels, int bits);
  using map_compresion_configs =
      std::unordered_map<std::string, CompressionModuleConfig>;
  using set_ignore_modules = std::set<std::string>;
  map_compresion_configs& GetModulesConfig() { return modules_configs; }
  set_ignore_modules& GetIgnoreModules() { return ignore_modules; }
  virtual void GetSizesAndOffsets(int num_elements, int world_size,
                                  const std::vector<TensorTableEntry>& entries,
                                  std::vector<int>& offsets,
                                  std::vector<int>& sizes);
  virtual size_t GetRequiredFreeSize();
  bool isInitialized() const { return initialized_; }
  void ApplyErrorFeedback(std::vector<TensorTableEntry>& entries);
  CompressionMode GetCompressionMode() const { return compression_mode_; }

protected:
  const int MIN_SIZE_TO_COMPRESS = 16;
  // The size of the bucket.
  //  int bucket_size_;
  HorovodGlobalState* global_state_;
  CompressionModuleConfig default_config;
  CompressionModuleConfig& GetModuleConfig(const std::string& name);
  bool initialized_;
  ErrorFeedback error_feedback_;
  CompressionMode compression_mode_;

private:
  // Compresses entries data into output. Returns size of compressed data.
  // @original parameter stands for where take the values from entry: original
  // tensor or output.
  size_t CompressFromEntries(unsigned char* output,
                             const std::vector<TensorTableEntry>& entries,
                             int fusion_offset, int chunk_num_elems,
                             bool disable_error_feedback,
                             void* ctx);
  // Decompresses input_data into entries.
  void DecompressIntoEntries(unsigned char* input_data,
                             const std::vector<TensorTableEntry>& entries,
                             int fusion_offset, int chunk_num_elems, bool add,
                             void* ctx);

  // Compresses input_data into output per entry. Returns size of compressed
  // data.
  size_t CompressPerEntry(unsigned char* input_data, unsigned char* output,
                          const std::vector<TensorTableEntry>& entries,
                          int fusion_offset, int chunk_num_elems,
                          bool disable_error_feedback, void* ctx);
  // Decompresses input_data into output.
  void DecompressPerEntry(unsigned char* input_data, unsigned char* output,
                          const std::vector<TensorTableEntry>& entries,
                          int fusion_offset, int chunk_num_elems, bool add,
                          void* ctx);

  void ParseYaml(const char* file);
  map_compresion_configs modules_configs;
  set_ignore_modules ignore_modules;
  unsigned char* fused_feedback_buf = nullptr;
};

class DummyCompressor : public Compressor {
public:
  DummyCompressor(horovod::common::HorovodGlobalState* global_state,
                  Summator* summator)
      : Compressor(global_state, summator) {}

  size_t BufferSize(int num_elems, DataType dtype,
                    const CompressionModuleConfig& compression_cfg) final;
};

class CPUDummyCompressor : public DummyCompressor {
public:
  CPUDummyCompressor(HorovodGlobalState* global_state, Summator* summator)
      : DummyCompressor(global_state, summator) {}

  size_t CompressBuffer(unsigned char* input_data, unsigned char* output,
                        unsigned char* feedback_data, int num_elems,
                        DataType dtype,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  void DecompressBuffer(unsigned char* input_data, unsigned char* output,
                        int num_elems, DataType dtype, bool add,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
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
  Quantizer(HorovodGlobalState* global_state, Summator* summator,
            int quantization_bits);

  virtual void GetSizesAndOffsets(int num_elements, int world_size,
                                  const std::vector<TensorTableEntry>& entries,
                                  std::vector<int>& offsets,
                                  std::vector<int>& sizes);
};

class MaxMinQuantizer : public Quantizer {
public:
  MaxMinQuantizer(HorovodGlobalState* global_state, Summator* summator,
                  int quantization_bits)
      : Quantizer(global_state, summator, quantization_bits) {}
};

class CPUMaxMinQuantizer : public MaxMinQuantizer {
public:
  CPUMaxMinQuantizer(HorovodGlobalState* global_state, Summator* summator,
                     int quantization_bits)
      : MaxMinQuantizer(global_state, summator, quantization_bits) {}
  size_t CompressBuffer(unsigned char* input_data, unsigned char* output,
                        unsigned char* feedback_data, int num_elems,
                        DataType dtype,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  void DecompressBuffer(unsigned char* input_data, unsigned char* output,
                        int num_elems, DataType dtype, bool add,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  void CompressBucket(unsigned char* input_data, float* meta_info_buffer,
                      unsigned char* output, unsigned char* feedback_data,
                      const CompressionModuleConfig& compression_cfg,
                      int num_elems, int bucket_no);
  void DecompressBucket(unsigned char* input_data, float* meta_info_buffer,
                        unsigned char* output,
                        const CompressionModuleConfig& compression_cfg,
                        int num_elems, int bucket_no, bool add);
  unsigned char EncodeValue(float v, float* feedback, float min, float unit);
  float DecodeValue(unsigned char input, float max, float min);
  size_t BufferSize(int num_elems, DataType dtype,
                    const CompressionModuleConfig& compression_cfg);

private:
  CPURandomizer randomizer;
};

class NormalizedQuantizer : public Quantizer {
public:
  NormalizedQuantizer(horovod::common::HorovodGlobalState* global_state,
                      Summator* summator, int quantization_bits,
                      CompressionType compression_type, NormType norm_type,
                      LevelsType levels_type)
      : Quantizer(global_state, summator, quantization_bits),
        compression_type_(compression_type), norm_type_(norm_type),
        levels_type_(levels_type) {}

protected:
  // Buffer to store static levels. Won't be sent.
  // TODO: do it for each bits type.
  std::map<int, float*> bits_to_levels_;
  CompressionType compression_type_;
  NormType norm_type_;
  LevelsType levels_type_;
  CPURandomizer randomizer;
};

class CPUNormalizedQuantizer : public NormalizedQuantizer {
public:
  CPUNormalizedQuantizer(horovod::common::HorovodGlobalState* global_state,
                         Summator* summator, int quantization_bits,
                         CompressionType compression_type, NormType norm_type,
                         LevelsType levels_type)
      : NormalizedQuantizer(global_state, summator, quantization_bits,
                            compression_type, norm_type, levels_type) {}
  Status
  Init(const std::vector<horovod::common::TensorTableEntry>& entries) override;
  size_t CompressBuffer(unsigned char* input_data, unsigned char* output,
                        unsigned char* feedback_data, int num_elems,
                        DataType dtype,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  void DecompressBuffer(unsigned char* input_data, unsigned char* output,
                        int num_elems, DataType dtype, bool add,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;

  unsigned char EncodeValue(float input, float* feedback, float norm, int bits);

  float DecodeValue(unsigned char input, float norm, int bits);

  void CompressBucket(unsigned char* input_data, float* meta_info_buffer,
                      unsigned char* output, unsigned char* feedback_data,
                      const CompressionModuleConfig& compression_cfg,
                      int num_elems, int bucket_no);
  void DecompressBucket(unsigned char* input_data, float* meta_info_buffer,
                        unsigned char* output,
                        const CompressionModuleConfig& compression_cfg,
                        int num_elems, int bucket_no, bool add);
  size_t BufferSize(int num_elems, DataType dtype,
                    const CompressionModuleConfig& compression_cfg);
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_COMPRESSOR_H
