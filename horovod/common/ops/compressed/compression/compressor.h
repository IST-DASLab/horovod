#ifndef HOROVOD_COMPRESSOR_H
#define HOROVOD_COMPRESSOR_H

#include "../../gpu_operations.h"
#include "cuda/cuda_functions.h"

namespace horovod {
namespace common {
const int COMPRESSION_BUCKET_SIZE = 512;
const int ALIGNMENT_UNIT = 2 * sizeof(float);

class Compressor {
public:
  Compressor(HorovodGlobalState* global_state);
  // Returns size of buffer to allocate for usage in compress (in bytes). We
  // assume that no compression will be done in-place.
  virtual int64_t BufferSize(int num_elems) = 0;
  virtual int64_t BufferSize(int num_elems,
                             const std::vector<TensorTableEntry>& entries,
                             int64_t fusion_offset, int64_t global_offset);
  // Returns size of compressed size (in bytes).
  virtual int64_t Compress(unsigned char* input_data, unsigned char* output,
                           int64_t num_elems) = 0;
  virtual void Decompress(unsigned char* input, unsigned char* output,
                          int64_t num_elems) = 0;
  // Compresses input_data into output per entry. Returns size of compressed
  // data.
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   const std::vector<TensorTableEntry>& entries,
                   int64_t fusion_offset, int64_t global_offset,
                   int64_t chunk_num_elems);
  // Decompresses input_data into output.
  void Decompress(unsigned char* input_data, unsigned char* output,
                  const std::vector<TensorTableEntry>& entries,
                  int64_t fusion_offset, int64_t global_offset,
                  int64_t chunk_num_elems);
  // Compresses entries data into output. Returns size of compressed data.
  // @original parameter stands for where take the values from entry: original tensor
  // or output.
  int64_t Compress(unsigned char* output,
                   const std::vector<TensorTableEntry>& entries,
                   int64_t fusion_offset, int64_t global_offset,
                   int64_t chunk_num_elems, bool original=true);
  // Decompresses input_data into entries.
  void Decompress(unsigned char* input_data,
                  const std::vector<TensorTableEntry>& entries,
                  int64_t fusion_offset, int64_t global_offset,
                  int64_t chunk_num_elems);
  virtual Status Init(const std::vector<TensorTableEntry>& entries) = 0;
  double getMetaInfoTime() const;
  double getCompressionTime() const;

protected:
  // The size of the bucket.
  int bucket_size_;
  HorovodGlobalState* global_state_;
  double meta_info_time_;
  double compression_time_;
};

class CPUDummyCompressor : public Compressor {
public:
  CPUDummyCompressor(HorovodGlobalState* global_state);

  int64_t BufferSize(int num_elems) override {
    return num_elems * sizeof(float);
  }

  Status Init(const std::vector<TensorTableEntry>& entries) override {
    return Status::OK();
  }

  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems) override;
};

class GPUCompressor {
public:
  GPUCompressor(GPUContext* gpu_context) : gpu_context_(gpu_context) {}
  // Returns size of buffer to pass in compress (in bytes)
protected:
  GPUContext* gpu_context_;
};

class GPUDummyCompressor : public GPUCompressor, public CPUDummyCompressor {
public:
  GPUDummyCompressor(GPUContext* gpuContext, HorovodGlobalState* global_state);

  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input_data, unsigned char* output,
                  int64_t num_elems) override;
};

class Quantizer : public Compressor {
public:
  Quantizer(HorovodGlobalState* global_state, int quantization_bits);

protected:
  int bits_;
};

class GPUQuantizer : public GPUCompressor, public Quantizer {
public:
  GPUQuantizer(GPUContext* gpu_context, HorovodGlobalState* global_state,
               int quantization_bits)
      : GPUCompressor(gpu_context), Quantizer(global_state, quantization_bits) {
  }

  Status Init(const std::vector<TensorTableEntry>& entries) override;

protected:
  CurandState* cuda_states_ = nullptr;
  int device_;
  // Number of bits used per value.
  FusionBufferManager bufferManager_;
};

class GPUMaxMinQuantizer : public GPUQuantizer {
public:
  GPUMaxMinQuantizer(GPUContext* gpu_context, HorovodGlobalState* global_state,
                     int quantization_bits);
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems) override;
  int64_t BufferSize(int num_elems) override;
};

class GPUNormalizedQuantizer : public GPUQuantizer {
public:
  GPUNormalizedQuantizer(horovod::common::GPUContext* gpu_context,
                         horovod::common::HorovodGlobalState* global_state,
                         int quantization_bits, float multiplier)
      : GPUQuantizer(gpu_context, global_state, quantization_bits),
        multiplier_(multiplier) {}

  Status Init(const std::vector<TensorTableEntry>& entries) override;

protected:
  // Buffer to store static levels. Won't be sent.
  float* levels_ = nullptr;
  // multiplier used in case of Exponential quantization
  float multiplier_;
};

class GPUNormLinfQuantizer : public GPUNormalizedQuantizer {
public:
  GPUNormLinfQuantizer(GPUContext* gpu_context,
                       HorovodGlobalState* global_state, int quantization_bits,
                       float multiplier = -1.0);
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems) override;
  int64_t BufferSize(int num_elems) override;
};

class GPUNormL2Quantizer : public GPUNormalizedQuantizer {
public:
  GPUNormL2Quantizer(GPUContext* gpu_context, HorovodGlobalState* global_state,
                     int quantization_bits, float multiplier);
  int64_t Compress(unsigned char* input_data, unsigned char* output,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems) override;
  int64_t BufferSize(int num_elems) override;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_COMPRESSOR_H
