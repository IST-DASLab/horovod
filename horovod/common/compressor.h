#ifndef HOROVOD_COMPRESSOR_H
#define HOROVOD_COMPRESSOR_H

#include "global_state.h"
#include "ops/cuda_functions.h"
#include "ops/cuda_operations.h"

namespace horovod {
namespace common{
const int COMPRESSION_BUCKET_SIZE = 512;
const float QUANTIZE_MULTIPLIER = 0.5;
const float DEFAULT_TOPK = 0.1;

class Compressor {
public:
  Compressor(CUDAContext *cuda_context);
  // Returns size of buffer to pass in compress (in bytes)
  virtual int64_t BufferSize() = 0;
  // Returns size of meaningful data inside data (in bytes).
  virtual int64_t Compress(void *input_data, void *compressed_data) = 0;
  virtual void Decompress(void *compressed_data, void *decompressed_data) = 0;
  // Correct input_data based on compression.
  virtual void Correct(void *input_data) = 0;
  virtual Status Init(HorovodGlobalState* globalState, int num_elements,
                      std::vector<TensorTableEntry>& entries) = 0;
  int64_t meta_info_time = 0;
  int64_t compression_time = 0;
protected:
  CUDAContext *cuda_context_;
  // The size of the bucket.
  int bucket_size_;

};

class Quantizer: public Compressor {
public:
  Quantizer(CUDAContext *cuda_context):Compressor(cuda_context){}
  int64_t BufferSize() override
  { return allocate_buffer_size_; }
  // No need in correction
  void Correct(void *input_data) override {}

  Status Init(HorovodGlobalState* globalState, int num_elements,
              std::vector<TensorTableEntry>& entries) override;
protected:
  CurandState* cuda_states_ = nullptr;
  // Size of previously processed chunk
  int64_t prev_chunk_size_ = -1;
  // Number of elements in chunk.
  int num_elements_;
  // Size of buffer used for any chunk
  int64_t allocate_buffer_size_;
  int device_;
  // Number of bits used per value.
  int bits_;
  FusionBufferManager bufferManager_;
  int64_t meta_buffer_size_;
};

class MaxMinQuantizer: public Quantizer {
public:
  MaxMinQuantizer(CUDAContext *cuda_context): Quantizer(cuda_context){
    std::cout << "MaxMinQuantizer" << std::endl;
  }
  int64_t Compress(void *input_data, void *compressed_data) override;
  void Decompress(void *compressed_data, void *decompressed_data) override;

  Status Init(HorovodGlobalState* globalState, int num_elements,
              std::vector<TensorTableEntry>& entries) override;
};

class NormalizedQuantizer: public Quantizer {
public:
  explicit NormalizedQuantizer(CUDAContext *cuda_context):Quantizer(cuda_context), multiplier_(-1.0){}
  NormalizedQuantizer(CUDAContext *cuda_context, float multiplier):Quantizer(cuda_context), multiplier_(multiplier){}
  friend NormalizedQuantizer *CreateNormalized(CUDAContext *cuda_context);

protected:
  enum NormalizationType {
    Linf,
    L2
  };
  // Buffer to store static levels. Won't be sent.
  float *levels_ = nullptr;
  // multiplier in case of Exponential quantization
  float multiplier_ = QUANTIZE_MULTIPLIER;
};

NormalizedQuantizer *CreateNormalized(CUDAContext *cuda_context);

class NormLinfQuantizer: public NormalizedQuantizer {
public:
    NormLinfQuantizer(CUDAContext *cuda_context): NormalizedQuantizer(cuda_context){}
    NormLinfQuantizer(CUDAContext *cuda_context, float multiplier): NormalizedQuantizer(cuda_context, multiplier){}
    int64_t Compress(void *input_data, void *compressed_data) override;
    void Decompress(void *compressed_data, void *decompressed_data) override;
    Status Init(HorovodGlobalState* globalState, int num_elements,
                std::vector<TensorTableEntry>& entries) override;
};

class NormL2Quantizer: public NormalizedQuantizer {
public:
    NormL2Quantizer(CUDAContext *cuda_context, float multiplier): NormalizedQuantizer(cuda_context, multiplier){}
    int64_t Compress(void *input_data, void *output) override;
    void Decompress(void *compressed_data, void *decompressed_data) override;
    Status Init(HorovodGlobalState* globalState, int num_elements,
                std::vector<TensorTableEntry>& entries) override;

private:
    // Size of buffer with L2 norms in meta info buffer.
    int norm_buffer_size_;
};
// Not implemented
class TopKcompressor: public Compressor {
public:
  TopKcompressor(CUDAContext* cudaContext):Compressor(cudaContext){}

protected:
  // TODO: move all constants, defaults to one place.
  float taken_amount = 0.1; // aka k
  Status Init(HorovodGlobalState* globalState, int num_elements,
              std::vector<TensorTableEntry>& entries) override;

};
} // namespace common
} // namespace horovod
#endif // HOROVOD_COMPRESSOR_H
