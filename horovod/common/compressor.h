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
  Compressor();
  // Returns size of buffer to allocate for usage in compress (in bytes). We assume that no compression will be done in-place.
  virtual int64_t BufferSize(int chunk_size) = 0;
  // Returns size of meaningful data inside data (in bytes).
  virtual int64_t Compress(unsigned char* input_data, void** output_p,
                           int64_t num_elems) = 0;
  virtual void Decompress(unsigned char* input, void** output_p,
                          int64_t num_elems) = 0;
  // Correct input_data based on compression.
  virtual void Correct(void* input_data, int num_elems) = 0;
  virtual Status Init(HorovodGlobalState* globalState,
                      const std::vector<TensorTableEntry>& entries) = 0;
  int64_t meta_info_time = 0;
  int64_t compression_time = 0;
protected:
  // The size of the bucket.
  int bucket_size_;
};

class DummyCompressor: public Compressor {
public:
  DummyCompressor():Compressor(){}

  int64_t BufferSize(int chunk_size) override { return chunk_size; }
  // No need in correction
  void Correct(void* input_data, int num_elems) override {}
  int64_t Compress(unsigned char* input_data, void** output_p,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, void** output_p,
                  int64_t num_elems) override;
  Status Init(HorovodGlobalState* globalState,
              const std::vector<TensorTableEntry>& entries) override {
    return Status::OK();}
};

class CUDACompressor: public Compressor {
public:
  CUDACompressor(CUDAContext *cuda_context): Compressor(), cuda_context_(cuda_context){}
  // Returns size of buffer to pass in compress (in bytes)
protected:
  CUDAContext *cuda_context_;
};


class CUDAQuantizer: public CUDACompressor {
public:
  CUDAQuantizer(CUDAContext *cuda_context, HorovodGlobalState* global_state);
  // No need in correction
  void Correct(void* input_data, int num_elems) override {}

  Status Init(HorovodGlobalState* globalState,
              const std::vector<TensorTableEntry>& entries) override;
protected:
  CurandState* cuda_states_ = nullptr;
  int device_;
  // Number of bits used per value.
  int bits_;
  FusionBufferManager bufferManager_;
private:
  size_t curand_array_size;
};

class CUDAMaxMinQuantizer: public CUDAQuantizer {
public:
  CUDAMaxMinQuantizer(CUDAContext *cuda_context,
                                           HorovodGlobalState* global_state): CUDAQuantizer(cuda_context, global_state){}
  int64_t Compress(unsigned char* input_data, void** output_p,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, void** output_p,
                  int64_t num_elems) override;
  int64_t BufferSize(int chunk_size) override;
};

class CUDANormalizedQuantizer: public CUDAQuantizer {
public:
  explicit CUDANormalizedQuantizer(CUDAContext *cuda_context, HorovodGlobalState* global_state):CUDAQuantizer(cuda_context, global_state), multiplier_(-1.0){}
  CUDANormalizedQuantizer(CUDAContext *cuda_context, HorovodGlobalState* global_state, float multiplier):CUDAQuantizer(cuda_context, global_state), multiplier_(multiplier){}
  friend CUDANormalizedQuantizer *CreateCUDANormalized(CUDAContext* cuda_context, HorovodGlobalState* global_state);

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

CUDANormalizedQuantizer *CreateCUDANormalized(CUDAContext* cuda_context, HorovodGlobalState* global_state);

class CUDANormLinfQuantizer: public CUDANormalizedQuantizer {
public:
  CUDANormLinfQuantizer(CUDAContext *cuda_context,
                                               HorovodGlobalState* global_state): CUDANormalizedQuantizer(cuda_context, global_state){}
  CUDANormLinfQuantizer(CUDAContext *cuda_context,
                                               HorovodGlobalState* global_state, float multiplier): CUDANormalizedQuantizer(cuda_context, global_state, multiplier){}
  int64_t Compress(unsigned char* input_data, void** output_p,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, void** output_p,
                  int64_t num_elems) override;
  Status Init(HorovodGlobalState* globalState,
              const std::vector<TensorTableEntry>& entries) override;
    int64_t BufferSize(int chunk_size) override;
};

class CUDANormL2Quantizer: public CUDANormalizedQuantizer {
public:
  CUDANormL2Quantizer(
      horovod::common::CUDAContext* cuda_context, HorovodGlobalState* global_state, float multiplier): CUDANormalizedQuantizer(cuda_context, global_state, multiplier){}
  int64_t Compress(unsigned char* input_data, void** output,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, void** output_p,
                  int64_t num_elems) override;
  Status Init(HorovodGlobalState* globalState,
              const std::vector<TensorTableEntry>& entries) override;
    int64_t BufferSize(int chunk_size) override;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_COMPRESSOR_H
