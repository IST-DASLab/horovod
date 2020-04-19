#ifndef HOROVOD_GPU_COMPRESSOR_H
#define HOROVOD_GPU_COMPRESSOR_H
#include "compressor.h"
#include "cuda/cuda_functions.h"
#include "../../gpu_operations.h"

namespace horovod {
namespace common {

class GPUDummyCompressor : public DummyCompressor {
public:
  GPUDummyCompressor(GPUContext* gpu_context, HorovodGlobalState* global_state)
      : gpu_context_(gpu_context), DummyCompressor(global_state), gpu_op_context_(gpu_context, global_state) {}

  int64_t Compress(unsigned char* input, unsigned char* output,
                   unsigned char* feedback,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems) override;
  Status Init(const std::vector<TensorTableEntry>& entries) override;
  void Finalize() override;

private:
  GPUContext* gpu_context_;
  GPUOpContext gpu_op_context_;
  int device_;
};

class GPUCompressionContext {
public:
  GPUCompressionContext(GPUContext* gpu_context, HorovodGlobalState* global_state)
      : gpu_context_(gpu_context), gpu_op_context_(gpu_context, global_state),
        current_nccl_stream_(global_state->current_nccl_stream),
        chunk_size_(
            global_state->parameter_manager.TensorFusionThresholdBytes()) {}

  virtual Status Init(const std::vector<TensorTableEntry>& entries);
protected:
  GPUContext* gpu_context_;
  GPUOpContext gpu_op_context_;
  CurandState* cuda_states_ = nullptr;
  int device_;
  int current_nccl_stream_;
  int chunk_size_;
  FusionBufferManager bufferManager_;
};

class GPUMaxMinQuantizer : public GPUCompressionContext, public MaxMinQuantizer {
public:
  GPUMaxMinQuantizer(GPUContext* gpu_context, HorovodGlobalState* global_state,
                     int quantization_bits);
  Status Init(const std::vector<TensorTableEntry>& entries) override;
  int64_t Compress(unsigned char* input, unsigned char* output,
                   unsigned char* feedback,
                   int64_t num_elems) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems) override;
  int64_t BufferSize(int num_elems) final;
  void Finalize() override;
};

class GPUNormalizedQuantizer : public NormalizedQuantizer {
public:
  GPUNormalizedQuantizer(horovod::common::HorovodGlobalState* global_state,
                         int quantization_bits, float multiplier)
      : NormalizedQuantizer(global_state, quantization_bits, multiplier) {}

  Status Init(GPUContext* gpu_context,
              const std::vector<horovod::common::TensorTableEntry>& entries);
};

class GPUNormLinfQuantizer : public GPUNormalizedQuantizer,
                             public GPUCompressionContext {
public:
  GPUNormLinfQuantizer(GPUContext* gpu_context,
                       HorovodGlobalState* global_state, int quantization_bits,
                       float multiplier = -1.0);
  Status Init(const std::vector<TensorTableEntry>& entries) override;
  int64_t Compress(unsigned char* input, unsigned char* output,
                   unsigned char* feedback, int64_t num_elems) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems) override;
  int64_t BufferSize(int num_elems) final;
  void Finalize() override;
};

class GPUNormL2Quantizer : public GPUNormalizedQuantizer,
                           public GPUCompressionContext {
public:
  GPUNormL2Quantizer(GPUContext* gpu_context, HorovodGlobalState* global_state,
                     int quantization_bits, float multiplier);
  Status Init(const std::vector<TensorTableEntry>& entries) override;

  int64_t Compress(unsigned char* input, unsigned char* output,
                   unsigned char* feedback, int64_t num_elems) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems) override;
  int64_t BufferSize(int num_elems) final;
  void Finalize() override;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_GPU_COMPRESSOR_H
