#ifndef HOROVOD_GPU_COMPRESSOR_H
#define HOROVOD_GPU_COMPRESSOR_H
#include "../../gpu_operations.h"
#include "compressor.h"
#include "cuda/cuda_functions.h"

namespace horovod {
namespace common {

class GPUCompressionContext {
public:
  GPUCompressionContext(GPUContext* gpu_context,
                        HorovodGlobalState* global_state)
      : gpu_context_(gpu_context), gpu_op_context_(gpu_context, global_state),
        global_state_(global_state) {}

  Status Init(const std::vector<TensorTableEntry>& entries);
  void Finalize();

  GPUContext* gpu_context_;
  GPUOpContext gpu_op_context_;
  HorovodGlobalState* global_state_;
  CurandState* cuda_states_ = nullptr;
  int device_;
  FusionBufferManager bufferManager_;
  gpuStream_t* stream_;
};

class GPUDummyCompressor : public DummyCompressor {
public:
  GPUDummyCompressor(GPUContext* gpu_context, HorovodGlobalState* global_state)
      : DummyCompressor(global_state) {
    gpu_compression_context_ = std::unique_ptr<GPUCompressionContext>(
        new GPUCompressionContext(gpu_context, global_state));
  }

  int64_t Compress(unsigned char* input, unsigned char* output,
                   unsigned char* feedback, int64_t num_elems, DataType dtype) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems, DataType dtype, bool add) override;
  Status Init(const std::vector<TensorTableEntry>& entries) override;
  void Finalize() override;

private:
  std::unique_ptr<GPUCompressionContext> gpu_compression_context_;
};

class GPUMaxMinQuantizer : public MaxMinQuantizer {
public:
  GPUMaxMinQuantizer(GPUContext* gpu_context, HorovodGlobalState* global_state,
                     int quantization_bits)
      : MaxMinQuantizer(global_state, quantization_bits) {
    gpu_compression_context_ = std::unique_ptr<GPUCompressionContext>(
        new GPUCompressionContext(gpu_context, global_state));
  };

  Status Init(const std::vector<TensorTableEntry>& entries) override;
  int64_t Compress(unsigned char* input, unsigned char* output,
                   unsigned char* feedback, int64_t num_elems,
                   DataType dtype) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems, DataType dtype, bool add) override;
  int64_t BufferSize(int num_elems, DataType dtype) final;
  void Finalize();

private:
  std::unique_ptr<GPUCompressionContext> gpu_compression_context_;
};

class GPUNormalizedQuantizer : public NormalizedQuantizer {
public:
  GPUNormalizedQuantizer(GPUContext* gpu_context,
                         horovod::common::HorovodGlobalState* global_state,
                         int quantization_bits,
                         CompressionType compression_type, NormType norm_type,
                         LevelsType levels_type)
      : NormalizedQuantizer(global_state, quantization_bits, compression_type,
                            norm_type, levels_type) {
    gpu_compression_context_ = std::unique_ptr<GPUCompressionContext>(
        new GPUCompressionContext(gpu_context, global_state));
  }

  Status Init(const std::vector<horovod::common::TensorTableEntry>& entries);
  void SetQuantizationLevels(float* levels) override;
  void Finalize();
  int64_t BufferSize(int num_elems, DataType dtype) final;
  int64_t Compress(unsigned char* input, unsigned char* output,
                   unsigned char* feedback, int64_t num_elems,
                   DataType dtype) override;
  void Decompress(unsigned char* input, unsigned char* output,
                  int64_t num_elems, DataType dtype, bool add) override;

protected:
  Half* levels_fp16_;
  std::unique_ptr<GPUCompressionContext> gpu_compression_context_;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_GPU_COMPRESSOR_H