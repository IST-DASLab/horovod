#ifndef HOROVOD_GPU_COMPRESSOR_H
#define HOROVOD_GPU_COMPRESSOR_H
#include "compressor.h"

namespace horovod {
namespace common {

class GPUCompressionContext {
public:
  GPUCompressionContext(GPUContext* gpu_context,
                        HorovodGlobalState* global_state)
      : gpu_context_(gpu_context), gpu_op_context_(gpu_context, global_state),
        global_state_(global_state) {}

  Status Init(const std::vector<TensorTableEntry>& entries);

  GPUContext* gpu_context_;
  GPUOpContext gpu_op_context_;
  HorovodGlobalState* global_state_;
  CurandState* cuda_states_ = nullptr;
  int device_;
  FusionBufferManager bufferManager_;
};

class GPUDummyCompressor : public DummyCompressor {
public:
  GPUDummyCompressor(GPUContext* gpu_context, HorovodGlobalState* global_state,
                     Summator* summator)
      : DummyCompressor(global_state, summator) {
    gpu_compression_context_ = std::unique_ptr<GPUCompressionContext>(
        new GPUCompressionContext(gpu_context, global_state));
  }

  size_t CompressBuffer(unsigned char* input, unsigned char* output,
                  unsigned char* feedback, int num_elems, DataType dtype,
                  const CompressionModuleConfig& compression_cfg,
                  void* ctx) override;
  void DecompressBuffer(unsigned char* input, unsigned char* output,
                        int num_elems, DataType dtype, bool add,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  Status Init(const std::vector<TensorTableEntry>& entries) override;

private:
  std::unique_ptr<GPUCompressionContext> gpu_compression_context_;
};

class GPUMaxMinQuantizer : public MaxMinQuantizer {
public:
  GPUMaxMinQuantizer(GPUContext* gpu_context, HorovodGlobalState* global_state,
                     Summator* summator, int quantization_bits)
      : MaxMinQuantizer(global_state, summator, quantization_bits) {
    gpu_compression_context_ = std::unique_ptr<GPUCompressionContext>(
        new GPUCompressionContext(gpu_context, global_state));
  };

  Status Init(const std::vector<TensorTableEntry>& entries) override;
  size_t CompressBuffer(unsigned char* input, unsigned char* output,
                        unsigned char* feedback, int num_elems, DataType dtype,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  void DecompressBuffer(unsigned char* input, unsigned char* output,
                        int num_elems, DataType dtype, bool add,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  size_t BufferSize(int num_elems, DataType dtype,
                    const CompressionModuleConfig& compression_cfg) final;
  virtual size_t GetRequiredFreeSize() final;

private:
  std::unique_ptr<GPUCompressionContext> gpu_compression_context_;
};

class GPUNormalizedQuantizer : public NormalizedQuantizer {
public:
  GPUNormalizedQuantizer(GPUContext* gpu_context,
                         HorovodGlobalState* global_state, Summator* summator,
                         int quantization_bits,
                         CompressionType compression_type, NormType norm_type,
                         LevelsType levels_type)
      : NormalizedQuantizer(global_state, summator, quantization_bits,
                            compression_type, norm_type, levels_type) {
    gpu_compression_context_ = std::unique_ptr<GPUCompressionContext>(
        new GPUCompressionContext(gpu_context, global_state));
  }

  Status Init(const std::vector<horovod::common::TensorTableEntry>& entries);
  void SetQuantizationLevels(float* levels, int bits) override;
  size_t BufferSize(int num_elems, DataType dtype,
                    const CompressionModuleConfig& compression_cfg) final;
  size_t CompressBuffer(unsigned char* input, unsigned char* output,
                        unsigned char* feedback, int num_elems, DataType dtype,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  void DecompressBuffer(unsigned char* input, unsigned char* output,
                        int num_elems, DataType dtype, bool add,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  virtual size_t GetRequiredFreeSize() final;

protected:
  std::map<int, Half*> bits_to_levels_fp16_;
  std::unique_ptr<GPUCompressionContext> gpu_compression_context_;
};

class GPUTopKCompressor : public Compressor {
public:
  GPUTopKCompressor(GPUContext* gpu_context, HorovodGlobalState* global_state,
                    Summator* summator, float topk_ratio);

  size_t BufferSize(int num_elems, DataType dtype,
                    const CompressionModuleConfig& compression_cfg) final;
  size_t CompressBuffer(unsigned char* input, unsigned char* output,
                        unsigned char* feedback, int num_elems, DataType dtype,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  void DecompressBuffer(unsigned char* input, unsigned char* output,
                        int num_elems, DataType dtype, bool add,
                        const CompressionModuleConfig& compression_cfg,
                        void* ctx) override;
  Status Init(const std::vector<horovod::common::TensorTableEntry>& entries);
  virtual size_t GetRequiredFreeSize() final;

protected:
  std::unique_ptr<GPUCompressionContext> gpu_compression_context_;
  unsigned char* utility_buf_ = nullptr;
  float topk_ratio_;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_GPU_COMPRESSOR_H
