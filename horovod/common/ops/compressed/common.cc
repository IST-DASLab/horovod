#include "common.h"

namespace horovod {
namespace common {

Compressor* CreateGPUCompressor(GPUContext* gpu_context,
                                HorovodGlobalState* global_state) {
  auto compression_type = GetEnumEnvOrDefault<CompressionType>(
      HOROVOD_COMPRESSION, CompressionType::NoneCompression);

  auto quantization_bits = GetIntEnvOrDefault(HOROVOD_QUANTIZATION_BITS, 32);
  Compressor* compressor;
  if (quantization_bits == 32 || compression_type == NoneCompression) {
    compressor = new GPUDummyCompressor(gpu_context, global_state);
  } else {
    switch (compression_type) {
    case CompressionType::MaxMin:
      compressor =
          new GPUMaxMinQuantizer(gpu_context, global_state, quantization_bits);
      break;
    case CompressionType::ExpL2:
      compressor = new GPUNormL2Quantizer(
          gpu_context, global_state, quantization_bits, QUANTIZE_MULTIPLIER);
      break;
    case CompressionType::Uni:
      compressor = new GPUNormLinfQuantizer(gpu_context, global_state,
                                            quantization_bits);
      break;
    case CompressionType::ExpLinf:
      compressor = new GPUNormLinfQuantizer(
          gpu_context, global_state, quantization_bits, QUANTIZE_MULTIPLIER);
      break;
    default:
      throw std::logic_error("Invalid compression type.");
    }
  }
  return compressor;
}

} // namespace common
} // namespace horovod
