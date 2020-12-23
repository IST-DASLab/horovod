#include "common.h"
#include "compression/gpu_compressor.h"

namespace horovod {
namespace common {

Compressor* CreateGPUCompressor(GPUContext* gpu_context,
                                HorovodGlobalState* global_state,
                                Summator* summator) {
  auto compression_type = GetEnumEnvOrDefault<CompressionType>(
      HOROVOD_COMPRESSION, CompressionType::NoneCompression);
  auto norm_type = GetEnumEnvOrDefault<NormType>(HOROVOD_COMPRESSION_NORM_TYPE,
                                                 NormType::Linf);
  auto levels_type = GetEnumEnvOrDefault<LevelsType>(
      HOROVOD_COMPRESSION_LEVELS_TYPE, LevelsType::Pos);

  auto quantization_bits = GetIntEnvOrDefault(HOROVOD_QUANTIZATION_BITS, 32);
  float topk_ratio = GetDoubleEnvOrDefault(HOROVOD_COMPRESSION_TOPK_RATIO, 1.0);
  Compressor* compressor;
  if ((quantization_bits == 32 && topk_ratio == 1.0) ||
      compression_type == NoneCompression) {
    compressor = new GPUDummyCompressor(gpu_context, global_state, summator);
  } else {
    switch (compression_type) {
    case CompressionType::MaxMin:
      compressor = new GPUMaxMinQuantizer(gpu_context, global_state, summator,
                                          quantization_bits);
      break;
    case CompressionType::Exp:
    case CompressionType::Uni:
      compressor = new GPUNormalizedQuantizer(
          gpu_context, global_state, summator, quantization_bits,
          compression_type, norm_type, levels_type);
      break;
    case CompressionType::TopK:
      compressor = new GPUTopKCompressor(gpu_context, global_state, summator,
                                         topk_ratio);
      break;
    default:
      throw std::logic_error("Invalid compression type.");
    }
  }
  return compressor;
}

float* FillLevels(int bits, int& size, CompressionType compression_type,
                  LevelsType levels_type) {
  float* host_levels;
  int init_num_levels;
  if (compression_type == CompressionType::Uni) {
    if (levels_type == LevelsType::Wide) {
      init_num_levels = 1 << bits;
      host_levels = new float[init_num_levels];

      host_levels[0] = 1.0;
      host_levels[init_num_levels - 1] = -1.0;
      float value = 2.0 / (init_num_levels - 1);
      // Symmetric level assignment
      for (int i = 1; i < (init_num_levels - 1); i++) {
        host_levels[i] = -1.0 + (init_num_levels - 1 - i) * value;
      }
    } else {
      init_num_levels = 1 << (bits - 1);
      host_levels = new float[init_num_levels];

      host_levels[0] = 1.0;
      float value = 1.0 / (init_num_levels - 1);
      for (int i = 1; i <= (init_num_levels - 1); i++) {
        host_levels[i] = (init_num_levels - i - 1) * value;
      }
    }
  } else if (compression_type == CompressionType::Exp) {
    // In fact number of levels will be different. 1 bit is taken for sign.
    // Exponential levels
    if (levels_type == LevelsType::Wide) {
      init_num_levels = 1 << bits;
      host_levels = new float[init_num_levels];
      host_levels[0] = QUANTIZE_MULTIPLIER;
      host_levels[init_num_levels - 1] = -QUANTIZE_MULTIPLIER;
      float level_v = QUANTIZE_MULTIPLIER * QUANTIZE_MULTIPLIER;
      for (int i = 1; i <= (init_num_levels - 1) / 2; i++) {
        host_levels[i] = level_v;
        host_levels[init_num_levels - 1 - i] = -level_v;
        level_v *= QUANTIZE_MULTIPLIER;
      }
    } else {
      init_num_levels = 1 << (bits - 1);
      host_levels = new float[init_num_levels];
      host_levels[0] = QUANTIZE_MULTIPLIER;
      float level_v = QUANTIZE_MULTIPLIER * QUANTIZE_MULTIPLIER;
      for (int i = 1; i <= init_num_levels - 1; i++) {
        host_levels[i] = level_v;
        level_v *= QUANTIZE_MULTIPLIER;
      }
    }
  }
  size = init_num_levels;
  return host_levels;
}

} // namespace common
} // namespace horovod
