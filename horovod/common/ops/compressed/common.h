#ifndef COMPRESSED_COMMON_H
#define COMPRESSED_COMMON_H
#include "compression/gpu_compressor.h"

namespace horovod {
namespace common {

const int BUFFER_THRESHOLD = 1000;
const float QUANTIZE_MULTIPLIER = 0.5;

Compressor* CreateGPUCompressor(GPUContext* gpu_context,
                                HorovodGlobalState* global_state,
                                Summator* summator);

float* FillLevels(int bits, int& size, CompressionType compression_type, LevelsType levels_type);

} // namespace common
} // namespace horovod

#endif // COMPRESSED_COMMON_H
