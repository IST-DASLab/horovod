#ifndef COMPRESSED_COMMON_H
#define COMPRESSED_COMMON_H
#include "compression/gpu_compressor.h"

namespace horovod {
namespace common {

const int BUFFER_THRESHOLD = 0;
const float QUANTIZE_MULTIPLIER = 0.5;

Compressor* CreateGPUCompressor(GPUContext* gpu_context,
                                HorovodGlobalState* global_state);

} // namespace common
} // namespace horovod

#endif // COMPRESSED_COMMON_H
