#ifndef HOROVOD_COMMON_OPS_COMPRESSED_UTILS
#define HOROVOD_COMMON_OPS_COMPRESSED_UTILS

#include <chrono>
#include <cstdint>
#include "../../message.h"

namespace horovod {
namespace common {
using clock_ = std::chrono::steady_clock;
using seconds_type = std::chrono::duration<double>;

const int ALIGNMENT_UNIT = 2 * sizeof(float);
#define ALIGNED_SIZE(size) round_to(size, ALIGNMENT_UNIT)

int64_t round_to(int64_t x, int64_t m);

double time_since(std::chrono::time_point<clock_>& start_);

int get_sizeof(DataType dtype);

} // namespace common
} // namespace horovod

#endif // HOROVOD_COMMON_OPS_COMPRESSED_UTILS