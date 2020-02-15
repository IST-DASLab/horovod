#ifndef HOROVOD_COMMON_OPS_COMPRESSED_UTILS
#define HOROVOD_COMMON_OPS_COMPRESSED_UTILS

#include <chrono>
#include <cstdint>

namespace horovod {
namespace common {
using clock_ = std::chrono::steady_clock;
int64_t round_to(int64_t x, int64_t m);

double time_since(std::chrono::time_point<clock_> start_);

} // namespace common
} // namespace horovod

#endif // HOROVOD_COMMON_OPS_COMPRESSED_UTILS