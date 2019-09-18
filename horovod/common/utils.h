#ifndef HOROVOD_UTILS_H
#define HOROVOD_UTILS_H
#include <cstdint>

namespace horovod {
namespace common {

int64_t round_to(int64_t x, int64_t m);

} // namespace common
} // namespace horovod

#endif // HOROVOD_UTILS_H
