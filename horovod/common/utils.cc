#include "utils.h"

namespace horovod {
namespace common {

int64_t round_to(int64_t x, int64_t m) {
  return x + ((m - x % m) % m);
}

} // namespace common
} // namespace horovod

