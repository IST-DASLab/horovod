#include "utils.h"
namespace horovod {
namespace common {
int64_t round_to(int64_t x, int64_t m) { return x + ((m - x % m) % m); }

double time_since(std::chrono::time_point<clock_> start_) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_)
      .count();
}

}
}

