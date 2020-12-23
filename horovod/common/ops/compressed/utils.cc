#include "utils.h"
#include "compression/cuda/cuda_def.h"
#include <stdexcept>
#include <iostream>

namespace horovod {
namespace common {
size_t round_to(size_t x, int64_t m) {
  return x + ((m - x % m) % m);
}

double time_since(std::chrono::time_point<clock_>& start_) {
  return std::chrono::duration_cast<seconds_type>(clock_::now() -
                                                          start_).count();
}

int get_sizeof(DataType dtype) {
  switch (dtype) {
  case HOROVOD_FLOAT16:
    return sizeof(Half);
  case HOROVOD_FLOAT32:
    return sizeof(float);
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in compressed mode.");
  }
}
} // namespace common
} // namespace horovod
