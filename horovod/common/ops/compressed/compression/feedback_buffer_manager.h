#ifndef HOROVOD_FEEDBACK_BUFFER_H
#define HOROVOD_FEEDBACK_BUFFER_H
#include <iostream>
#include <unordered_map>

#include "../../../hashes.h"
#include "../../../operations.h"

namespace horovod {
namespace common {

// Encapsulates the process of creating and destroying feedback buffers.
// Assumes that size of the tensor won't change during the training.
class FeedbackBufferManager {
public:
  Status InitializeBuffer(const std::string& name, int64_t tensor_size,
                          int device, std::shared_ptr<OpContext> context, std::function<void()> on_start_init,
                          std::function<void()> on_end_init);

  // Returns the buffer associated with the given device and framework, or null.
  std::shared_ptr<PersistentBuffer>& GetBuffer(const std::string& tensor_name,
      int device, Framework framework);

private:
  // Memory buffers for Tensor Fusion.  They are keyed off device ID and
  // framework, and all are allocated tensor_fusion_threshold bytes if
  // initialized.
  std::unordered_map<
  std::tuple<std::string, int, Framework>,
  std::shared_ptr<PersistentBuffer>> feedback_buffers_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_FEEDBACK_BUFFER_H
