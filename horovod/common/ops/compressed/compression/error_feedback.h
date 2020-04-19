#ifndef HOROVOD_ERROR_FEEDBACK_H
#define HOROVOD_ERROR_FEEDBACK_H

#include "feedback_buffer_manager.h"
#include "vector_operations.h"

namespace horovod {
namespace common {

class ErrorFeedback {
public:
  ErrorFeedback(Summator* summator);
  // Initializes feedback buffers and computes offsets of the entries in feedback buffer.
  Status Init(const std::vector<TensorTableEntry>& entries);
  bool isEnabled() { return enabled_;}
  // Apply saved error feedback
  void Apply(std::vector<TensorTableEntry>& entries);
  // Get error feedback buffer to update during compression.
  unsigned char* GetData(const TensorTableEntry& entry);
private:
  Summator* summator_;
  FeedbackBufferManager bufferManager_;
  bool enabled_ = false;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_ERROR_FEEDBACK_H