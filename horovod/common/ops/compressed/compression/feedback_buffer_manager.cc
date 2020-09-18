#include "feedback_buffer_manager.h"

#include "../../../logging.h"
#include <cstring>

#include <cuda_runtime.h>

namespace horovod {
namespace common {

Status FeedbackBufferManager::InitializeBuffer(
    const std::string& tensor_name, int64_t tensor_size, int device,
    std::shared_ptr<horovod::common::OpContext> context,
    std::function<void()> on_start_init, std::function<void()> on_end_init) {
  on_start_init();
  auto& buffer = feedback_buffers_[std::make_tuple(tensor_name, device,
                                                   context->framework())];

  if (buffer == nullptr) {
    // Lazily allocate persistent buffer for Tensor Fusion and keep it
    // forever per device.
    Status status = context->AllocatePersistent(tensor_size, &buffer);

    LOG(DEBUG, 0) << "FeedbackBufferManager: Allocating " << tensor_size << " "
                 << tensor_name << std::endl;
    if (!status.ok()) {
      LOG(ERROR, 0) << "Allocation failed" << std::endl;
    }

    void* buf = const_cast<void*>(buffer->AccessData(context));
    if (device == CPU_DEVICE_ID) {
      std::memset(buf, 0, tensor_size);
    } else {
      cudaMemset(buf, 0, tensor_size);
      cudaDeviceSynchronize();
    }
  }
  on_end_init();
  return Status::OK();
}

std::shared_ptr<PersistentBuffer>&
FeedbackBufferManager::GetBuffer(const std::string& tensor_name, int device,
                                 horovod::common::Framework framework) {
  return feedback_buffers_[std::make_tuple(tensor_name, device, framework)];
}

} // namespace common
} // namespace horovod
