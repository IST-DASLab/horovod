#include "error_feedback.h"
#include "../../../utils/env_parser.h"
#include "cuda/cuda_functions.h"

namespace horovod {
namespace common {

ErrorFeedback::ErrorFeedback(Summator* summator) : summator_(summator) {
  SetBoolFromEnv(HOROVOD_COMPRESSION_ERROR_FEEDBACK, enabled_, true);
}

Status ErrorFeedback::Init(const std::vector<TensorTableEntry>& entries) {
  if (!enabled_)
    return Status::OK();
  Status status;
  for (auto& entry : entries) {
    // TODO: Add timeline callbacks
    status = bufferManager_.InitializeBuffer(
        entry.tensor_name, entry.tensor->shape().num_elements() * sizeof(float),
        entry.device, entry.context, [&]() {}, [&]() {});
    if (!status.ok())
      return status;
  }
  return Status::OK();
}

// v := e_fb + v
void ErrorFeedback::Apply(std::vector<TensorTableEntry>& entries) {
  if (!enabled_)
    return;
  for (auto& entry : entries) {
    int n_elems = entry.tensor->shape().num_elements();
    auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device,
                                            entry.context->framework());
    float* error_fb_buf =
        (float*)const_cast<void*>(buffer->AccessData(entry.context));
    summator_->Add(error_fb_buf, entry, n_elems);
  }
}

unsigned char*
ErrorFeedback::GetData(const horovod::common::TensorTableEntry& entry) {
  if (!enabled_)
    return nullptr;
  auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device,
                                          entry.context->framework());
  return (unsigned char*)const_cast<void*>(buffer->AccessData(entry.context));
}

} // namespace common
} // namespace horovod
