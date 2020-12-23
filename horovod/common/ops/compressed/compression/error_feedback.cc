#include "error_feedback.h"
#include "../../../utils/env_parser.h"
#include "../utils.h"

#include "../reducers/reducer.h"

namespace horovod {
namespace common {

ErrorFeedback::ErrorFeedback(Summator* summator) : summator_(summator) {
  SetBoolFromEnv(HOROVOD_COMPRESSION_ERROR_FEEDBACK, enabled_, true);
}
ErrorFeedback::~ErrorFeedback() { delete summator_; }
Status ErrorFeedback::Init(const std::vector<TensorTableEntry>& entries) {
  if (!enabled_)
    return Status::OK();
  Status status;
  for (auto& entry : entries) {
    // TODO: Add timeline callbacks
    status = bufferManager_.InitializeBuffer(
        entry.tensor_name,
        entry.tensor->shape().num_elements() *
            get_sizeof(entry.tensor->dtype()),
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
    if (entry.tensor->dtype() == HOROVOD_FLOAT32) {
      float* error_fb_buf =
          (float*)const_cast<void*>(buffer->AccessData(entry.context));
      summator_->Add(error_fb_buf, entry, n_elems);
    } else {
      Half* error_fb_buf =
          (Half*)const_cast<void*>(buffer->AccessData(entry.context));
      summator_->Add(error_fb_buf, entry, n_elems);
    }
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

void ErrorFeedback::CopyToErrorFeedback(
    unsigned char* feedback_buffer_input,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    int num_elems, int fusion_offset, void* ctx) {
  assert(entries.size() > 1);
  int offset_cumm = 0;
  int nelem = 0;
  int buffer_offset = 0, entry_offset = 0;
  auto dtype = entries[0].tensor->dtype();
  unsigned char* entry_feedback_data;
  cudaStream_t* stream = (cudaStream_t*)ctx;
  for (auto& entry : entries) {
    nelem = entry.tensor->shape().num_elements();
    entry_offset = 0;
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }

    if (offset_cumm - fusion_offset >= num_elems) {
      break;
    }

    if (offset_cumm < fusion_offset) {
      // If the first part of entry is placed in the previous slice.
      nelem = offset_cumm + nelem - fusion_offset;
      entry_offset = entry.tensor->shape().num_elements() - nelem;
    }

    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + num_elems - std::max(offset_cumm, fusion_offset);
    }

    buffer_offset = std::max(offset_cumm - fusion_offset, 0);
    auto offset = buffer_offset * get_sizeof(dtype);
    entry_feedback_data = GetData(entry) + entry_offset * get_sizeof(dtype);

    if (entry.device == CPU_DEVICE_ID) {
      memcpy(feedback_buffer_input + offset, entry_feedback_data,
             nelem * get_sizeof(dtype));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(entry_feedback_data, feedback_buffer_input + offset,
                                 nelem * get_sizeof(dtype),
                                 cudaMemcpyDeviceToDevice, *stream));
    }
    offset_cumm += entry.tensor->shape().num_elements();
  }
}

} // namespace common
} // namespace horovod
