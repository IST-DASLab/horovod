#include "error_feedback.h"
#include "../../../utils/env_parser.h"
#include "cuda/cuda_functions.h"

namespace horovod {
namespace common {

ErrorFeedback::ErrorFeedback(HorovodGlobalState* global_state,
                             struct GPUContext* gpu_context)
    : global_state_(global_state), gpu_context_(gpu_context) {
  SetBoolFromEnv(HOROVOD_COMPRESSION_ERROR_FEEDBACK, enabled, true);
}

Status ErrorFeedback::Init(const std::vector<TensorTableEntry>& entries) {
  if (!enabled)
    return Status::OK();
  int64_t offset = 0;
  Status status;
  for (auto& entry : entries) {
    // TODO: Add timeline callbacks
    status = bufferManager_.InitializeBuffer(
        entry.tensor_name,
        //            std::min(entry.tensor->size(),
        entry.tensor->shape().num_elements() * sizeof(float), entry.device,
        entry.context, [&]() {}, [&]() {});
    if (!status.ok())
      return status;
    offset += entry.tensor->size();
  }
  if (decompressed_buf == nullptr) {
    // Allocate extra space for decompressed buffer.
    // Current implementation of compression doesn't support error feedback.
    // So we have to decompress buffer to compute error feedback.
    std::string b_name = "decompressed_buffer";
    bufferManager_.InitializeBuffer(
        b_name, global_state_->parameter_manager.TensorFusionThresholdBytes(),
        entries[0].device, entries[0].context, [&]() {}, [&]() {});
    auto& buf = bufferManager_.GetBuffer(b_name, entries[0].device,
                                         entries[0].context->framework());
    decompressed_buf =
        (float*)const_cast<void*>(buf->AccessData(entries[0].context));
  }
  return Status::OK();
}

// v := e_fb + v
void ErrorFeedback::ApplyErrorFeedback(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    void* sendbuf, int64_t num_elements, int64_t global_offset) {
  if (!enabled)
    return;
  float* values_buffer = (float*)sendbuf;
  if (entries.size() == 1) {
    auto& entry = entries[0];
    auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device,
                                            entry.context->framework());
    float* error_fb_buf =
        (float*)const_cast<void*>(buffer->AccessData(entry.context)) +
        global_offset;
    CUDA_add(num_elements, error_fb_buf, values_buffer, values_buffer,
             gpu_context_
                 ->streams[global_state_->current_nccl_stream][entry.device]);
  } else {
    int64_t offset_cumm = 0;
    int n_elems;
    for (auto& entry : entries) {
      n_elems = std::min((int64_t)entry.tensor->shape().num_elements(),
                         num_elements - offset_cumm);
      auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device,
                                              entry.context->framework());
      float* error_fb_buf =
          (float*)const_cast<void*>(buffer->AccessData(entry.context));
      CUDA_add(n_elems, error_fb_buf, values_buffer + offset_cumm,
               values_buffer + offset_cumm,
               gpu_context_
                   ->streams[global_state_->current_nccl_stream][entry.device]);
      offset_cumm += n_elems;
    }
  }
}

// e_fb := e_fb + v - D(v)
void ErrorFeedback::UpdateErrorFeedback(
    const std::vector<TensorTableEntry>& entries, const void* original_buf,
    void* compressed_buf, int64_t chunk_num_elements, int64_t fusion_offset,
    int64_t global_offset, horovod::common::Compressor* compressor) {
  if (!enabled)
    return;
  unsigned char* compressed_buffer = (unsigned char*)compressed_buf;
  float* original_buffer = (float*)original_buf;
  auto& stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];
  float* decompressed_buffer = decompressed_buf;
  compressor->Decompress(compressed_buffer, (unsigned char*)decompressed_buffer,
                         chunk_num_elements);
  //  compressor_->Decompress(compressed_buffer, (unsigned
  //  char*)decompressed_buffer, entries,
  //      fusion_offset, global_offset, chunk_num_elements);

  if (entries.size() == 1) {
    auto& entry = entries[0];
    auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device,
                                            entry.context->framework());
    float* error_fb_buf =
        (float*)const_cast<void*>(buffer->AccessData(entry.context)) +
        fusion_offset + global_offset;
    // e_fb := v - D(Q(v))
    CUDA_diff(chunk_num_elements, decompressed_buffer, original_buffer,
              error_fb_buf, stream);
  } else {
    int n_elems;
    int64_t offset_cumm = 0;
    for (auto& entry : entries) {
      n_elems = entry.tensor->shape().num_elements();
      if (offset_cumm + n_elems <= fusion_offset) {
        offset_cumm += n_elems;
        continue;
      }
      if (offset_cumm - fusion_offset >= chunk_num_elements)
        break;
      if (offset_cumm < fusion_offset) {
        // If the first part of param group is placed in previous slice
        // depending on reduction algorithm.
        n_elems = offset_cumm + n_elems - fusion_offset;
      }
      if (std::max(offset_cumm, fusion_offset) + n_elems >
          fusion_offset + chunk_num_elements) {
        // if layer doesn't fit the rest of slice.
        n_elems = fusion_offset + chunk_num_elements -
                  std::max(offset_cumm, fusion_offset);
      }
      auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device,
                                              entry.context->framework());
      int64_t off = std::max(offset_cumm - fusion_offset, 0l);
      float* error_fb_buf =
          ((float*)const_cast<void*>(buffer->AccessData(entry.context))) +
          std::max(fusion_offset - offset_cumm, 0l);
      // e_fb := v - D(Q(v))
      CUDA_diff(n_elems, decompressed_buffer + off, original_buffer + off,
                error_fb_buf, stream);
      offset_cumm += entry.tensor->shape().num_elements();
    }
  }
}

} // namespace common
} // namespace horovod
