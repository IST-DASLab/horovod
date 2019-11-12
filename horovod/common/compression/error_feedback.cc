#include "error_feedback.h"
#include <cassert>


namespace horovod {
namespace common {

ErrorFeedback::ErrorFeedback(HorovodGlobalState* global_state, struct CUDAContext* cuda_context): global_state_(global_state), cuda_context_(cuda_context){
  const char *env_str = getenv(HOROVOD_COMPRESSION_ERROR_FEEDBACK);
  if (env_str != nullptr && std::strtol(env_str, nullptr, 10) > 0)
    enabled = true;
}



Status ErrorFeedback::Init(const std::vector<TensorTableEntry>& entries) {
  if (!enabled)
    return Status::OK();
  int64_t offset = 0;
  Status status;
  for (auto& entry: entries) {
    // TODO: Add timeline callbacks
    status = bufferManager_.InitializeBuffer(entry.tensor_name,
//            std::min(entry.tensor->size(),
            entry.tensor->shape().num_elements() * sizeof(float),
            entry.device, entry.context,
                                    [&]() {},
                                    [&]() {});
    if (!status.ok())
      return status;
    offset += entry.tensor->size();
  }
  if (decompressed_buf == nullptr) {
    // Allocate extra space for decompressed buffer.
    // Current implementation of compression doesn't support error feedback.
    // So we have to decompress buffer to compute error feedback.
    std::string b_name = "decompressed_buffer";
    bufferManager_.InitializeBuffer(b_name,
                                    global_state_->param_manager.TensorFusionThresholdBytes(),
                                    entries[0].device, entries[0].context,
                                    [&]() {},
                                    [&]() {});
    auto& buf = bufferManager_.GetBuffer(b_name,
        entries[0].device, entries[0].context->framework());
    decompressed_buf = (float*) const_cast<void*>(buf->AccessData(entries[0].context));
  }
  return Status::OK();
}

void printDebug4(float *buff, int n=8) {
  float *debugarr = new float[n];
  cudaMemcpy(debugarr, buff, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    std::cout << debugarr[i] << " ";
  }
  std::cout << std::endl;
}


// v := e_fb + v
void ErrorFeedback::ApplyErrorFeedback(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    void* sendbuf, int64_t num_elements, int64_t global_offset) {
  if (!enabled)
    return;
  float* values_buffer = (float*) sendbuf;
  if (entries.size() == 1) {
    auto &entry = entries[0];
    auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device, entry.context->framework());
    float* error_fb_buf = (float*) const_cast<void*>(buffer->AccessData(entry.context))
                                   + global_offset;
//    if (global_state_->rank == 0) {
//      std::cout << entry.tensor_name << " Original ";
//      printDebug4(values_buffer);
//      std::cout << "Feedback ";
//      printDebug4(error_fb_buf);
//    }

    CUDA_add(num_elements, error_fb_buf, values_buffer, cuda_context_->streams[entry.device]);
  } else {
    int64_t offset_cumm = 0;
    int n_elems;
    for (auto& entry: entries) {
      n_elems = std::min((int64_t) entry.tensor->shape().num_elements(), num_elements - offset_cumm);
      auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device, entry.context->framework());
      float* error_fb_buf = (float*)const_cast<void*>(buffer->AccessData(entry.context));
      CUDA_add(n_elems, error_fb_buf, values_buffer + offset_cumm, cuda_context_->streams[entry.device]);
      offset_cumm += n_elems;
    }
  }
}

// e_fb := e_fb + v - D(v)
void ErrorFeedback::UpdateErrorFeedback(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    const void* original_buf, void* compressed_buf,
    int64_t num_elements, int64_t offset_fusion, int64_t global_offset,
    horovod::common::Compressor* compressor) {
  if (!enabled)
    return;
  unsigned char* compressed_buffer = (unsigned char*) compressed_buf;
  float* original_buffer = (float*) original_buf;
  auto& stream = cuda_context_->streams[entries[0].device];
  float* decompressed_buffer = decompressed_buf;
  compressor->Decompress(compressed_buffer, (void**)&decompressed_buffer, num_elements);

  if (entries.size() == 1) {
    auto &entry = entries[0];
    auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device, entry.context->framework());
    float* error_fb_buf = (float*)const_cast<void*>(buffer->AccessData(entry.context))
        + offset_fusion + global_offset;
//    if (global_state_->rank == 0) {
//      std::cout << "Original ";
//      printDebug4(original_buffer);
//      std::cout << "Decompressed ";
//      printDebug4(decompressed_buffer);
//      std::cout << "Feedback ";
//      printDebug4(error_fb_buf);
//    }
    // e_fb := e_fb + v
    CUDA_add(num_elements, original_buffer, error_fb_buf, stream);
    // e_fb := e_fb - D(v)
    CUDA_substract(num_elements, decompressed_buffer, error_fb_buf, stream);
//    if (global_state_->rank == 0) {
//      std::cout << "Feedback after";
//      printDebug4(error_fb_buf);
//    }
  } else {
    int n_elems;
    int64_t offset_cumm = 0;
    for (auto& entry : entries) {
      n_elems = std::min((int64_t) entry.tensor->shape().num_elements(), num_elements - offset_cumm);
      if (offset_cumm + n_elems < offset_fusion){
        offset_cumm += n_elems;
        continue;
      }
      auto& buffer = bufferManager_.GetBuffer(entry.tensor_name, entry.device,
                                              entry.context->framework());
      float* error_fb_buf =
          (float*) const_cast<void*>(buffer->AccessData(entry.context)) +
          std::max(offset_fusion - offset_cumm, 0l);

      // e_fb := e_fb + v
      CUDA_add(n_elems, original_buffer + offset_cumm, error_fb_buf,
          stream);
      // e_fb := e_fb - D(v)
      CUDA_substract(n_elems, decompressed_buffer + offset_cumm, error_fb_buf,
                     stream);
      offset_cumm += n_elems;
    }
  }
}

} // namespace common
} // namespace horovod
