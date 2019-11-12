#ifndef HOROVOD_ERROR_FEEDBACK_H
#define HOROVOD_ERROR_FEEDBACK_H

#include "../global_state.h"
#include "compressor.h"
#include "feedback_buffer_manager.h"
namespace horovod {
namespace common {

class ErrorFeedback {
public:
  ErrorFeedback(HorovodGlobalState* global_state, struct CUDAContext* cuda_context);
  // Initializes feedback buffers and computes offsets of the entries in feedback buffer.
  Status Init(const std::vector<TensorTableEntry>& entries);
  // Get offsets of the tensors in fusion buffer.
  std::vector<int64_t> GetOffsets();

  // Apply saved error feedback
  // Args:
  // Entries passed to Reduction step.
  // Gradients of the entries
  // num_elements in buffer
  // global_offset of the buffer in case of one big tensor broken into divisions. So this is essentially an offset in feedback buffer.
  void ApplyErrorFeedback(const std::vector<TensorTableEntry>& entries,
      void* sendbuf, int64_t num_elements, int64_t global_offset);

  // Update error feedback.
  // Args:
  // entries - Entries passed to reduction step
  // original_buf - Buffer before compression.
  // compressed_buf - Buffer about to send.
  // num_elems- size of decompressed sendbuf.
  // offset_fusion(in elems) - offset of the sendbuf begin in the fusion buffer.
  // global_offset(in elems) of the buffer in case of one big tensor broken into divisions. So this is essentially an offset in feedback buffer.
  // compressor - compressor sendbuf was compressed with.
  void UpdateErrorFeedback(const std::vector<TensorTableEntry>& entries,
      const void* original_buf, void* compressed_buf,
      int64_t num_elements, int64_t offset_fussion, int64_t global_offset,
      Compressor* compressor);

private:
  HorovodGlobalState* global_state_;
  struct CUDAContext* cuda_context_;
  float* decompressed_buf = nullptr;
  FeedbackBufferManager bufferManager_;
  bool enabled = false;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_ERROR_FEEDBACK_H
