#include "mpi_ring.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_Allreduce_Ring::MPI_Allreduce_Ring(MPIContext* mpi_context,
                                       GPUContext* gpu_context,
                                       HorovodGlobalState* global_state,
                                       Compressor* compressor)
    : MPIReducer(mpi_context, gpu_context, global_state, compressor) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "Ring";
  }
}

size_t MPI_Allreduce_Ring::GetRequiredFreeSize() {
  int world_size = global_state_->controller->GetSize();
  size_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  return chunk_size * world_size + chunk_size + chunk_size;
}

Status MPI_Allreduce_Ring::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = tensor_fusion_threshold_;
  chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  int64_t buffer_size = chunk_size * world_size + chunk_size + chunk_size;
  Status status = bufferManager_.InitializeBuffer(
      buffer_size, first_entry.device, first_entry.context,
      global_state_->current_nccl_stream,
      [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
      [&]() { timeline.ActivityEndAll(entries); });
  if (!status.ok()) {
    for (auto& e : entries) {
      timeline.End(e.tensor_name, nullptr);
      e.callback(status);
    }
    return status;
  }
  auto buffer = bufferManager_.GetBuffer(first_entry.device,
                                         first_entry.context->framework(),
                                         global_state_->current_nccl_stream);
  void* buffer_data =
      const_cast<void*>(buffer->AccessData(first_entry.context));
  gradients_send_ = (unsigned char*)buffer_data;
  gradients_recv_ = gradients_send_ + chunk_size * world_size;
  decompress_buffer_ = gradients_recv_ + chunk_size;
  return Reducer::Init(entries);
}

Status MPI_Allreduce_Ring::AllreduceDivision(
    int num_elements, std::vector<horovod::common::TensorTableEntry>& entries,
    unsigned char* buffer_ptr, int global_offset) {
  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();
  std::vector<int> chunk_sizes, offsets;
  compressor_->GetSizesAndOffsets(num_elements, world_size, entries, offsets,
                                  chunk_sizes);
  gpuStream_t stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];

  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;
  MPI_Request recv_req;

  int recv_segment_idx, send_segment_idx;
  int buf_send_idx, buf_recv_idx;
  int send_size, recv_size;
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    buf_send_idx = offsets[send_segment_idx];
    buf_recv_idx = offsets[recv_segment_idx];

    recv_size = ALIGNED_SIZE(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], entries, buf_recv_idx));

    MPI_CHECK(MPI_Irecv(gradients_recv_, recv_size, MPI_UNSIGNED_CHAR,
                        recv_from, 0, comm_, &recv_req));
    send_size = ALIGNED_SIZE(compressor_->Compress(
        buffer_ptr, gradients_send_, entries, buf_send_idx, global_offset,
        chunk_sizes[send_segment_idx], false, &stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Send(gradients_send_, send_size, MPI_UNSIGNED_CHAR, send_to,
                       0, comm_));

    // Wait for recv to complete before reduction
    MPI_CHECK(MPI_Wait(&recv_req, MPI_STATUSES_IGNORE));

    compressor_->Decompress(gradients_recv_, buffer_ptr, entries, buf_recv_idx,
                            chunk_sizes[recv_segment_idx], true, &stream);
  }

  send_segment_idx = (rank + world_size + 1) % world_size;
  buf_send_idx = offsets[send_segment_idx];
  unsigned char* send_buf = gradients_send_;
  send_size = ALIGNED_SIZE(compressor_->Compress(
      buffer_ptr, send_buf, entries, buf_send_idx, global_offset,
      chunk_sizes[send_segment_idx], true, (void*)&stream));
  compressor_->Decompress(send_buf, buffer_ptr, entries, buf_send_idx,
                          chunk_sizes[send_segment_idx], false, (void*)&stream);
  unsigned char* recv_buf = send_buf + send_size;
  unsigned char* compressed_buf = recv_buf;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // Propagate reduced and compressed chunks without decompression.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    // Segment to send - at every iteration we send segment (r+1-i)
    buf_recv_idx = offsets[recv_segment_idx];
    recv_size = ALIGNED_SIZE(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], entries, buf_recv_idx));

    // Segment to recv - at every iteration we receive segment (r-i)
    MPI_CHECK(MPI_Sendrecv(send_buf, send_size, MPI_UNSIGNED_CHAR, send_to, 0,
                           recv_buf, recv_size, MPI_UNSIGNED_CHAR, recv_from, 0,
                           comm_, MPI_STATUSES_IGNORE));
    send_buf += send_size;
    recv_buf += recv_size;
    send_size = recv_size;
  }

  // Decompress all chunks we received.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    buf_recv_idx = offsets[recv_segment_idx];

    compressor_->Decompress(compressed_buf, buffer_ptr, entries, buf_recv_idx,
                            chunk_sizes[recv_segment_idx], false, &stream);
    recv_size = ALIGNED_SIZE(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], entries, buf_recv_idx));
    compressed_buf += recv_size;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::OK();
}

} // namespace common
} // namespace horovod