#include "ring.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_GPUAllreduce_Ring::MPI_GPUAllreduce_Ring(MPIContext* mpi_context,
                                             GPUContext* gpu_context,
                                             HorovodGlobalState* global_state,
                                             Compressor* compressor)
    : MPIReducer(mpi_context, gpu_context, global_state, compressor) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "Ring";
  }
}

Status MPI_GPUAllreduce_Ring::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;

  int64_t allocated_compression_buffer_size_send = round_to(
      compressor_->BufferSize(chunk_size / sizeof(float)), ALIGNMENT_UNIT);
  int64_t allocated_compression_buffer_size_recv =
      allocated_compression_buffer_size_send;
  int64_t buffer_size = allocated_compression_buffer_size_send * world_size +
                        allocated_compression_buffer_size_recv + chunk_size;
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
  gradients_recv_ =
      gradients_send_ + allocated_compression_buffer_size_send * world_size;
  decompress_buffer_ = gradients_recv_ + allocated_compression_buffer_size_recv;
  status = compressor_->Init(entries);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }

  return Status::OK();
}

Status MPI_GPUAllreduce_Ring::AllreduceDivision(
    void* sendbuf, void* recvbuf, int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries,
    int64_t global_offset) {
  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();

  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;
  MPI_Request recv_req;
  MPI_Status recv_status;
  auto segment_size = [num_elems_per_node, residue](int segment) {
    return num_elems_per_node + ((segment < residue) ? 1 : 0);
  };
  std::vector<size_t> segment_ends(world_size);
  segment_ends[0] = segment_size(0);
  for (size_t i = 1; i < segment_ends.size(); ++i) {
    segment_ends[i] = segment_size(i) + segment_ends[i - 1];
  }
  float* send = (float*)sendbuf;
  float* recv = (float*)recvbuf;
  int recv_segment_idx, send_segment_idx;
  int64_t buf_send_idx, buf_recv_idx;
  int64_t send_size, recv_size;
  auto start = clock_::now();
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    buf_send_idx =
        (segment_ends[send_segment_idx] - segment_size(send_segment_idx));
    buf_recv_idx =
        (segment_ends[recv_segment_idx] - segment_size(recv_segment_idx));
    float* segment_send = send + buf_send_idx;
    float* segment_update = recv + buf_recv_idx;

    recv_size =
        round_to(compressor_->BufferSize(segment_size(recv_segment_idx),
                                         entries, buf_recv_idx, global_offset),
                 ALIGNMENT_UNIT);

    start = clock_::now();
    MPI_Irecv(gradients_recv_, recv_size, MPI_UNSIGNED_CHAR, recv_from, 0, comm,
              &recv_req);
    send_size = round_to(compressor_->Compress((unsigned char*)segment_send,
                                               gradients_send_, entries,
                                               buf_send_idx, global_offset,
                                               segment_size(send_segment_idx)),
                         ALIGNMENT_UNIT);

    MPI_Send(gradients_send_, send_size, MPI_UNSIGNED_CHAR, send_to, 0, comm);

    // Wait for recv to complete before reduction
    MPI_Wait(&recv_req, &recv_status);
    global_state_->communication_time += time_since(start);

    compressor_->Decompress(gradients_recv_, decompress_buffer_, entries,
                            buf_recv_idx, global_offset,
                            segment_size(recv_segment_idx));
    CUDA_add(segment_size(recv_segment_idx), (float*)decompress_buffer_,
             segment_update,
             gpu_context_->streams[global_state_->current_nccl_stream]
                                  [entries[0].device]);
  }
  send_segment_idx = (rank + world_size + 1) % world_size;
  buf_send_idx =
      (segment_ends[send_segment_idx] - segment_size(send_segment_idx));
  float* segment_send = recv + buf_send_idx;
  unsigned char* send_buf = gradients_send_;
  send_size =
      round_to(compressor_->Compress((unsigned char*)segment_send, send_buf,
                                     entries, buf_send_idx, global_offset,
                                     segment_size(send_segment_idx)),
               ALIGNMENT_UNIT);
  compressor_->Decompress(send_buf, (unsigned char*)segment_send, entries,
                          buf_send_idx, global_offset,
                          segment_size(send_segment_idx));

  unsigned char* recv_buf = send_buf + send_size;
  unsigned char* compressed_buf = recv_buf;
  // Propagate reduced and compressed chunks without decompression.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    // Segment to send - at every iteration we send segment (r+1-i)
    buf_recv_idx =
        (segment_ends[recv_segment_idx] - segment_size(recv_segment_idx));
    recv_size =
        round_to(compressor_->BufferSize(segment_size(recv_segment_idx),
                                         entries, buf_recv_idx, global_offset),
                 ALIGNMENT_UNIT);

    start = clock_::now();
    // Segment to recv - at every iteration we receive segment (r-i)
    MPI_Sendrecv(send_buf, send_size, MPI_UNSIGNED_CHAR, send_to, 0, recv_buf,
                 recv_size, MPI_UNSIGNED_CHAR, recv_from, 0, comm,
                 &recv_status);
    global_state_->communication_time += time_since(start);
    send_buf += send_size;
    recv_buf += recv_size;
    send_size = recv_size;
  }
  // Decompress all chunks we sent.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    float* segment_decompressed = recv + (segment_ends[recv_segment_idx] -
                                          segment_size(recv_segment_idx));
    buf_recv_idx =
        (segment_ends[recv_segment_idx] - segment_size(recv_segment_idx));

    recv_size =
        round_to(compressor_->BufferSize(segment_size(recv_segment_idx),
                                         entries, buf_recv_idx, global_offset),
                 ALIGNMENT_UNIT);

    compressor_->Decompress(
        compressed_buf, (unsigned char*)segment_decompressed, entries,
        buf_recv_idx, global_offset, segment_size(recv_segment_idx));
    compressed_buf += recv_size;
  }
  global_state_->compression_time = compressor_->getCompressionTime();
  global_state_->meta_info_time = compressor_->getMetaInfoTime();
  return Status::OK();
}

} // namespace common
} // namespace horovod