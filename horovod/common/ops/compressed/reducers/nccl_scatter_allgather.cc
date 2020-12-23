#include "nccl_scatter_allgather.h"
#include "../utils.h"

#ifdef NCCL_P2P_SUPPORTED

namespace horovod {
namespace common {

NCCL_Allreduce_ScatterAllgather::NCCL_Allreduce_ScatterAllgather(
    NCCLContext* nccl_context, GPUContext* gpu_context,
    GPUOpContext* gpu_op_context, HorovodGlobalState* global_state,
    Compressor* compressor)
    : NCCLReducer(nccl_context, gpu_context, gpu_op_context, global_state,
                  compressor) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "NCCL_Allreduce_ScatterAllgather";
  }
}

size_t NCCL_Allreduce_ScatterAllgather::GetRequiredFreeSize() {
  int world_size = global_state_->controller->GetSize();
  size_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  return chunk_size * (world_size - 1) + chunk_size * (world_size - 1) +
         chunk_size;
}

Status NCCL_Allreduce_ScatterAllgather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size =
    std::max(entries[0].tensor->size(), tensor_fusion_threshold_);
  chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  int64_t buffer_size = chunk_size * (world_size - 1) +
                        chunk_size * (world_size - 1) + chunk_size;
  stream_ =
      &gpu_context_
           ->streams[global_state_->current_nccl_stream][entries[0].device];

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
  gradients_recv_ = gradients_send_ + chunk_size * (world_size - 1);
  decompress_buffer_ = gradients_recv_ + chunk_size * (world_size - 1);

  return Reducer::Init(entries);
}

Status NCCL_Allreduce_ScatterAllgather::AllreduceDivision(
    int num_elements, ncclComm_t* comm, std::vector<TensorTableEntry>& entries,
    unsigned char* buffer_ptr) {

  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();
  std::vector<int> chunk_sizes, offsets;
  compressor_->GetSizesAndOffsets(num_elements, world_size, entries, offsets,
                                  chunk_sizes);
  int start_elem = offsets[rank];
  int recv_num_elems = chunk_sizes[rank];
  int recv_compressed_size =
      round_to(compressor_->BufferSize(recv_num_elems, entries, start_elem),
               ALIGNMENT_UNIT);
  int send_num_elems = 0;
  int send_compressed_size = 0;

  auto& timeline = global_state_->timeline;
  unsigned char* send_buf = gradients_send_;
  unsigned char* recv_buf = gradients_recv_;
  std::queue<int> send_sizes;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int start_offset = offsets[node_rank];
    send_num_elems = chunk_sizes[node_rank];
    send_compressed_size = ALIGNED_SIZE(
        compressor_->Compress(buffer_ptr, send_buf, entries, start_offset,
                              send_num_elems, false, stream_));
    send_buf += send_compressed_size;
    send_sizes.push(send_compressed_size);
  }
  if (timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_COMPRESSION,
                              *stream_);
  }
  send_buf = gradients_send_;
  NCCL_CALL_CHECK("ncclGroupStart", ncclGroupStart(), *comm);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    send_compressed_size = send_sizes.front();
    NCCL_CALL_CHECK("ncclRecv",
                    ncclRecv(recv_buf, recv_compressed_size, ncclChar,
                             node_rank, *comm, *stream_),
                    *comm);
    NCCL_CALL_CHECK("ncclSend",
                    ncclSend(send_buf, send_compressed_size, ncclChar,
                             node_rank, *comm, *stream_),
                    *comm);
    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
    send_sizes.pop();
  }
  NCCL_CALL_CHECK("ncclGroupEnd", ncclGroupEnd(), *comm);
  if (timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_NETWORK,
                              *stream_);
  }

  recv_buf = gradients_recv_;
  for (int i = 0; i < world_size - 1; i++) {
    compressor_->Decompress(recv_buf, buffer_ptr, entries, start_elem,
                            recv_num_elems, true, stream_);
    recv_buf += recv_compressed_size;
  }

  compressor_->Compress(buffer_ptr, gradients_send_, entries, start_elem,
                        recv_num_elems, true, stream_);
  compressor_->Decompress(gradients_send_, buffer_ptr, entries, start_elem,
                          recv_num_elems, false, stream_);
  if (timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_DECOMPRESSION,
                              *stream_);
  }

  recv_buf = gradients_recv_;
  // second round of communication. receive the sums from other nodes
  send_compressed_size = recv_compressed_size;
  NCCL_CALL_CHECK("ncclGroupStart", ncclGroupStart(), *comm);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int their_start_offset = offsets[node_rank];
    recv_num_elems = chunk_sizes[node_rank];
    recv_compressed_size = ALIGNED_SIZE(compressor_->BufferSize(
        recv_num_elems, entries, their_start_offset));
    NCCL_CALL_CHECK("ncclRecv",
                    ncclRecv(recv_buf, recv_compressed_size, ncclChar,
                             node_rank, *comm, *stream_),
                    *comm);
    NCCL_CALL_CHECK("ncclSend",
                    ncclSend(gradients_send_, send_compressed_size, ncclChar,
                             node_rank, *comm, *stream_),
                    *comm);
    recv_buf += recv_compressed_size;
  }
  NCCL_CALL_CHECK("ncclGroupEnd", ncclGroupEnd(), *comm);
  if (timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_NETWORK,
                              *stream_);
  }

  recv_buf = gradients_recv_;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    // Offset of the received chunk
    int their_start_offset = offsets[node_rank];
    recv_num_elems = chunk_sizes[node_rank];
    recv_compressed_size = ALIGNED_SIZE(compressor_->BufferSize(
        recv_num_elems, entries, their_start_offset));
    compressor_->Decompress(recv_buf, buffer_ptr, entries, their_start_offset,
                            recv_num_elems, false, stream_);
    recv_buf += recv_compressed_size;
  }
  if (timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_DECOMPRESSION,
                              *stream_);
  }

  return Status::OK();
}

} // namespace common
} // namespace horovod

#endif