#include "nccl_scatter_allgather.h"

#if NCCL_VERSION_CHECK(2, 7, 0)

namespace horovod {
namespace common {

NCCL_Allreduce_ScatterAllgather::NCCL_Allreduce_ScatterAllgather(
    NCCLContext* nccl_context, GPUContext* gpu_context,
    GPUOpContext* gpu_op_context, HorovodGlobalState* global_state,
    Compressor* compressor, Summator* summator)
    : NCCLReducer(nccl_context, gpu_context, gpu_op_context, global_state,
                  compressor, summator) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "NCCL_Allreduce_ScatterAllgather";
  }
}

Status NCCL_Allreduce_ScatterAllgather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  auto dtype = entries[0].tensor->dtype();
  int64_t allocated_compression_buffer_size_send =
      round_to(compressor_->BufferSize(chunk_size / get_sizeof(dtype), dtype),
               ALIGNMENT_UNIT);
  int64_t allocated_compression_buffer_size_recv =
      allocated_compression_buffer_size_send;
  int64_t buffer_size =
      allocated_compression_buffer_size_send * (world_size - 1) +
      +allocated_compression_buffer_size_recv * (world_size - 1) + chunk_size;

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
  gradients_recv_ = gradients_send_ +
                    allocated_compression_buffer_size_send * (world_size - 1);
  decompress_buffer_ =
      gradients_recv_ +
      allocated_compression_buffer_size_recv * (world_size - 1);

  status = compressor_->Init(entries);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
  status = error_feedback_.Init(entries);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
  return Status::OK();
}

Status NCCL_Allreduce_ScatterAllgather::AllreduceDivision(
    int num_elements, ncclComm_t* comm, std::vector<TensorTableEntry>& entries,
    int64_t global_offset) {

  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();
  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
  int start_elem = num_elems_per_node * rank + std::min(residue, rank);
  int recv_num_elems = num_elems_per_node + (rank < residue ? 1 : 0);
  int recv_compressed_size =
      round_to(compressor_->BufferSize(recv_num_elems, entries, start_elem,
                                       global_offset),
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
    int start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    send_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    send_compressed_size = round_to(
        compressor_->Compress(send_buf, entries, error_feedback_, start_offset,
                              global_offset, send_num_elems),
        ALIGNMENT_UNIT);
    send_buf += send_compressed_size;
    send_sizes.push(send_compressed_size);
  }
  if (timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_COMPRESSION,
                              *stream_);
  }

  send_buf = gradients_send_;
  NCCL_CALL_CHECK("ncclGroupStart", ncclGroupStart());
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    send_compressed_size = send_sizes.front();
    NCCL_CALL_CHECK("ncclRecv",
                    ncclRecv(recv_buf, recv_compressed_size, ncclChar, node_rank, *comm, *stream_));
    NCCL_CALL_CHECK("ncclSend",
                    ncclSend(send_buf, send_compressed_size, ncclChar, node_rank, *comm, *stream_));
    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
    send_sizes.pop();
  }
  NCCL_CALL_CHECK("ncclGroupEnd", ncclGroupEnd());
  if (timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_NETWORK,
                              *stream_);
  }

  recv_buf = gradients_recv_;
  for (int i = 0; i < world_size - 1; i++) {
    compressor_->Decompress(recv_buf, entries, start_elem, global_offset,
                            recv_num_elems, true);
    recv_buf += recv_compressed_size;
  }

  compressor_->Compress(gradients_send_, entries, error_feedback_, start_elem,
                        global_offset, recv_num_elems, false, true);
  compressor_->Decompress(gradients_send_, entries, start_elem, global_offset,
                          recv_num_elems, false);
  if (timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_DECOMPRESSION,
                              *stream_);
  }

  recv_buf = gradients_recv_;
  // second round of MPI communication. receive the sums from other nodes
  send_compressed_size = recv_compressed_size;
  NCCL_CALL_CHECK("ncclGroupStart", ncclGroupStart());
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int their_start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    recv_compressed_size =
        round_to(compressor_->BufferSize(recv_num_elems, entries,
                                         their_start_offset, global_offset),
                 ALIGNMENT_UNIT);
    NCCL_CALL_CHECK("ncclRecv", ncclRecv(recv_buf, recv_compressed_size, ncclChar, node_rank, *comm, *stream_));
    NCCL_CALL_CHECK("ncclSend", ncclSend(gradients_send_, send_compressed_size, ncclChar, node_rank, *comm, *stream_));
    recv_buf += recv_compressed_size;
  }
  NCCL_CALL_CHECK("ncclGroupEnd", ncclGroupEnd());
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
    int their_start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    recv_compressed_size =
        round_to(compressor_->BufferSize(recv_num_elems, entries,
                                         their_start_offset, global_offset),
                 ALIGNMENT_UNIT);
    compressor_->Decompress(recv_buf, entries, their_start_offset,
                            global_offset, recv_num_elems, false);
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