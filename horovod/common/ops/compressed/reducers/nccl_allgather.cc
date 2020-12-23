#include "nccl_allgather.h"
#include "../utils.h"

namespace horovod {
namespace common {

NCCL_Allreduce_AllGather::NCCL_Allreduce_AllGather(
    NCCLContext* nccl_context, GPUContext* gpu_context,
    GPUOpContext* gpu_op_context, HorovodGlobalState* global_state,
    Compressor* compressor)
    : NCCLReducer(nccl_context, gpu_context, gpu_op_context, global_state,
                  compressor) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "NCCL_Allreduce_AllGather";
  }
}

size_t NCCL_Allreduce_AllGather::GetRequiredFreeSize() {
  int world_size = global_state_->controller->GetSize();
  size_t chunk_size = tensor_fusion_threshold_;
  return chunk_size * world_size + chunk_size;
}

Status NCCL_Allreduce_AllGather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size =
      std::max(entries[0].tensor->size(), tensor_fusion_threshold_);
  int64_t buffer_size = chunk_size * world_size + chunk_size;

  auto status = bufferManager_.InitializeBuffer(
      buffer_size, first_entry.device, first_entry.context,
      global_state_->current_nccl_stream,
      [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
      [&]() { timeline.ActivityEndAll(entries); });
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
  auto buffer = bufferManager_.GetBuffer(first_entry.device,
                                         first_entry.context->framework(),
                                         global_state_->current_nccl_stream);
  void* buffer_data =
      const_cast<void*>(buffer->AccessData(first_entry.context));
  gradients_recv_ = (unsigned char*)buffer_data;
  decompress_buffer_ = gradients_recv_ + chunk_size * world_size;
  stream_ =
      &gpu_context_
           ->streams[global_state_->current_nccl_stream][entries[0].device];
  return Reducer::Init(entries);
}

Status NCCL_Allreduce_AllGather::AllreduceDivision(
    int num_elements, ncclComm_t* nccl_comm,
    std::vector<horovod::common::TensorTableEntry>& entries,
    unsigned char* buffer_ptr) {
  int world_size = global_state_->controller->GetSize();
  int rank = global_state_->controller->GetRank();
  int64_t send_rcv_size =
      compressor_->BufferSize(num_elements, entries, 0);
  unsigned char* send_buf = gradients_recv_ + rank * send_rcv_size;
  compressor_->Compress(buffer_ptr, send_buf, entries, 0,
                        num_elements, false, stream_);
  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_COMPRESSION,
                              *stream_);
  }

  // Do allgather.
  auto nccl_result =
      ncclAllGather((void*)send_buf, (void*)gradients_recv_,
                    (size_t)send_rcv_size, ncclChar, *nccl_comm, *stream_);
  nccl_context_->ErrorCheck("ncclAllGather", nccl_result, *nccl_comm);
  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_NETWORK,
                              *stream_);
  }
  unsigned char* recv_buf = gradients_recv_;
  compressor_->Decompress(send_buf, buffer_ptr, entries, 0, num_elements,
                          false, stream_);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      recv_buf += send_rcv_size;
      continue;
    }
    compressor_->Decompress(recv_buf, buffer_ptr, entries, 0, num_elements,
                            true, stream_);
    recv_buf += send_rcv_size;
  }
  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_->event_queue, Q_DECOMPRESSION,
                              *stream_);
  }
  return Status::OK();
}

} // namespace common
} // namespace horovod
