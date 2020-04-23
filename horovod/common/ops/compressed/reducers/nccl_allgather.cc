#include "nccl_allgather.h"

namespace horovod {
namespace common {

NCCL_Allreduce_AllGather::NCCL_Allreduce_AllGather(
    NCCLContext* nccl_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state, Compressor* compressor,
    Summator* summator)
    : NCCLReducer(nccl_context, gpu_context, global_state, compressor, summator) {
  if (global_state_->controller->GetRank() == 0) {
    LOG(INFO) << "NCCL_Allreduce_AllGather";
  }
}

Status NCCL_Allreduce_AllGather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = tensor_fusion_threshold_;
  int64_t allocated_compression_buffer_size_send =
      compressor_->BufferSize(chunk_size / sizeof(float));
  int64_t buffer_size =
      allocated_compression_buffer_size_send * world_size + chunk_size;

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
  decompress_buffer_ =
      gradients_recv_ + allocated_compression_buffer_size_send * world_size;
  stream_ =
      &gpu_context_
           ->streams[global_state_->current_nccl_stream][entries[0].device];

  status = compressor_->Init(entries);
  if (!status.ok()) {
    return status;
  }
  status = error_feedback_.Init(entries);
  if (!status.ok()) {
    return status;
  }
}


Status NCCL_Allreduce_AllGather::AllreduceDivision(
    int num_elements, ncclComm_t* nccl_comm,
    std::vector<horovod::common::TensorTableEntry>& entries,
    int64_t global_offset) {
  int world_size = global_state_->controller->GetSize();
  int rank = global_state_->controller->GetRank();
  int64_t send_rcv_size =
      compressor_->BufferSize(num_elements, entries, 0, global_offset);
  unsigned char* send_buf = gradients_recv_ + rank * send_rcv_size;
  compressor_->Compress(send_buf, entries, error_feedback_, 0, global_offset,
                        num_elements);
  // Do allreduce.
  auto nccl_result =
      ncclAllGather((void*)send_buf, (void*)gradients_recv_,
                    (size_t)send_rcv_size, ncclChar, *nccl_comm, *stream_);
  nccl_context_->ErrorCheck("ncclAllGather", nccl_result);
  unsigned char* recv_buf = gradients_recv_;
  compressor_->Decompress(send_buf, entries, 0, global_offset, num_elements);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank){
      recv_buf += send_rcv_size;
      continue;
    }
    compressor_->Decompress(recv_buf, decompress_buffer_, entries, 0,
                            num_elements);
    summator_->Add((float*)decompress_buffer_, entries, 0, global_offset,
                   num_elements, false);
    recv_buf += send_rcv_size;
  }
  return Status::OK();
}

} // namespace common
} // namespace horovod
