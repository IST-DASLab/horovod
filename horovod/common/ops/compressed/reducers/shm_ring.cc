#include "shm_ring.h"
#include "../utils.h"
#include "p2p_comm.h"
#include "shm_utils.h"

namespace horovod {
namespace common {

SHM_Allreduce_Ring::SHM_Allreduce_Ring(MPIContext* mpi_context,
                                       GPUContext* gpu_context,
                                       HorovodGlobalState* global_state,
                                       Compressor* compressor,
                                       CommunicatorType comm_type)
    : SHMReducer(mpi_context, gpu_context, global_state, compressor,
                 comm_type) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "SHM_Ring";
  }
  hcomm_.reset();
}

size_t SHM_Allreduce_Ring::GetRequiredFreeSize() {
  int world_size = global_state_->controller->GetSize();
  size_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  return 2 * chunk_size;
}

Status SHM_Allreduce_Ring::Init(const std::vector<TensorTableEntry>& entries,
                                MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size =
      std::max(entries[0].tensor->size(), tensor_fusion_threshold_);
  chunk_size = (chunk_size + world_size - 1) / world_size;
  int64_t buffer_size = chunk_size + chunk_size;
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
  gradients_recv_ = gradients_send_ + chunk_size;
  if (hcomm_ == nullptr) {
    int rank = global_state_->controller->GetRank();
    std::vector<int> send_ranks;
    std::vector<int> recv_ranks;
    recv_ranks.push_back((rank + world_size - 1) % world_size);
    send_ranks.push_back((rank + 1) % world_size);
    if (comm_type_ == CommunicatorType::SHM) {
      shmComm* sComm = new shmComm(rank);
      sComm->Init(comm, send_ranks, recv_ranks, (world_size - 1) * chunk_size);
      hcomm_.reset(sComm);
    } else {
      p2pComm* pComm = new p2pComm(rank);
      pComm->Init(comm, send_ranks, recv_ranks, gradients_recv_, chunk_size);
      hcomm_.reset(pComm);
    }
  }
  return Reducer::Init(entries);
}

Status
SHM_Allreduce_Ring::AllreduceDivision(int num_elements,
                                      std::vector<TensorTableEntry>& entries,
                                      unsigned char* buffer_ptr) {
  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();
  gpuStream_t stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];
  std::vector<int> chunk_sizes, offsets;
  compressor_->GetSizesAndOffsets(num_elements, world_size, entries, offsets,
                                  chunk_sizes);
  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;

  int recv_segment_idx, send_segment_idx;
  int buf_send_idx, buf_recv_idx;
  int send_size, recv_size;
  void* peer_buf = nullptr;
  int agg_send_offset = 0;
  int agg_recv_offset = 0;
  // First round of Ring algorithm.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    buf_send_idx = offsets[send_segment_idx];
    buf_recv_idx = offsets[recv_segment_idx];

    recv_size = ALIGNED_SIZE(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], entries, buf_recv_idx));
    send_size = ALIGNED_SIZE(compressor_->Compress(
        buffer_ptr, gradients_send_, entries, buf_send_idx,
        chunk_sizes[send_segment_idx], false, &stream));

    hcomm_->Send(gradients_send_, send_size, send_to, stream, agg_send_offset);
    agg_send_offset += send_size;
    hcomm_->RecvBuf(&peer_buf, recv_from, stream, agg_recv_offset);
    agg_recv_offset += recv_size;
    CUDA_CHECK(cudaMemcpyAsync(gradients_recv_, peer_buf, recv_size,
                               cudaMemcpyDeviceToDevice, stream));
    compressor_->Decompress(gradients_recv_, buffer_ptr, entries, buf_recv_idx,
                            chunk_sizes[recv_segment_idx], true, &stream);
  }
  hcomm_->WaitSendAll();
  send_segment_idx = (rank + world_size + 1) % world_size;
  buf_send_idx = offsets[send_segment_idx];
  unsigned char* send_buf = gradients_send_;
  send_size = ALIGNED_SIZE(
      compressor_->Compress(buffer_ptr, send_buf, entries, buf_send_idx,
                            chunk_sizes[send_segment_idx], true, &stream));
  compressor_->Decompress(send_buf, buffer_ptr, entries, buf_send_idx,
                          chunk_sizes[send_segment_idx], false, &stream);
  agg_send_offset = 0;
  agg_recv_offset = 0;
  // Second round
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    // Segment to send - at every iteration we send segment (r+1-i)
    buf_recv_idx = offsets[recv_segment_idx];
    recv_size = ALIGNED_SIZE(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], entries, buf_recv_idx));
    hcomm_->Send(gradients_send_, send_size, send_to, stream, agg_send_offset);
    agg_send_offset += send_size;
    hcomm_->RecvBuf(&peer_buf, recv_from, stream, agg_recv_offset);
    agg_recv_offset += recv_size;
    CUDA_CHECK(cudaMemcpyAsync(gradients_send_, peer_buf, recv_size,
                               cudaMemcpyDeviceToDevice, stream));
    compressor_->Decompress(gradients_send_, buffer_ptr, entries, buf_recv_idx,
                            chunk_sizes[recv_segment_idx], false, &stream);
    send_size = recv_size;
  }
  hcomm_->WaitSendAll();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::OK();
}

} // namespace common
} // namespace horovod
