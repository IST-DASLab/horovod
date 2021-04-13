#include "mpi_ps.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_Allreduce_PS::MPI_Allreduce_PS(MPIContext* mpi_context,
                                   GPUContext* gpu_context,
                                   HorovodGlobalState* global_state,
                                   Compressor* compressor)
    : MPIReducer(mpi_context, gpu_context, global_state, compressor) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "MPI Parameter-Server";
  }
}

size_t MPI_Allreduce_PS::GetRequiredFreeSize() {
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = tensor_fusion_threshold_;
  return chunk_size + chunk_size * (world_size - 1);
}

Status MPI_Allreduce_PS::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  int world_size = global_state_->controller->GetSize();
  auto& timeline = global_state_->timeline;
  int64_t chunk_size = tensor_fusion_threshold_;
  int64_t buffer_size = chunk_size + chunk_size * (world_size - 1);

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
  gradients_send_ = (unsigned char*)buffer_data;
  gradients_recv_ = (unsigned char*)gradients_send_ + chunk_size;
  return Reducer::Init(entries);
}

Status MPI_Allreduce_PS::AllreduceDivision(
    int num_elements, std::vector<horovod::common::TensorTableEntry>& entries,
    unsigned char* buffer_ptr, int global_offset) {
  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();
  auto& timeline = global_state_->timeline;
  gpuStream_t stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];

  timeline.ActivityStartAll(entries, Q_COMPRESSION);
  int64_t send_rcv_size = 0;
  int op;
  if (rank == 0) {
    std::vector<MPI_Request> requests;
    std::vector<int> idx_map;
    send_rcv_size =
        ALIGNED_SIZE(compressor_->BufferSize(num_elements, entries, 0));
    // First round.
    // Collect all gradients, decompress and aggregate them.
    for (int node_rank = 1; node_rank < world_size; node_rank++) {
      requests.push_back(MPI_Request());
      MPI_CHECK(MPI_Irecv(gradients_recv_ + (node_rank - 1) * send_rcv_size,
                          send_rcv_size, MPI_UNSIGNED_CHAR, node_rank, 0, comm_,
                          &requests.back()));
      idx_map.push_back(node_rank - 1);
    }

    while (requests.size() > 0) {
      int req_idx;
      MPI_Waitany((int)requests.size(), requests.data(), &req_idx,
                  MPI_STATUSES_IGNORE);
      int idx = idx_map[req_idx];
      requests.erase(requests.begin() + req_idx);
      idx_map.erase(idx_map.begin() + req_idx);
      compressor_->Decompress(gradients_recv_ + idx * send_rcv_size, buffer_ptr,
                              entries, 0, num_elements, true, &stream);
    }

    // Second round.
    // Broadcast the result
    compressor_->Compress(buffer_ptr, gradients_send_, entries, 0,
                          global_offset, num_elements, true, &stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    send_rcv_size = ALIGNED_SIZE(
        compressor_->Compress(buffer_ptr, gradients_send_, entries, 0,
                              global_offset, num_elements, false, &stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Send(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR, 0, 0,
                       comm_));
  }
  MPI_CHECK(
      MPI_Bcast(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR, 0, comm_));
  compressor_->Decompress(gradients_send_, buffer_ptr, entries, 0, num_elements,
                          false, &stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::OK();
}

} // namespace common
} // namespace horovod