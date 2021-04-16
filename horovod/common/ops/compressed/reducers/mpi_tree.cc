#include "mpi_tree.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_Allreduce_Tree::MPI_Allreduce_Tree(MPIContext* mpi_context,
                                       GPUContext* gpu_context,
                                       HorovodGlobalState* global_state,
                                       Compressor* compressor)
    : MPIReducer(mpi_context, gpu_context, global_state, compressor) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "MPI_Tree";
  }
}

Status MPI_Allreduce_Tree::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int64_t chunk_size = tensor_fusion_threshold_;
  int64_t buffer_size = chunk_size + chunk_size + chunk_size;
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
  decompress_buffer_ = gradients_recv_ + chunk_size;
  return Reducer::Init(entries);
}

size_t MPI_Allreduce_Tree::GetRequiredFreeSize() {
  return 3 * tensor_fusion_threshold_;
}

Status MPI_Allreduce_Tree::AllreduceDivision(
    int num_elements, std::vector<horovod::common::TensorTableEntry>& entries,
    unsigned char* buffer_ptr, int global_offset) {
  auto& first_entry = entries[0];

  const int world_size = global_state_->controller->GetSize();
  const int rank = global_state_->controller->GetRank();
  int shift = 1;
  int peer_rank;
  gpuStream_t stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];

  auto dtype = entries[0].tensor->dtype();
  int64_t send_rcv_size =
      ALIGNED_SIZE(compressor_->BufferSize(num_elements, entries, 0));
  // First round of Tree algorithm. Bottom-up.
  while (shift < world_size) {
    shift *= 2;
    if (rank % shift == 0) {
      peer_rank = rank + shift / 2;
      MPI_CHECK(MPI_Recv(gradients_recv_, send_rcv_size, MPI_UNSIGNED_CHAR,
                         peer_rank, 0, comm_, MPI_STATUSES_IGNORE));
      compressor_->Decompress(gradients_recv_, buffer_ptr, entries, 0,
                              num_elements, true, &stream);
    } else {
      peer_rank = rank - shift / 2;
      compressor_->Compress(buffer_ptr, gradients_send_, entries, 0,
                            global_offset, num_elements, false, &stream);
      gpu_context_->StreamSynchronize(stream);
      MPI_CHECK(MPI_Send(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR,
                         peer_rank, 0, comm_));
      break;
    }
  }

  if (rank == 0) {
    compressor_->Compress(buffer_ptr, gradients_send_, entries, 0,
                          global_offset, num_elements, false, &stream);
    gpu_context_->StreamSynchronize(stream);
  }
  // Second round. Top-down. Propagate reduced values.
  while (shift > 1) {
    if (rank % shift == 0) {
      shift /= 2;
      peer_rank = rank + shift;
      MPI_CHECK(MPI_Send(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR,
                         peer_rank, 0, comm_));
    } else {
      shift /= 2;
      if (rank % shift == 0) {
        peer_rank = rank - shift;
        MPI_CHECK(MPI_Recv(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR,
                           peer_rank, 0, comm_, MPI_STATUSES_IGNORE));
      }
    }
  }
  compressor_->Decompress(gradients_send_, buffer_ptr, entries, 0, num_elements,
                          false, &stream);
  gpu_context_->StreamSynchronize(stream);
  return Status::OK();
}

} // namespace common
} // namespace horovod
