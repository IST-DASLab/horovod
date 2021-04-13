#include "shm_tree.h"
#include "../utils.h"
#include "shm_utils.h"

namespace horovod {
namespace common {
SHM_Allreduce_Tree::SHM_Allreduce_Tree(MPIContext* mpi_context,
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

size_t SHM_Allreduce_Tree::GetRequiredFreeSize() {
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = tensor_fusion_threshold_;
  int64_t buffer_size = 3 * chunk_size + chunk_size * world_size / 2;
  return buffer_size;
}

Status SHM_Allreduce_Tree::Init(const std::vector<TensorTableEntry>& entries,
                                MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  int world_size = global_state_->controller->GetSize();
  auto& timeline = global_state_->timeline;
  int64_t chunk_size = tensor_fusion_threshold_;
  int64_t buffer_size = chunk_size + chunk_size + chunk_size;
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
  decompress_buffer_ = gradients_recv_ + chunk_size;

  if (hcomm_ == nullptr) {
    int rank = global_state_->controller->GetRank();
    std::vector<int> ranks;
    int s = 1;
    while (s < world_size) {
      if (rank % (2 * s) == 0) {
        ranks.push_back(rank + s);
      } else {
        ranks.push_back(rank - s);
        break;
      }
      s *= 2;
    }
    if (comm_type_ == CommunicatorType::SHM) {
      shmComm* sComm = new shmComm(rank);
      sComm->Init(comm, ranks, ranks, chunk_size);
      hcomm_.reset(sComm);
    }
  }
  return Reducer::Init(entries);
}

Status SHM_Allreduce_Tree::AllreduceDivision(
    int num_elements, std::vector<TensorTableEntry>& entries,
    unsigned char* buffer_ptr, int global_offset) {
  const int world_size = global_state_->controller->GetSize();
  const int rank = global_state_->controller->GetRank();

  int shift = 1;
  int peer_rank;
  void* peer_buf;
  gpuStream_t stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];
  int64_t send_rcv_size =
      ALIGNED_SIZE(compressor_->BufferSize(num_elements, entries, 0));

  // First round of Tree algorithm. Bottom-up.
  while (shift < world_size) {
    shift *= 2;
    if (rank % shift == 0) {
      peer_rank = rank + shift / 2;
      hcomm_->RecvBuf(&peer_buf, peer_rank, stream, 0);
      CUDA_CHECK(cudaMemcpyAsync(gradients_recv_, peer_buf, send_rcv_size,
                                 cudaMemcpyDeviceToDevice, stream));
      compressor_->Decompress(gradients_recv_, buffer_ptr, entries, 0,
                              num_elements, true, &stream);
    } else {
      peer_rank = rank - shift / 2;
      compressor_->Compress(buffer_ptr, gradients_send_, entries, 0,
                            global_offset, num_elements, false, &stream);
      hcomm_->Send(gradients_send_, send_rcv_size, peer_rank, stream, 0);
      break;
    }
  }

  if (rank == 0) {
    compressor_->Compress(buffer_ptr, gradients_send_, entries, 0,
                          global_offset, num_elements, true, &stream);
  }
  // Second round. Top-down. Propagate reduced values.
  while (shift > 1) {
    if (rank % shift == 0) {
      shift /= 2;
      peer_rank = rank + shift;
      hcomm_->Send(gradients_send_, send_rcv_size, peer_rank, stream, 0);
    } else {
      shift /= 2;
      if (rank % shift == 0) {
        peer_rank = rank - shift;
        hcomm_->RecvBuf(&peer_buf, peer_rank, stream, 0);
        CUDA_CHECK(cudaMemcpyAsync(gradients_send_, peer_buf, send_rcv_size,
                                   cudaMemcpyDeviceToDevice, stream));
      }
    }
  }
  compressor_->Decompress(gradients_send_, buffer_ptr, entries, 0, num_elements,
                          false, &stream);
  hcomm_->WaitSendAll();
  global_state_->controller->Barrier(Communicator::GLOBAL);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::OK();
}

} // namespace common
} // namespace horovod
