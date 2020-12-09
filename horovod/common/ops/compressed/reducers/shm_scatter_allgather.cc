#include "shm_scatter_allgather.h"
#include "../utils.h"
#include "p2p_comm.h"
#include "shm_utils.h"
#include <map>

namespace horovod {
namespace common {

SHM_Allreduce_ScatterReduceAllgather::SHM_Allreduce_ScatterReduceAllgather(
    MPIContext* mpi_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state, Compressor* compressor,
    CommunicatorType comm_type)
    : SHMReducer(mpi_context, gpu_context, global_state, compressor,
                 comm_type) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "SHM_SRA";
  }
  hcomm_.reset();
}

size_t SHM_Allreduce_ScatterReduceAllgather::GetRequiredFreeSize() {
  int world_size = global_state_->controller->GetSize();
  size_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  return chunk_size * (world_size - 1) + chunk_size * (world_size - 1) +
         2 * chunk_size * world_size;
}

Status SHM_Allreduce_ScatterReduceAllgather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size =
      ALIGNED_SIZE((tensor_fusion_threshold_ + world_size - 1) / world_size);
  int64_t buffer_size = chunk_size * world_size + chunk_size * (world_size - 1);

  // Allocate buffers used for compression/decompression.
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

  // Initialize shared memory
  if (hcomm_ == nullptr) {
    int rank = global_state_->controller->GetRank();
    std::vector<int> ranks;
    for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
      if (peer_rank == rank)
        continue;
      ranks.push_back(peer_rank);
    }

    if (comm_type_ == CommunicatorType::SHM) {
      shmComm* sComm = new shmComm(rank);
      sComm->Init(comm, ranks, ranks, 2 * chunk_size);
      hcomm_.reset(sComm);
    } else if (comm_type_ == CommunicatorType::P2P) {
      p2pComm* pComm = new p2pComm(rank);
      pComm->Init(comm, ranks, ranks, gradients_recv_, chunk_size);
      hcomm_.reset(pComm);
    }
  }
  if (streams_.size() == 0) {
    for (int i = 0; i < world_size; i++) {
      streams_.push_back(cudaStream_t());
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&streams_.back(), cudaStreamNonBlocking));
      // No need to call Event Destroy in destructor as
      // CUDA driver must be deinitalized by the time destructor is called.
      events_.push_back(cudaEvent_t());
      CUDA_CHECK(
          cudaEventCreateWithFlags(&events_.back(), cudaEventDisableTiming));
    }
  }
  return Reducer::Init(entries);
}

Status SHM_Allreduce_ScatterReduceAllgather::AllreduceDivision(
    int num_elements, std::vector<TensorTableEntry>& entries,
    int64_t global_offset) {
  int world_size = global_state_->controller->GetSize();
  int rank = global_state_->controller->GetRank();
  std::vector<int> chunk_sizes, offsets;
  compressor_->GetSizesAndOffsets(num_elements, world_size, entries, offsets,
                                  chunk_sizes);
  int start_elem = offsets[rank];
  int recv_num_elems = chunk_sizes[rank];
  int recv_compressed_size = ALIGNED_SIZE(compressor_->BufferSize(
      recv_num_elems, entries, start_elem, global_offset));
  int send_num_elems = 0;
  void* peer_buf = nullptr;
  size_t compressed_size = 0;
  size_t total_compressed_size = 0;
  std::map<int, int> compressed_offsets;
  std::map<int, int> cumm_compressed_offsets;
  auto first_entry = entries[0];
  unsigned char* compressed_buf = gradients_send_;
  // First round of SRA.
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int start_offset = offsets[node_rank];
    send_num_elems = chunk_sizes[node_rank];

    compressed_size = ALIGNED_SIZE(compressor_->Compress(
        compressed_buf, entries, start_offset, global_offset,
        send_num_elems, true, false, &streams_[node_rank]));

    hcomm_->Send(compressed_buf, compressed_size, node_rank,
                 streams_[node_rank]);
    compressed_offsets[node_rank] = compressed_size;
    cumm_compressed_offsets[node_rank] = total_compressed_size;
    total_compressed_size += compressed_size;
    compressed_buf += compressed_size;
  }

  std::vector<int> indices;
  int node_rank;
  for (node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank != rank)
      indices.push_back(node_rank);
  }
  // Receive buffers
  while (indices.size() > 0) {
    for (auto it = indices.begin(); it != indices.end();) {
      node_rank = *it;
      if (hcomm_->RecvBufAsync((void**)&peer_buf, node_rank,
                               streams_[node_rank])) {
        it = indices.erase(it);
        int idx = node_rank - ((node_rank > rank) ? 1 : 0);
        CUDA_CHECK(cudaMemcpyAsync(gradients_recv_ + idx * recv_compressed_size,
                                   peer_buf, recv_compressed_size,
                                   cudaMemcpyDeviceToDevice,
                                   streams_[node_rank]));
        CUDA_CHECK(cudaEventRecord(events_[node_rank], streams_[node_rank]));
      } else {
        it++;
      }
    }
  }
  for (node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank != rank)
      indices.push_back(node_rank);
  }
  // Decompress received buffers.
  while (indices.size() > 0) {
    for (auto it = indices.begin(); it != indices.end();) {
      node_rank = *it;
      if (cudaEventQuery(events_[node_rank]) == cudaSuccess) {
        it = indices.erase(it);
        int idx = node_rank - ((node_rank > rank) ? 1 : 0);
        compressor_->Decompress(gradients_recv_ + idx * recv_compressed_size,
                                entries, start_elem, global_offset,
                                recv_num_elems, true, &streams_[rank]);
      } else {
        it++;
      }
    }
  }

  // Second round of SRA.
  compressor_->Compress(compressed_buf, entries, start_elem,
                        global_offset, recv_num_elems, false, true,
                        &streams_[rank]);
  CUDA_CHECK(cudaEventRecord(events_[rank], streams_[rank]));
  compressor_->Decompress(compressed_buf, entries, start_elem, global_offset,
                          recv_num_elems, false, &streams_[rank]);
  compressed_size = recv_compressed_size;
  hcomm_->WaitSendAll();
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    CUDA_CHECK(cudaStreamWaitEvent(streams_[node_rank], events_[rank], 0));
    hcomm_->Send(compressed_buf, compressed_size, node_rank,
                 streams_[node_rank], compressed_offsets[node_rank]);
  }

  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank != rank)
      indices.push_back(node_rank);
  }

  int recv_shm_offset = recv_compressed_size;
  while (indices.size() > 0) {
    for (auto it = indices.begin(); it != indices.end();) {

      int node_rank = *it;
      if (hcomm_->RecvBufAsync((void**)&peer_buf, node_rank,
                               streams_[node_rank], recv_shm_offset)) {
        it = indices.erase(it);
        int their_start_offset = offsets[node_rank];
        int their_recv_num_elems = chunk_sizes[node_rank];
        int compressed_offset = cumm_compressed_offsets[node_rank];
        compressed_size = compressed_offsets[node_rank];
        cudaMemcpyAsync(gradients_recv_ + compressed_offset, peer_buf,
                        compressed_size, cudaMemcpyDeviceToDevice,
                        streams_[node_rank]);
        compressor_->Decompress(
            gradients_recv_ + compressed_offset, entries, their_start_offset,
            global_offset, their_recv_num_elems, false, &streams_[node_rank]);
      } else {
        it++;
      }
    }
  }
  for (auto stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  hcomm_->WaitSendAll();
  return Status::OK();
}

} // namespace common
} // namespace horovod