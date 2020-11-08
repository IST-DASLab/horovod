#include "shm_scatter_allgather.h"
#include "../utils.h"
#include <map>
#include "shm_utils.h"
#include "p2p_comm.h"

namespace horovod {
namespace common {

SHM_Allreduce_ScatterReduceAllgather::SHM_Allreduce_ScatterReduceAllgather(
    MPIContext* mpi_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state, Compressor* compressor,
    Summator* summator)
    : SHMReducer(mpi_context, gpu_context, global_state, compressor, summator) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "SHM_SRA";
  }
  hcomm_.reset();
}

Status SHM_Allreduce_ScatterReduceAllgather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  auto dtype = entries[0].tensor->dtype();
  int64_t allocated_compression_buffer_size_send = ALIGNED_SIZE(
      compressor_->BufferSize(chunk_size / get_sizeof(dtype), dtype));
  int64_t allocated_compression_buffer_size_recv =
      allocated_compression_buffer_size_send;
  int64_t buffer_size =
      allocated_compression_buffer_size_send * world_size +
      +allocated_compression_buffer_size_recv * (world_size - 1);

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
  gradients_recv_ =
      gradients_send_ + allocated_compression_buffer_size_send * world_size;

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

  // Initialize shared memory
  if (hcomm_ == nullptr) {
    std::cout << "Initialize communicator" << std::endl;
    auto reduction_type = GetEnumEnvOrDefault<ReductionType>(
        HOROVOD_REDUCTION, ReductionType::NoneReduction);

    int rank = global_state_->controller->GetRank();
    std::vector<int> ranks;
    for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
      if (peer_rank == rank)
        continue;
      ranks.push_back(peer_rank);
    }

    if (reduction_type == ReductionType::SHM_ScatterAllgather) {
      shmComm* sComm = new shmComm(rank);
      sComm->Init(comm, ranks, 2 * allocated_compression_buffer_size_send);
      hcomm_.reset(sComm);
    } else if (reduction_type == ReductionType::P2P_ScatterAllgather) {
      p2pComm* pComm = new p2pComm(rank);
      pComm->Init(comm, ranks, gradients_recv_,
                  allocated_compression_buffer_size_recv);
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
}

Status SHM_Allreduce_ScatterReduceAllgather::AllreduceDivision(
    int num_elements, std::vector<horovod::common::TensorTableEntry>& entries,
    int64_t global_offset) {
  int world_size = global_state_->controller->GetSize();
  int rank = global_state_->controller->GetRank();
  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
  int start_elem = num_elems_per_node * rank + std::min(residue, rank);
  int recv_num_elems = num_elems_per_node + (rank < residue ? 1 : 0);
  int recv_compressed_size = ALIGNED_SIZE(compressor_->BufferSize(
      recv_num_elems, entries, start_elem, global_offset));
  int send_num_elems = 0;
  float* peer_buf = nullptr;
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
    int start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    send_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);

    compressed_size = ALIGNED_SIZE(compressor_->Compress(
        compressed_buf, entries, error_feedback_, start_offset, global_offset,
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
  compressor_->Compress(compressed_buf, entries, error_feedback_, start_elem,
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
        int their_start_offset =
            (num_elems_per_node * node_rank) + std::min(residue, node_rank);
        int their_recv_num_elems =
            num_elems_per_node + ((node_rank < residue) ? 1 : 0);
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
    cudaStreamSynchronize(stream);
  }
  hcomm_->WaitSendAll();
  return Status::OK();
}

} // namespace common
} // namespace horovod