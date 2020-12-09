#include "mpi_scatter_allgather.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_Allreduce_ScatterReduceAllgather::MPI_Allreduce_ScatterReduceAllgather(
    MPIContext* mpi_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state, Compressor* compressor)
    : MPIReducer(mpi_context, gpu_context, global_state, compressor) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "ScatterAllgather";
  }
}

size_t MPI_Allreduce_ScatterReduceAllgather::GetRequiredFreeSize() {
  int world_size = global_state_->controller->GetSize();
  size_t chunk_size =
      ALIGNED_SIZE((tensor_fusion_threshold_ + world_size - 1) / world_size);
  return chunk_size * (world_size - 1) +
                        + chunk_size * (world_size - 1) + chunk_size;
}

Status MPI_Allreduce_ScatterReduceAllgather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries,
    MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size =
      ALIGNED_SIZE((tensor_fusion_threshold_ + world_size - 1) / world_size);
  int64_t buffer_size = chunk_size * (world_size - 1) +
                        +chunk_size * (world_size - 1) + chunk_size;

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

Status MPI_Allreduce_ScatterReduceAllgather::AllreduceDivision(
    int num_elements, std::vector<TensorTableEntry>& entries,
    int64_t global_offset) {

  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();
  std::vector<int> chunk_sizes, offsets;
  compressor_->GetSizesAndOffsets(num_elements, world_size, entries, offsets,
                                  chunk_sizes);
  int start_elem = offsets[rank];
  int recv_num_elems = chunk_sizes[rank];
  int recv_compressed_size = ALIGNED_SIZE(compressor_->BufferSize(
      recv_num_elems, entries, start_elem, global_offset));
  int send_num_elems = 0;
  int send_compressed_size = 0;
  gpuStream_t stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];
  auto& timeline = global_state_->timeline;
  unsigned char* send_buf = gradients_send_;
  unsigned char* recv_buf = gradients_recv_;
  std::vector<MPI_Request> send_requests;
  std::vector<MPI_Request> recv_requests;
  std::queue<int> send_sizes;
  timeline.ActivityStartAll(entries, Q_COMPRESSION);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int64_t start_offset = offsets[node_rank];
    send_num_elems = chunk_sizes[node_rank];
    send_compressed_size = ALIGNED_SIZE(compressor_->Compress(
        send_buf, entries, start_offset, global_offset,
        send_num_elems, true, false, &stream));
    send_buf += send_compressed_size;
    send_sizes.push(send_compressed_size);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  timeline.ActivityEndAll(entries);

  send_buf = gradients_send_;
  timeline.ActivityStartAll(entries, Q_NETWORK);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    recv_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm_, &recv_requests.back()));
    send_compressed_size = send_sizes.front();
    send_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend(send_buf, send_compressed_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm_, &send_requests.back()));

    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
    send_sizes.pop();
  }
  // TODO: handling errors!!!
  std::vector<int> idx_map;
  for (int i = 0; i < world_size - 1; i++) {
    idx_map.push_back(i);
  }

  while (recv_requests.size() > 0) {
    int req_idx;
    MPI_CHECK(MPI_Waitany((int)recv_requests.size(), recv_requests.data(),
                          &req_idx, MPI_STATUSES_IGNORE));
    int idx = idx_map[req_idx];
    recv_requests.erase(recv_requests.begin() + req_idx);
    idx_map.erase(idx_map.begin() + req_idx);
    compressor_->Decompress(gradients_recv_ + idx * recv_compressed_size,
                            entries, start_elem, global_offset, recv_num_elems,
                            true, &stream);
  }
  MPI_CHECK(MPI_Waitall((int)send_requests.size(), send_requests.data(),
                        MPI_STATUSES_IGNORE));
  send_requests.clear();
  timeline.ActivityEndAll(entries);
  // End of the first round.

  compressor_->Compress(gradients_send_, entries, start_elem,
                        global_offset, recv_num_elems, false, true, &stream);
  cudaStreamSynchronize(stream);
  compressor_->Decompress(gradients_send_, entries, start_elem, global_offset,
                          recv_num_elems, false, &stream);
  recv_buf = gradients_recv_;

  timeline.ActivityStartAll(entries, Q_NETWORK);
  // second round of MPI communication. receive the sums from other nodes
  send_compressed_size = recv_compressed_size;
  std::vector<std::tuple<int64_t, int, int>> recv_offsets;
  int64_t recv_acc_size = 0;
  recv_requests.clear();
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int their_start_offset = offsets[node_rank];
    recv_num_elems = chunk_sizes[node_rank];
    recv_compressed_size = ALIGNED_SIZE(compressor_->BufferSize(
        recv_num_elems, entries, their_start_offset, global_offset));

    recv_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm_, &recv_requests.back()));
    send_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend(gradients_send_, send_compressed_size,
                        MPI_UNSIGNED_CHAR, node_rank, 0, comm_,
                        &send_requests.back()));
    recv_buf += recv_compressed_size;
    recv_offsets.emplace_back(recv_acc_size, their_start_offset,
                              recv_num_elems);
    recv_acc_size += recv_compressed_size;
  }
  while (recv_requests.size() > 0) {
    int req_idx;
    int their_start_offset;
    MPI_CHECK(MPI_Waitany((int)recv_requests.size(), recv_requests.data(),
                          &req_idx, MPI_STATUSES_IGNORE));

    std::tie(recv_acc_size, their_start_offset, recv_num_elems) =
        recv_offsets[req_idx];
    recv_requests.erase(recv_requests.begin() + req_idx);
    recv_offsets.erase(recv_offsets.begin() + req_idx);
    compressor_->Decompress(gradients_recv_ + recv_acc_size, entries,
                            their_start_offset, global_offset, recv_num_elems,
                            false, &stream);
  }
  timeline.ActivityEndAll(entries);
  MPI_CHECK(MPI_Waitall((int)send_requests.size(), send_requests.data(),
                        MPI_STATUSES_IGNORE));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::OK();
}

void printDebug(float* bf, int num_elems, int device, std::string prefix) {
  float* host_buf;
  std::stringstream ss;
  ss << prefix;
  if (device == CPU_DEVICE_ID) {
    host_buf = bf;
  } else {
    host_buf = new float[num_elems];
    CUDA_CHECK(cudaMemcpy(host_buf, bf, num_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  for (int i = 0; i < num_elems; i++) {
    ss << host_buf[i] << " ";
  }
  ss << std::endl;
  LOG(DEBUG) << ss.str();
  if (device != CPU_DEVICE_ID)
    delete[] host_buf;
}

} // namespace common
} // namespace horovod
