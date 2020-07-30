#include "scatter_allgather.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_Allreduce_ScatterReduceAllgather::MPI_Allreduce_ScatterReduceAllgather(
    MPIContext* mpi_context, HorovodGlobalState* global_state,
    Compressor* compressor, Summator* summator)
    : MPIReducer(mpi_context, global_state, compressor, summator) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "ScatterAllgather";
  }
}

Status MPI_Allreduce_ScatterReduceAllgather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;
  auto dtype = entries[0].tensor->dtype();
  int64_t allocated_compression_buffer_size_send = round_to(
      compressor_->BufferSize(chunk_size / get_sizeof(dtype), dtype), ALIGNMENT_UNIT);
  int64_t allocated_compression_buffer_size_recv =
      allocated_compression_buffer_size_send;
  int64_t buffer_size =
      allocated_compression_buffer_size_send * (world_size - 1) +
      +allocated_compression_buffer_size_recv * (world_size - 1) + chunk_size;

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

Status MPI_Allreduce_ScatterReduceAllgather::AllreduceDivision(
    int num_elements, MPI_Comm comm, std::vector<TensorTableEntry>& entries,
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
  std::vector<MPI_Request> send_requests;
  std::vector<MPI_Request> recv_requests;
  std::queue<int> send_sizes;
  timeline.ActivityStartAll(entries, Q_COMPRESSION);
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
  compressor_->Finalize();
  timeline.ActivityEndAll(entries);

  send_buf = gradients_send_;
  auto start = clock_::now();
  timeline.ActivityStartAll(entries, Q_NETWORK);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    recv_requests.push_back(MPI_Request());
    MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              comm, &recv_requests.back());
    send_compressed_size = send_sizes.front();
    send_requests.push_back(MPI_Request());
    MPI_Isend(send_buf, send_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              comm, &send_requests.back());

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
    MPI_Waitany((int)recv_requests.size(), recv_requests.data(), &req_idx,
                MPI_STATUSES_IGNORE);
    int idx = idx_map[req_idx];
    recv_requests.erase(recv_requests.begin() + req_idx);
    idx_map.erase(idx_map.begin() + req_idx);
    compressor_->Decompress(gradients_recv_ + idx * recv_compressed_size,
                            entries, start_elem, global_offset, recv_num_elems,
                            true);
  }
  //  MPI_Waitall((int)requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  timeline.ActivityEndAll(entries);
  global_state_->communication_time += time_since(start);

  //  recv_buf = gradients_recv_;
  //  for (int i = 0; i < world_size - 1; i++) {
  //    compressor_->Decompress(recv_buf, entries, start_elem, global_offset,
  //                            recv_num_elems, true);
  //    recv_buf += recv_compressed_size;
  //  }

  // Quantize the sum into gradients_recv_[0] and maxandmin_recv[0]
  compressor_->Compress(gradients_send_, entries, error_feedback_, start_elem,
                        global_offset, recv_num_elems, false, true);
  compressor_->Decompress(gradients_send_, entries, start_elem, global_offset,
                          recv_num_elems, false);
  compressor_->Finalize();
  recv_buf = gradients_recv_;

  timeline.ActivityStartAll(entries, Q_NETWORK);
  // second round of MPI communication. receive the sums from other nodes
  send_compressed_size = recv_compressed_size;
  start = clock_::now();
  std::vector<std::pair<int64_t, int>> recv_offsets;
  int64_t recv_acc_size = 0;
  recv_requests.clear();
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

    recv_requests.push_back(MPI_Request());
    MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              comm, &recv_requests.back());
    send_requests.push_back(MPI_Request());
    MPI_Isend(gradients_send_, send_compressed_size, MPI_UNSIGNED_CHAR,
              node_rank, 0, comm, &send_requests.back());
    recv_buf += recv_compressed_size;
    recv_offsets.emplace_back(recv_acc_size, their_start_offset);
    recv_acc_size += recv_compressed_size;
  }
  while (recv_requests.size() > 0) {
    int req_idx;
    int their_start_offset;
    MPI_Waitany((int)recv_requests.size(), recv_requests.data(), &req_idx,
                MPI_STATUSES_IGNORE);

    std::tie(recv_acc_size, their_start_offset) = recv_offsets[req_idx];
    recv_requests.erase(recv_requests.begin() + req_idx);
    recv_offsets.erase(recv_offsets.begin() + req_idx);
    compressor_->Decompress(gradients_recv_ + recv_acc_size, entries,
                            their_start_offset, global_offset, recv_num_elems,
                            false);
  }
  //  MPI_Waitall((int)requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  timeline.ActivityEndAll(entries);
  global_state_->communication_time += time_since(start);
  //  timeline.ActivityStartAll(entries, Q_DECOMPRESSION);
  //  // dequantization
  //  recv_buf = gradients_recv_;
  //  for (int node_rank = 0; node_rank < world_size; node_rank++) {
  //    if (node_rank == rank) {
  //      continue;
  //    }
  //    // Offset of the received chunk
  //    int their_start_offset =
  //        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
  //    recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
  //    recv_compressed_size =
  //        round_to(compressor_->BufferSize(recv_num_elems, entries,
  //                                         their_start_offset, global_offset),
  //                 ALIGNMENT_UNIT);
  //    compressor_->Decompress(recv_buf, entries, their_start_offset,
  //                            global_offset, recv_num_elems, false);
  //    recv_buf += recv_compressed_size;
  //  }
  compressor_->Finalize();
  MPI_Waitall((int)send_requests.size(), send_requests.data(),
              MPI_STATUSES_IGNORE);
  //  timeline.ActivityEndAll(entries);
  global_state_->compression_time = compressor_->getCompressionTime();
  global_state_->meta_info_time = compressor_->getMetaInfoTime();
  return Status::OK();
}

void printDebug(float* bf, int num_elems, int device) {
  float* host_buf;
  if (device == CPU_DEVICE_ID) {
    host_buf = bf;
  } else {
    host_buf = new float[num_elems];
    cudaMemcpy(host_buf, bf, num_elems * sizeof(float), cudaMemcpyDeviceToHost);
  }
  for (int i = 0; i < num_elems; i++) {
    std::cout << host_buf[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::flush;
  if (device != CPU_DEVICE_ID)
    delete[] host_buf;
}

} // namespace common
} // namespace horovod
