#include "scatter_allgather.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_GPUAllreduce_ScatterReduceAllgather::
    MPI_GPUAllreduce_ScatterReduceAllgather(MPIContext* mpi_context,
                                            GPUContext* gpu_context,
                                            HorovodGlobalState* global_state,
                                            Compressor* compressor)
    : MPIReducer(mpi_context, gpu_context, global_state, compressor) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "ScatterAllgather";
  }
}

Status MPI_GPUAllreduce_ScatterReduceAllgather::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int world_size = global_state_->controller->GetSize();
  int64_t chunk_size = (tensor_fusion_threshold_ + world_size - 1) / world_size;

  int64_t allocated_compression_buffer_size_send = round_to(
      compressor_->BufferSize(chunk_size / sizeof(float)), ALIGNMENT_UNIT);
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
  status = errorFeedbackManager_.Init(entries);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }

  return Status::OK();
}

Status MPI_GPUAllreduce_ScatterReduceAllgather::AllreduceDivision(
    void* input, void* output, int num_elements, MPI_Comm comm,
    std::vector<TensorTableEntry>& entries, int64_t global_offset) {
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

  errorFeedbackManager_.ApplyErrorFeedback(entries, input, num_elements,
                                           global_offset);
  unsigned char* send_buf = gradients_send_;
  unsigned char* recv_buf = gradients_recv_;
  std::vector<MPI_Request> requests;
  unsigned char* input_buf;

  auto start = clock_::now();
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    requests.push_back(MPI_Request());
    MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              comm, &requests.back());
    int start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    input_buf = (unsigned char*)input + start_offset * sizeof(float);
    send_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);

    send_compressed_size = round_to(
        compressor_->Compress(input_buf, send_buf, entries, start_offset,
                              global_offset, send_num_elems),
        ALIGNMENT_UNIT);

    errorFeedbackManager_.UpdateErrorFeedback(entries, input_buf, send_buf,
                                              send_num_elems, start_offset,
                                              global_offset, compressor_);
    requests.push_back(MPI_Request());
    MPI_Isend(send_buf, send_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              comm, &requests.back());

    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
  }
  // TODO: handling errors!!!
  MPI_Waitall((int)requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  global_state_->communication_time += time_since(start);

  recv_buf = gradients_recv_;
  for (int i = 0; i < world_size - 1; i++) {
    compressor_->Decompress(recv_buf, decompress_buffer_, entries, start_elem,
                            global_offset, recv_num_elems);
    summator_.Add((float*)decompress_buffer_, (float*)output + start_elem,
                  recv_num_elems, entries[0].device);
    recv_buf += recv_compressed_size;
  }
  send_buf = gradients_send_;
  // Quantize the sum into gradients_recv_[0] and maxandmin_recv[0]
  compressor_->Compress((unsigned char*)output + start_elem * sizeof(float),
                        send_buf, entries, start_elem, global_offset,
                        recv_num_elems);
  unsigned char* result = (unsigned char*)output + start_elem * sizeof(float);
  compressor_->Decompress(send_buf, result, entries, start_elem, global_offset,
                          recv_num_elems);

  recv_buf = gradients_recv_;

  // second round of MPI communication. receive the sums from other nodes
  send_compressed_size = recv_compressed_size;
  int count = 0;
  start = clock_::now();
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

    requests[2 * count] = MPI_Request();
    MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              comm, &requests[2 * count]);
    requests[2 * count + 1] = MPI_Request();
    MPI_Isend(gradients_send_, send_compressed_size, MPI_UNSIGNED_CHAR,
              node_rank, 0, comm, &requests[2 * count + 1]);
    recv_buf += recv_compressed_size;
    count++;
  }

  MPI_Waitall((int)requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  global_state_->communication_time += time_since(start);

  // dequantization
  recv_buf = gradients_recv_;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    // Offset of the received chunk
    int their_start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    recv_compressed_size =
        round_to(compressor_->BufferSize(recv_num_elems, entries,
                                         their_start_offset, global_offset),
                 ALIGNMENT_UNIT);
    result = (unsigned char*)output + their_start_offset * sizeof(float);
    compressor_->Decompress(recv_buf, result, entries, their_start_offset,
                            global_offset, recv_num_elems);
    recv_buf += recv_compressed_size;
  }
  return Status::OK();
}

} // namespace common
} // namespace horovod
