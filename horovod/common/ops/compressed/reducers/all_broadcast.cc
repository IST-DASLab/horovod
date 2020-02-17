#include "all_broadcast.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_GPUAllreduce_AllBroadcast::MPI_GPUAllreduce_AllBroadcast(
    MPIContext* mpi_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state, Compressor* compressor)
    : MPIReducer(mpi_context, gpu_context, global_state, compressor) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "AllBroadcast";
  }
}

Status MPI_GPUAllreduce_AllBroadcast::Init(
    const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  int world_size = global_state_->controller->GetSize();
  auto& timeline = global_state_->timeline;

  int64_t chunk_size =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  int64_t allocated_compression_buffer_size_send =
      compressor_->BufferSize(chunk_size / sizeof(float));
  int64_t allocated_compression_buffer_size_recv =
      allocated_compression_buffer_size_send;
  int64_t buffer_size =
      allocated_compression_buffer_size_send +
      allocated_compression_buffer_size_recv * (world_size - 1) + chunk_size;
  // TODO: Add timeline callbacks
  const auto& status = bufferManager_.InitializeBuffer(
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
  gradients_recv_ =
      (unsigned char*)gradients_send_ +
      round_to(allocated_compression_buffer_size_send, ALIGNMENT_UNIT);
  decompress_buffer_ =
      gradients_recv_ +
      round_to(allocated_compression_buffer_size_recv * (world_size - 1),
               ALIGNMENT_UNIT);
  return Status::OK();
}

Status MPI_GPUAllreduce_AllBroadcast::AllreduceDivision(
    void* sendbuf, void* recvbuf, int num_elements, MPI_Comm comm,
    std::vector<TensorTableEntry>& entries, int64_t global_offset) {
  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();

  Status status = compressor_->Init(entries);
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

  errorFeedbackManager_.ApplyErrorFeedback(entries, sendbuf, num_elements,
                                           global_offset);
  int64_t send_rcv_size =
      compressor_->Compress((unsigned char*)sendbuf, gradients_send_, entries,
                            0, global_offset, num_elements);

  errorFeedbackManager_.UpdateErrorFeedback(entries, sendbuf, gradients_send_,
                                            num_elements, 0, global_offset,
                                            compressor_);
  std::vector<MPI_Request> requests;
  int count = 0;
  auto start = clock_::now();
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    requests.push_back(MPI_Request());
    MPI_Irecv(gradients_recv_ + count * send_rcv_size, send_rcv_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm, &requests.back());

    requests.push_back(MPI_Request());
    MPI_Isend(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              comm, &requests.back());
    count++;
  }
  MPI_Waitall((int)requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  global_state_->communication_time += time_since(start);
  compressor_->Decompress(gradients_send_, (unsigned char*)recvbuf, entries, 0,
                          global_offset, num_elements);

  for (int i = 0; i < world_size - 1; i++) {
    compressor_->Decompress(gradients_recv_ + i * send_rcv_size,
                            decompress_buffer_, entries, 0, global_offset,
                            num_elements);
    //  add decompressed value to the right place of data_buffer
    summator_.Add((float*)decompress_buffer_, (float*)recvbuf, num_elements,
        entries[0].device);
  }
  global_state_->compression_time = compressor_->getCompressionTime();
  global_state_->meta_info_time = compressor_->getMetaInfoTime();
  return Status::OK();
}

} // namespace common
} // namespace horovod