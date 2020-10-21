#include "mpi_allgather.h"
#include "../utils.h"

namespace horovod {
namespace common {

MPI_Allreduce_AllGather::MPI_Allreduce_AllGather(
    MPIContext* mpi_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state, Compressor* compressor,
    Summator* summator)
    : MPIReducer(mpi_context, gpu_context, global_state, compressor, summator) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "AllGather";
  }
}

Status
MPI_Allreduce_AllGather::Init(const std::vector<TensorTableEntry>& entries,
                              MPI_Comm comm) {
  comm_ = comm;
  auto& first_entry = entries[0];
  int world_size = global_state_->controller->GetSize();
  auto& timeline = global_state_->timeline;
  int64_t chunk_size =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  auto dtype = entries[0].tensor->dtype();
  int64_t allocated_compression_buffer_size_send =
      compressor_->BufferSize(chunk_size / get_sizeof(dtype), dtype);
  int64_t allocated_compression_buffer_size_recv =
      allocated_compression_buffer_size_send;
  int64_t buffer_size =
      allocated_compression_buffer_size_send +
      allocated_compression_buffer_size_recv * (world_size - 1) + chunk_size;

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
  gradients_recv_ =
      (unsigned char*)gradients_send_ + allocated_compression_buffer_size_send;
  decompress_buffer_ =
      gradients_recv_ +
      allocated_compression_buffer_size_recv * (world_size - 1);
  status = compressor_->Init(entries);
  if (!status.ok()) {
    return status;
  }
  status = error_feedback_.Init(entries);
  return status;
}

Status MPI_Allreduce_AllGather::AllreduceDivision(
    int num_elements, std::vector<TensorTableEntry>& entries,
    int64_t global_offset) {
  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();
  auto& timeline = global_state_->timeline;
  timeline.ActivityStartAll(entries, Q_COMPRESSION);
  gpuStream_t stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];

  int64_t send_rcv_size = ALIGNED_SIZE(compressor_->Compress(
      gradients_send_, entries, error_feedback_, 0, global_offset, num_elements,
      true, false, &stream));
  cudaStreamSynchronize(stream);
  compressor_->Finalize();
  timeline.ActivityEndAll(entries);
  std::vector<MPI_Request> requests;
  int count = 0;
  timeline.ActivityStartAll(entries, Q_NETWORK);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Irecv(gradients_recv_ + count * send_rcv_size, send_rcv_size,
                        MPI_UNSIGNED_CHAR, node_rank, 0, comm_,
                        &requests.back()));

    requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm_, &requests.back()));
    count++;
  }
  MPI_CHECK(
      MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE));
  timeline.ActivityEndAll(entries);
  timeline.ActivityStartAll(entries, Q_DECOMPRESSION);
  compressor_->Decompress(gradients_send_, entries, 0, global_offset,
                          num_elements, false, (void*)&stream);
  for (int i = 0; i < world_size - 1; i++) {
    compressor_->Decompress(gradients_recv_ + i * send_rcv_size, entries, 0,
                            global_offset, num_elements, true, (void*)&stream);
  }
  cudaStreamSynchronize(stream);
  compressor_->Finalize();
  timeline.ActivityEndAll(entries);
  return Status::OK();
}

} // namespace common
} // namespace horovod
