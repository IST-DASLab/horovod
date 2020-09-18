#include "mpi_ps.h"

namespace horovod {
namespace common {

MPI_Allreduce_PS::MPI_Allreduce_PS(
    horovod::common::MPIContext* mpi_context,
    horovod::common::HorovodGlobalState* global_state,
    horovod::common::Compressor* compressor,
    horovod::common::Summator* summator)
    : MPIReducer(mpi_context, global_state, compressor, summator) {
  if (global_state->controller->GetLocalRank() == 0) {
    LOG(INFO) << "MPI Parameter-Server";
  }
}

Status MPI_Allreduce_PS::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
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
      allocated_compression_buffer_size_recv * (world_size - 1);

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
  status = compressor_->Init(entries);
  if (!status.ok()) {
    return status;
  }
  status = error_feedback_.Init(entries);
  return status;
}

Status MPI_Allreduce_PS::AllreduceDivision(
    int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries,
    int64_t global_offset) {
  int rank = global_state_->controller->GetRank();
  int world_size = global_state_->controller->GetSize();
  auto& timeline = global_state_->timeline;
  timeline.ActivityStartAll(entries, Q_COMPRESSION);
  int64_t send_rcv_size = 0;
  int op;
  if (rank == 0) {
    std::vector<MPI_Request> requests;
    std::vector<int> idx_map;
    send_rcv_size =
        compressor_->BufferSize(num_elements, entries, 0, global_offset);

    // First round.
    // Collect all gradients, decompress and aggregate them.
    for (int node_rank = 1; node_rank < world_size; node_rank++) {
      requests.push_back(MPI_Request());
      MPI_CHECK(MPI_Irecv(gradients_recv_ + (node_rank - 1) * send_rcv_size,
                          send_rcv_size, MPI_UNSIGNED_CHAR, node_rank, 0, comm,
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
      compressor_->Decompress(gradients_recv_ + idx * send_rcv_size,
                              entries, 0, global_offset, num_elements,
                              true);
    }

    // Second round.
    // Broadcast the result
    compressor_->Compress(gradients_send_, entries, error_feedback_, 0,
                          global_offset, num_elements, false, false);
    compressor_->Finalize();
  } else {
    send_rcv_size =
        compressor_->Compress(gradients_send_, entries, error_feedback_, 0,
                              global_offset, num_elements);
    compressor_->Finalize();
    op =
        MPI_Send(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR, 0, 0, comm);
    if (op != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Send failed, see MPI output for details.");
    }
  }
  op = MPI_Bcast(gradients_send_, send_rcv_size, MPI_UNSIGNED_CHAR, 0, comm);
  if (op != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Broadcast failed, see MPI output for details.");
  }
  compressor_->Decompress(gradients_send_, entries, 0, global_offset,
                          num_elements, false);
  return Status::OK();
}

} // namespace common
} // namespace horovod