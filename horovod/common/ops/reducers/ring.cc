#include "ring.h"
#include "../../logging.h"
#include "../../utils.h"

namespace horovod {
namespace common {

bool MPI_CUDARingReducer::Enabled(const ParameterManager& param_manager,
                                         const TensorTableEntry& entry,
                                         const Response& response) const {
  if (reduction_type != ReductionType::Ring ||
      entry.tensor->dtype() != HOROVOD_FLOAT32) {
    return false;
  }
  return CUDAAllreduce::Enabled(param_manager, entry, response);
}

MPI_CUDARingReducer::MPI_CUDARingReducer(
    MPIContext* mpi_context, CUDAContext* cuda_context,
    HorovodGlobalState* global_state)
    : MPI_CUDACompressedReducer(mpi_context, cuda_context, global_state) {
  if (global_state_->local_rank == 0 && reduction_type == ReductionType::Ring) {
    LOG(INFO) << "Ring";
  }
}

Status MPI_CUDARingReducer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries, int world_size) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  int64_t chunk_size = (tensor_fusion_threshold + world_size - 1) / world_size;

  int64_t allocated_compression_buffer_size_send = round_to(compressor->BufferSize(chunk_size), ALIGNMENT_UNIT);
  int64_t allocated_compression_buffer_size_recv = allocated_compression_buffer_size_send;
  if (allocated_compression_buffer_size_send == chunk_size) {
    // There won't be any kind of compression,
    // therefore no need of allocations of dequan_buf and compression_buf_send
    allocated_compression_buffer_size_send = 0;
    chunk_size = 0;
  }
  int64_t buffer_size = allocated_compression_buffer_size_send * world_size
                         + allocated_compression_buffer_size_recv  + chunk_size;
  HERE
  Status status = bufferManager.InitializeBuffer(
      buffer_size, first_entry.device, first_entry.context,
      [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
      [&]() { timeline.ActivityEndAll(entries); });
  if (!status.ok()) {
    for (auto& e : entries) {
      timeline.End(e.tensor_name, nullptr);
      e.callback(status);
    }
    return status;
  }
  HERE
  auto& buffer = bufferManager.GetBuffer(first_entry.device,
                                         first_entry.context->framework());
  void* buffer_data =
      const_cast<void*>(buffer->AccessData(first_entry.context));
  gradients_send =
      (unsigned char*) buffer_data;
  gradients_recv =
      gradients_send +
      allocated_compression_buffer_size_send * world_size;
  decompress_buffer =gradients_recv + allocated_compression_buffer_size_recv;
  status = compressor->Init(entries);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }

  return Status::OK();
}

void printDebug2(float *buff, int n=8) {
  float *debugarr = new float[n];
  cudaMemcpy(debugarr, buff, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    std::cout << debugarr[i] << " ";
  }
  std::cout << std::endl;
}


Status MPI_CUDARingReducer::AllreduceDivision(
    void* sendbuf, void* recvbuf, int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int64_t glovbal_offset) {
  auto& first_entry = entries[0];
  int rank, world_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);

  char *buffer = new char[num_elements * sizeof(float)];
  cudaMemcpy((void*)buffer, sendbuf,
             num_elements * sizeof(float), cudaMemcpyDeviceToHost);

  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
// Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;
  MPI_Request recv_req;
  MPI_Status recv_status;
  auto segment_size = [num_elems_per_node, residue](int segment){return num_elems_per_node + ((segment < residue) ? 1 : 0);};
  std::vector<size_t> segment_ends(world_size);
  segment_ends[0] = segment_size(0);
  for (size_t i = 1; i < segment_ends.size(); ++i) {
    segment_ends[i] = segment_size(i) + segment_ends[i - 1];
  }
  float* send = (float*) sendbuf;
  float* recv = (float*) recvbuf;
  HERE
  int recv_segment_idx, send_segment_idx;
  int64_t send_size, recv_size;
  int64_t start = now();
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    float* segment_send = send + (segment_ends[send_segment_idx] -
                                   segment_size(send_segment_idx));
    recv_size = compressor->BufferSize(segment_size(recv_segment_idx) * sizeof(float));
    MPI_Irecv(gradients_recv, recv_size,
              MPI_UNSIGNED_CHAR, recv_from, 0, MPI_COMM_WORLD, &recv_req);
    send_size = compressor->Compress((unsigned char*) segment_send, (void**)&gradients_send, segment_size(send_segment_idx));
    MPI_Send(gradients_send, send_size,
             MPI_UNSIGNED_CHAR, send_to, 0, MPI_COMM_WORLD);

    float *segment_update = recv + (segment_ends[recv_segment_idx] -
                                     segment_size(recv_segment_idx));

    // Wait for recv to complete before reduction
    MPI_Wait(&recv_req, &recv_status);
    compressor->Decompress(gradients_recv, (void**)&decompress_buffer, segment_size(recv_segment_idx));
    CUDA_add(segment_size(recv_segment_idx), (float*)decompress_buffer, segment_update,
        cuda_context_->streams[first_entry.device]);
  }
  HERE
  send_segment_idx = (rank + world_size + 1) % world_size;
  float *segment_send = recv + (segment_ends[send_segment_idx] -
                                segment_size(send_segment_idx));
  unsigned char* send_buf = gradients_send;
  send_size = round_to(compressor->Compress((unsigned char*)segment_send, (void**)&send_buf, segment_size(send_segment_idx)), ALIGNMENT_UNIT);
  compressor->Decompress(send_buf, (void**)&segment_send, segment_size(send_segment_idx));

  unsigned char* recv_buf = send_buf + send_size;
  unsigned char *compressed_buf = recv_buf;
  // Propagate reduced and compressed chunks without decompression.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    // Segment to send - at every iteration we send segment (r+1-i)
    recv_size = round_to(compressor->BufferSize(segment_size(recv_segment_idx) * sizeof(float)), ALIGNMENT_UNIT);
    // Segment to recv - at every iteration we receive segment (r-i)
    MPI_Sendrecv(send_buf, send_size,
                 MPI_UNSIGNED_CHAR, send_to, 0, recv_buf,
                 recv_size, MPI_UNSIGNED_CHAR, recv_from,
                 0, MPI_COMM_WORLD, &recv_status);
    send_buf += send_size;
    recv_buf += recv_size;
    send_size = recv_size;
  }

  // Decompress all chunks we sent.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    float* segment_decompressed = recv + (segment_ends[recv_segment_idx] -
                                   segment_size(recv_segment_idx));
    recv_size = round_to(compressor->BufferSize(segment_size(recv_segment_idx) * sizeof(float)), ALIGNMENT_UNIT);
    if (send_size == round_to(segment_size(recv_segment_idx) * sizeof(float), ALIGNMENT_UNIT)) {
      // In case of no compression. Only need to copy to the right place.
      cudaMemcpy((void*)segment_decompressed,
                 (void*)compressed_buf, recv_size,
                 cudaMemcpyDeviceToDevice);
    } else {
      compressor->Decompress(compressed_buf, (void**)&segment_decompressed,
                             segment_size(recv_segment_idx));
    }
    compressed_buf += recv_size;
  }
  HERE
  global_state_->allreduce_time += now() - start;
  return Status::OK();
}

} // namespace common
} // namespace horovod