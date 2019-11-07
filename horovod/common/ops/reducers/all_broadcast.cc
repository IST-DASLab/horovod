#include "all_broadcast.h"
#include "../cuda_functions.h"
#include "../cuda_operations.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../utils.h"
#include "../../logging.h"


#define QUANTIZE_THRESHOLD 100000
#define COMPRESSION_BUCKET_SIZE 512

namespace horovod {
namespace common {

bool MPI_CUDAAllBroadcastReducer::Enabled(
    const ParameterManager& param_manager,
    const TensorTableEntry& entry,
    const Response& response) const {
  if (entry.tensor->dtype() != HOROVOD_FLOAT32 || reduction_type != ReductionType::AllBroadcast) {
    return false;
  }
  return CUDAAllreduce::Enabled(param_manager, entry, response);
}

MPI_CUDAAllBroadcastReducer::MPI_CUDAAllBroadcastReducer(MPIContext* mpi_context, CUDAContext* cuda_context,
                                 HorovodGlobalState* global_state):
      MPI_CUDACompressedReducer(mpi_context, cuda_context, global_state) {
  if (global_state_->local_rank == 0 && reduction_type == ReductionType::AllBroadcast) {
    LOG(INFO) << "AllBroadcast";
  }
}

Status MPI_CUDAAllBroadcastReducer::Init(const std::vector<TensorTableEntry>& entries, int world_size) {
  auto& first_entry = entries[0];
  int64_t chunk_size = global_state_->param_manager.TensorFusionThresholdBytes();
  int64_t allocated_compression_buffer_size_send = compressor->BufferSize(chunk_size);
  int64_t allocated_compression_buffer_size_recv = allocated_compression_buffer_size_send;
  int64_t buffer_size = allocated_compression_buffer_size_send
      + allocated_compression_buffer_size_recv * (world_size - 1) + chunk_size;
  const auto &status = bufferManager.InitializeBuffer(
      buffer_size, first_entry.device, first_entry.context,
      [&]() {},
      [&]() {});
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }

  auto& buffer = bufferManager.GetBuffer(first_entry.device,
                                         first_entry.context->framework());
  void* buffer_data =
      const_cast<void*>(buffer->AccessData(first_entry.context));
  gradients_send = (unsigned char *) buffer_data;
  gradients_recv =
      (unsigned char *) gradients_send + round_to(allocated_compression_buffer_size_send, ALIGNMENT_UNIT);
  decompress_buffer = gradients_recv + round_to(allocated_compression_buffer_size_recv * (world_size - 1), ALIGNMENT_UNIT);
  return Status::OK();
}


Status MPI_CUDAAllBroadcastReducer::NonCompressedInit(std::vector<TensorTableEntry>& entries,
                             int num_elements, int world_size) {
  auto& first_entry = entries[0];
  int64_t chunk_size = round_to(num_elements * sizeof(float), 2 * sizeof(float));
  if (gradients_recv == nullptr || chunk_size > global_state_->param_manager.TensorFusionThresholdBytes()) {
    int64_t size = fmaxf(chunk_size, global_state_->param_manager.TensorFusionThresholdBytes());
    int64_t buffer_size = size * (world_size - 1);
    Status status =
        bufferManager.InitializeBuffer(buffer_size, first_entry.device,
                                       first_entry.context, [&]() {}, [&]() {});
    if (!status.ok()) {
      for (auto& e : entries) {
        e.callback(status);
      }
      return status;
    }

    auto& buffer = bufferManager.GetBuffer(first_entry.device,
                                           first_entry.context->framework());
    gradients_recv = (unsigned char*)const_cast<void*>(
        buffer->AccessData(first_entry.context));
  }
  return Status::OK();
}

void printDebug(float *buff) {
  const int N = 8;
  float *debugarr = new float[N];
  cudaMemcpy(debugarr, buff, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    std::cout << debugarr[i] << " ";
  }
  std::cout << std::endl;
}


Status MPI_CUDAAllBroadcastReducer::NonCompressed_Allreduce(void* sendbuf, void* recvbuf, int num_elements,
                                                MPI_Comm comm, std::vector<TensorTableEntry>& entries, int buffer_len) {
  if (sendbuf == MPI_IN_PLACE) {
    sendbuf = recvbuf;
  }
  void *input_data = sendbuf;
  auto& first_entry = entries[0];
  int rank, world_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);
  Status status = NonCompressedInit(entries, num_elements, world_size);
  if (!status.ok()) {
    return status;
  }
  int64_t chunk_size = buffer_len;

  HERE
  int64_t start = now();
  std::vector<MPI_Request> requests;
  HERE
//  printDebug((float *)recvbuf);
  int count = 0;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    requests.push_back(MPI_Request());
    MPI_Irecv(gradients_recv + count * chunk_size,
              chunk_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());

    requests.push_back(MPI_Request());
    MPI_Isend(input_data,
              chunk_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());
    count++;
  }
  HERE
  MPI_Waitall((int)requests.size(), &requests[0],
              MPI_STATUSES_IGNORE);
  global_state_->communication_time += now() - start;
  HERE
  for (int i = 0; i < world_size - 1; i++) {
    // add decompressed value to the right place of data_buffer
    CUDA_add(num_elements, (float *)(gradients_recv + i * chunk_size), (float*)recvbuf,
             cuda_context_->streams[first_entry.device]);
  }
//  printDebug((float*) recvbuf);
  HERE
  return Status::OK();
}

Status MPI_CUDAAllBroadcastReducer::AllreduceDivision(
    void* sendbuf, void* recvbuf, int num_elements, MPI_Comm comm,
    std::vector<TensorTableEntry>& entries, int buffer_len) {
  auto& first_entry = entries[0];
  int rank, world_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);
  Status status = compressor->Init(global_state_, entries);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
//  if (global_state_->rank == 0)
//  printDebug((float*) recvbuf);
  int64_t send_rcv_size = round_to(
      compressor->Compress((unsigned char *)sendbuf, (void **)&gradients_send, num_elements), ALIGNMENT_UNIT);
  std::vector<MPI_Request> requests;
  HERE
  int count = 0;
  int64_t start = now();
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    requests.push_back(MPI_Request());
    MPI_Irecv(gradients_recv + count * send_rcv_size,
              send_rcv_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());

    requests.push_back(MPI_Request());
    MPI_Isend(gradients_send,
              send_rcv_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());
    count++;
  }
  MPI_Waitall((int)requests.size(), &requests[0],
              MPI_STATUSES_IGNORE);
  global_state_->communication_time += now() - start;
  compressor->Decompress(gradients_send, &recvbuf, num_elements);
  HERE
//  if (global_state_->rank == 0)
//    printDebug((float*) recvbuf);

  for (int i = 0; i < world_size - 1; i++) {
    compressor->Decompress(gradients_recv + i * send_rcv_size,
                           (void **)&decompress_buffer, num_elements);
//    if (global_state_->rank == 0) {
//      printDebug((float*)decompress_buffer);
//    }

    // add decompressed value to the right place of data_buffer
    CUDA_add(num_elements, (float *) decompress_buffer, (float*) recvbuf,
             cuda_context_->streams[first_entry.device]);
  }
  compressor->Correct(recvbuf, 0);
  global_state_->compression_time = compressor->compression_time;
  global_state_->meta_info_time = compressor->meta_info_time;
  HERE
  return Status::OK();
}

} // namespace common
} // namespace horovod
