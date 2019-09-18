#include "quantization.h"
#include "cuda_functions.h"
#include "cuda_operations.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "../utils.h"

#define QUANTIZE_THRESHOLD 100000
#define COMPRESSION_BUCKET_SIZE 512

int64_t quantize_threshold = 0;
int64_t quantize_bucket_size = 0;

namespace horovod {
namespace common {

bool SimpleQuantizer::Enabled(
    const ParameterManager& param_manager,
    const TensorTableEntry& entry,
    const Response& response) const {
  if (global_state_->quantization_bits == 32 ||
      entry.tensor->dtype() != HOROVOD_FLOAT32 || !AcceptableEntry(entry)) {
    return false;
  }
  return CUDAAllreduce::Enabled(param_manager, entry, response);
}

SimpleQuantizer::SimpleQuantizer(MPIContext* mpi_context, CUDAContext* cuda_context,
                                 HorovodGlobalState* global_state):
      MPI_Quantized_CUDAAllreduce(mpi_context, cuda_context, global_state) {
  std::cout << "Simple Quantizer" << std::endl;
  const char *env_str = getenv(HOROVOD_COMPRESSOR);
  if (env_str == nullptr)
    compressor = new MaxMinQuantizer(cuda_context);
  else {
    switch (*env_str) {
    case 'm':
      compressor = new MaxMinQuantizer(cuda_context);
      break;
    case 'n':
      compressor = CreateNormalized(cuda_context);
      break;
    default:
      compressor = new MaxMinQuantizer(cuda_context);
      break;
    }
  }
}

Status SimpleQuantizer::Init(std::vector<TensorTableEntry>& entries,
    int num_elements, int world_size) {
  InitCUDA(entries);
  auto& first_entry = entries[0];
  chunk_size = fmaxf(num_elements * sizeof(float), global_state_->param_manager.TensorFusionThresholdBytes());
  Status status = compressor->Init(global_state_, num_elements, entries);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
  int64_t allocated_compression_buffer_size = compressor->BufferSize();
  int64_t buffer_size = chunk_size + allocated_compression_buffer_size * world_size;
  status = bufferManager.InitializeBuffer(
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
  dequan_buffer = (float*)buffer_data;
  quantized_gradients_send =
      (unsigned char*)(dequan_buffer) + chunk_size;
  quantized_gradients_recv =
      quantized_gradients_send + allocated_compression_buffer_size;

  return Status::OK();
}

Status
SimpleQuantizer::Execute(
    std::vector<horovod::common::TensorTableEntry>& entries,
    const horovod::common::Response& response) {
  auto& first_entry = entries[0];
  int64_t num_elements = NumElements(entries);
  void* buffer_data;
  size_t buffer_len;

  if (entries.size() > 1) {
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    auto cuda_result =
        cudaStreamSynchronize(cuda_context_->streams[first_entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
  } else {
    buffer_data = (void*)first_entry.tensor->data();
    buffer_len = (size_t)first_entry.tensor->size();
  }
  HERE
  int64_t start = now();
  Status status = MPI_Quantized_Allreduce(MPI_IN_PLACE, buffer_data, num_elements,
      MPI_COMM_WORLD, entries, buffer_len);
  global_state_->allreduce_time += now() - start;
  if (!status.ok()) {
    return status;
  }
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);

    auto cuda_result =
        cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

  } else {
    if (first_entry.tensor->data() != first_entry.output->data()) {
      cudaMemcpy((void*)first_entry.output->data(), (const void*)buffer_data,
                 first_entry.output->size(), cudaMemcpyDeviceToDevice);
    }
  }
  HERE
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

Status SimpleQuantizer::MPI_Quantized_Allreduce(void* sendbuf, void* recvbuf, int num_elements,
                                       MPI_Comm comm, std::vector<TensorTableEntry>& entries, int buffer_len) {
  if (sendbuf == MPI_IN_PLACE) {
    sendbuf = recvbuf;
  }
  void *input_data = sendbuf;
  auto& first_entry = entries[0];
  int rank, world_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);
  Status status = Init(entries, num_elements, world_size);
  if (!status.ok()) {
    return status;
  }
  int64_t compressed_size = compressor->Compress(input_data, quantized_gradients_send);
  std::vector<MPI_Request> requests;
  HERE
  int count = 0;
  int64_t start = now();
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    requests.push_back(MPI_Request());
    MPI_Irecv(quantized_gradients_recv + count * compressed_size,
              compressed_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());

    requests.push_back(MPI_Request());
    MPI_Isend(quantized_gradients_send,
              compressed_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());
    count++;
  }
  MPI_Waitall((int)requests.size(), &requests[0],
              MPI_STATUSES_IGNORE);
  global_state_->communication_time += now() - start;
  HERE
  compressor->Decompress(quantized_gradients_send, recvbuf);
  for (int i = 0; i < world_size - 1; i++) {
    compressor->Decompress(quantized_gradients_recv + i * compressed_size,
        dequan_buffer);
    // add decompressed value to the right place of data_buffer
    CUDA_add(num_elements, dequan_buffer, (float*)recvbuf,
             cuda_context_->streams[first_entry.device]);
  }
  compressor->Correct(recvbuf);
  global_state_->compression_time = compressor->compression_time;
  global_state_->meta_info_time = compressor->meta_info_time;
  HERE
  return Status::OK();
}

} // namespace common
} // namespace horovod
