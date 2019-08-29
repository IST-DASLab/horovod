#include "quantization.h"
#include "cuda_functions.h"
#include "cuda_operations.h"
#include <cuda.h>
#include <cuda_runtime.h>


#define QUANTIZE_THRESHOLD 100000
#define QUANTIZE_BUCKET_SIZE 512

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
}


int64_t round_to1(int64_t x, int64_t m) {
  return x + ((m - x % m) % m);
}

Status SimpleQuantizer::Init(std::vector<TensorTableEntry>& entries, int world_size) {
  InitCUDA(entries);

//  int dequan_buf_size = global_state_->param_manager.TensorFusionThresholdBytes();
//  if (dequan_buffer == nullptr || NumElements(entries) * sizeof(float) >= dequan_buf_size) {
//      std::cout << "buf size: " << dequan_buf_size << std::endl;
//      if (dequan_buffer != nullptr) {
//        cudaFree(dequan_buffer);
//        cudaFree(maxandmin_send);
//        cudaFree(maxandmin_recv);
//        cudaFree(quantized_gradients_send);
//        cudaFree(quantized_gradients_recv);
//      }
//      dequan_buf_size = fmaxf(NumElements(entries) * sizeof(float), dequan_buf_size);
//      quantized_buffer_size = ceil(1.0 * dequan_buf_size * global_state_->quantization_bits / (sizeof(float) * 8));
//      maxmin_size = 2 * ceil(1.0 * dequan_buf_size / (quantize_bucket_size * sizeof(float)));
//      num_elems_in_chunk = ceil(((double)dequan_buf_size) / sizeof(float));
//
//
//      cudaMalloc(&dequan_buffer, dequan_buf_size);
//      GPU_init_curand(cuda_states, num_elems_in_chunk, time(NULL),
//                      cuda_context_->streams[entries[0].device]);
//      std::cout << "max min recv: " << maxmin_size * sizeof(float) << std::endl;
//      std::cout << "max min send: " << maxmin_size * sizeof(float) * (world_size - 1) << std::endl;
//
//      cudaMalloc(&maxandmin_send, maxmin_size * sizeof(float));
//      cudaMalloc(&maxandmin_recv, maxmin_size * sizeof(float) * (world_size - 1));
//      std::cout << "quantize recv: " << quantized_buffer_size * (world_size - 1) << std::endl;
//      std::cout << "quantize send: " << quantized_buffer_size << std::endl;
//
//      // We quantize values from 32 bits to <bits> bits, so the size of quantized chunk is <bits> / 32 of full precision chunk.
//      cudaMalloc(
//          &quantized_gradients_send,
//          quantized_buffer_size);
//      cudaMalloc(
//          &quantized_gradients_recv,
//          quantized_buffer_size * (world_size - 1));
//      HERE
//  }
  auto& first_entry = entries[0];
  chunk_size = fmaxf(NumElements(entries) * sizeof(float), global_state_->param_manager.TensorFusionThresholdBytes());
  quantized_buffer_size = ceil(1.0 * chunk_size * global_state_->quantization_bits / (sizeof(float) * 8));
  maxmin_size = 2 * ceil(1.0 * chunk_size / (quantize_bucket_size * sizeof(float)));
  num_elems_in_chunk = ceil(((double)chunk_size) / sizeof(float));

  chunk_size = round_to1(chunk_size, 2 * sizeof(float));
  maxmin_size = round_to1(maxmin_size, 2 * sizeof(float));
  quantized_buffer_size = round_to1(quantized_buffer_size, 2 * sizeof(float));


  int64_t maxmin_buffers_size =
      maxmin_size * sizeof(float) * (global_state_->size - 1);

  size_t curand_array_size = GPU_get_curand_array_size(num_elems_in_chunk);

  int64_t buffer_size =
      chunk_size + 2 * maxmin_buffers_size +
      + 2 * quantized_buffer_size * (global_state_->size - 1) +
      curand_array_size;
  Status status = bufferManager.InitializeBuffer(
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
  maxandmin_send = (float*)(((char*)dequan_buffer) + chunk_size);
  maxandmin_recv = (float*)(((char*)maxandmin_send) + maxmin_buffers_size);
  quantized_gradients_send =
      (unsigned char*)(maxandmin_recv) + maxmin_buffers_size;
  quantized_gradients_recv =
      quantized_gradients_send +
      quantized_buffer_size * (global_state_->size - 1);

  //Avoid extra rand init.
  auto new_cuda_states = (CurandState*) (quantized_gradients_recv +
                                         quantized_buffer_size * (global_state_->size - 1));
  HERE
  if (cuda_states != new_cuda_states) {
    cuda_states = new_cuda_states;
    std::cout << "Simple Quantizer" << std::endl;
    GPU_init_curand(cuda_states, num_elems_in_chunk, time(NULL),
                    cuda_context_->streams[first_entry.device]);
  }
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
  MPI_Quantized_Allreduce(MPI_IN_PLACE, buffer_data, num_elements,
      MPI_COMM_WORLD, entries, buffer_len);
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

int SimpleQuantizer::MPI_Quantized_Allreduce(void* sendbuf, void* recvbuf, int num_elements,
                                       MPI_Comm comm, std::vector<TensorTableEntry>& entries, int buffer_len) {

  if (sendbuf == MPI_IN_PLACE) {
    sendbuf = recvbuf;
  }
  auto& first_entry = entries[0];

  int rank, world_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);
  Init(entries, world_size);
  int bits = global_state_->quantization_bits;
  int entries_per_byte = 8 / bits;
  int num_buckets = (num_elements + quantize_bucket_size - 1) / quantize_bucket_size;

  HERE
  GPU_find_max_and_min_bucket(
      (float*) sendbuf,
      maxandmin_send, num_elements, quantize_bucket_size,
      cuda_context_->streams[first_entry.device]);

  GPU_quantize_value_bits(
      quantized_gradients_send,
      (float*) sendbuf,
      maxandmin_send, num_elements, bits, quantize_bucket_size,
      cuda_states, cuda_context_->streams[first_entry.device]);
  std::vector<MPI_Request> requests;
  HERE
  int count = 0;

  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;

    requests.push_back(MPI_Request());
    MPI_Irecv(quantized_gradients_recv + count * quantized_buffer_size,
              (num_elements + entries_per_byte - 1) / entries_per_byte,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());

    requests.push_back(MPI_Request());
    MPI_Irecv(maxandmin_recv + count * maxmin_size, num_buckets * 2,
              mpi_context_->GetMPIDataType(first_entry.tensor), node_rank, 0,
              comm, &requests.back());

    requests.push_back(MPI_Request());
    MPI_Isend(quantized_gradients_send,
              (num_elements + entries_per_byte - 1) / entries_per_byte,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());
    // There was a bug with num_buckets
    requests.push_back(MPI_Request());
    MPI_Isend(maxandmin_send, num_buckets * 2,
              mpi_context_->GetMPIDataType(first_entry.tensor), node_rank, 0,
              comm, &requests.back());
    count++;
  }
  MPI_Waitall((int)requests.size(), &requests[0],
              MPI_STATUSES_IGNORE);
  HERE
  GPU_dequantize_value_bits(
      quantized_gradients_send, maxandmin_send, (float*) recvbuf, num_elements,
      bits, quantize_bucket_size, cuda_context_->streams[first_entry.device]);

  for (int i = 0; i < world_size - 1; i++) {
    GPU_dequantize_value_bits(
        quantized_gradients_recv + i * quantized_buffer_size,
        maxandmin_recv + i * maxmin_size, dequan_buffer, num_elements, bits,
        quantize_bucket_size, cuda_context_->streams[first_entry.device]);

    // add dequantized value to right place of data_buffer
    GPU_add(num_elements, dequan_buffer,
            (float*) recvbuf,
            cuda_context_->streams[first_entry.device]);
  }
  HERE
  return MPI_SUCCESS;
}

} // namespace common
} // namespace horovod
