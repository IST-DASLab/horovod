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

SimpleQuantizer::SimpleQuantizer(MPIContext* mpi_context, CUDAContext* cuda_context,
                                 HorovodGlobalState* global_state):
      MPI_Quantized_CUDAAllreduce(mpi_context, cuda_context, global_state) {
}


void SimpleQuantizer::Init(std::vector<TensorTableEntry>& entries, int world_size) {
    int dequan_buf_size = global_state_->param_manager.TensorFusionThresholdBytes();
    if (dequan_buffer == nullptr || NumElements(entries) * sizeof(float) >= dequan_buf_size) {
      if (dequan_buffer != nullptr) {
        cudaFree(dequan_buffer);
        cudaFree(maxandmin_send);
        cudaFree(maxandmin_recv);
        cudaFree(quantized_gradients_send);
        cudaFree(quantized_gradients_recv);
      }
      dequan_buf_size = fmaxf(NumElements(entries) * sizeof(float), dequan_buf_size);
      quantized_buffer_size = ceil(1.0 * dequan_buf_size * global_state_->quantization_bits / (sizeof(float) * 8));
      maxmin_size = 2 * ceil(1.0 * dequan_buf_size / (quantize_bucket_size * sizeof(float)));
      num_elems_in_chunk = ceil(((double)dequan_buf_size) / sizeof(float));


      cudaMalloc(&dequan_buffer, dequan_buf_size);
      GPU_init_curand(cuda_states, num_elems_in_chunk, time(NULL),
                      cuda_context_->streams[entries[0].device]);
      cudaMalloc(&maxandmin_send, maxmin_size * sizeof(float));
      cudaMalloc(&maxandmin_recv, maxmin_size * sizeof(float) * (world_size - 1));

      // We quantize values from 32 bits to <bits> bits, so the size of quantized chunk is <bits> / 32 of full precision chunk.
      cudaMalloc(
          &quantized_gradients_send,
          quantized_buffer_size);
      cudaMalloc(
          &quantized_gradients_recv,
          quantized_buffer_size * (world_size - 1));
    }
}

int SimpleQuantizer::MPI_Quantized_Allreduce(void* sendbuf, void* recvbuf, int num_elements,
                                       MPI_Comm comm, std::vector<TensorTableEntry>& entries) {

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



  GPU_find_max_and_min_bucket(
      (float*)sendbuf,
      maxandmin_send, num_elements, quantize_bucket_size,
      cuda_context_->streams[first_entry.device]);

  GPU_quantize_value_bits(
      quantized_gradients_send,
      (float*) sendbuf,
      maxandmin_send, num_elements, bits, quantize_bucket_size,
      cuda_states, cuda_context_->streams[first_entry.device]);
  std::vector<MPI_Request> requests;

  int count1 = 0;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;

    requests.push_back(MPI_Request());
    MPI_Irecv(quantized_gradients_recv + count1 * quantized_buffer_size,
              (num_elements + entries_per_byte - 1) / entries_per_byte,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());

    requests.push_back(MPI_Request());
    MPI_Irecv(maxandmin_recv + count1 * maxmin_size, num_buckets * 2,
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
    count1++;
  }
  MPI_Waitall((int)requests.size(), &requests[0],
              MPI_STATUSES_IGNORE);

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
  return MPI_SUCCESS;
}

} // namespace common
} // namespace horovod
