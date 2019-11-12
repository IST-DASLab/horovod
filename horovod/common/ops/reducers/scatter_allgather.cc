// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "scatter_allgather.h"
#include "../../logging.h"
#include "../../utils.h"

namespace horovod {
namespace common {

bool MPI_CUDAScatterAllgatherReducer::Enabled(
    const ParameterManager& param_manager,
    const TensorTableEntry& entry,
    const Response& response) const {
  if (reduction_type != ReductionType::ScatterAllgather ||
      entry.tensor->dtype() != HOROVOD_FLOAT32) {
    return false;
  }
  return CUDAAllreduce::Enabled(param_manager, entry, response);
}

MPI_CUDAScatterAllgatherReducer::MPI_CUDAScatterAllgatherReducer(
    MPIContext* mpi_context, CUDAContext* cuda_context,
    HorovodGlobalState* global_state)
    : MPI_CUDACompressedReducer(mpi_context, cuda_context, global_state){
  if (global_state_->local_rank == 0 && reduction_type == ReductionType::ScatterAllgather) {
    LOG(INFO) << "ScatterAllgather";
  }
}

Status MPI_CUDAScatterAllgatherReducer::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries, int world_size) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;

  int64_t chunk_size = (tensor_fusion_threshold + global_state_->size - 1) / global_state_->size;
//  int64_t chunk_size = tensor_fusion_threshold;

  int64_t allocated_compression_buffer_size_send = round_to(compressor->BufferSize(chunk_size), ALIGNMENT_UNIT);
  int64_t allocated_compression_buffer_size_recv = allocated_compression_buffer_size_send;
  if (allocated_compression_buffer_size_send == chunk_size) {
    // There won't be any kind of compression,
    // therefore no need of allocations of dequan_buf and compression_buf_send
    allocated_compression_buffer_size_send = 0;
    chunk_size = 0;
  }
  int64_t buffer_size = allocated_compression_buffer_size_send * (world_size - 1) +
                        + allocated_compression_buffer_size_recv * (world_size - 1) + chunk_size;

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

  auto& buffer = bufferManager.GetBuffer(first_entry.device,
                                         first_entry.context->framework());
  void* buffer_data =
      const_cast<void*>(buffer->AccessData(first_entry.context));
  gradients_send =
      (unsigned char*) buffer_data;
  gradients_recv =
      gradients_send + allocated_compression_buffer_size_send * (world_size - 1);
  decompress_buffer = gradients_recv +
      allocated_compression_buffer_size_recv * (world_size - 1);
  status = compressor->Init(entries);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }

  return Status::OK();
}

void printDebug1(float *buff, int n=8) {
  float *debugarr = new float[n];
  cudaMemcpy(debugarr, buff, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    std::cout << debugarr[i] << " ";
  }
  std::cout << std::endl;
}


Status MPI_CUDAScatterAllgatherReducer::AllreduceDivision(
    void* input, void* output, int num_elements, MPI_Comm comm,
    std::vector<TensorTableEntry>& entries, int64_t glovbal_offset) {
  auto& first_entry = entries[0];
  int rank = global_state_->rank;
  int world_size = global_state_->size;
  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
  int start_elem = num_elems_per_node * rank + std::min(residue, rank);
  int recv_num_elems = num_elems_per_node + (rank < residue ? 1 : 0);
  int recv_compressed_size = round_to(compressor->BufferSize(recv_num_elems * sizeof(float)), ALIGNMENT_UNIT);
  int send_num_elems = 0;
  int send_compressed_size = 0;

  unsigned char *send_buf = gradients_send;
  unsigned char *recv_buf = gradients_recv;
  std::vector<MPI_Request> requests;
  HERE
  unsigned char* input_buf;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    requests.push_back(MPI_Request());
    MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());
    int start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    input_buf = (unsigned char*) input + start_offset * sizeof(float);
    send_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    send_compressed_size = round_to(compressor->Compress(input_buf, (void **)&send_buf, send_num_elems), ALIGNMENT_UNIT);

    requests.push_back(MPI_Request());
    MPI_Isend(send_buf, send_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests.back());

    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
  }
  HERE
  // TODO: handling errors!!!
  MPI_Waitall((int)requests.size(), &requests[0],
              MPI_STATUSES_IGNORE);
  recv_buf = gradients_recv;
  for (int i = 0; i < world_size - 1; i++) {
    compressor->Decompress(recv_buf, (void **)&decompress_buffer, recv_num_elems);
    CUDA_add(recv_num_elems, (float *)decompress_buffer,
             (float*)output + start_elem,
             cuda_context_->streams[first_entry.device]);
    recv_buf += recv_compressed_size;
  }
  HERE
  send_buf = gradients_send;
  // Quantize the sum into gradients_recv[0] and maxandmin_recv[0]
  compressor->Compress((unsigned char*)output + start_elem * sizeof(float),
                       (void **)&send_buf, recv_num_elems);
  unsigned char *result = (unsigned char *)output + start_elem * sizeof(float);
  compressor->Decompress(send_buf, (void **) &result, recv_num_elems);

  recv_buf = gradients_recv;

  // second round of MPI communication. receive the sums from other nodes
  send_compressed_size = recv_compressed_size;
  int count = 0;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    recv_compressed_size = round_to(compressor->BufferSize(recv_num_elems * sizeof(float)), ALIGNMENT_UNIT);

    requests[2 * count] = MPI_Request();
    MPI_Irecv(recv_buf,
              recv_compressed_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests[2 * count]);
    requests[2 * count + 1] = MPI_Request();
    MPI_Isend(gradients_send,
              send_compressed_size,
              MPI_UNSIGNED_CHAR, node_rank, 0, comm,
              &requests[2 * count + 1]);
    recv_buf += recv_compressed_size;
    count++;
  }

  MPI_Waitall((int)requests.size(), &requests[0],
              MPI_STATUSES_IGNORE);
  HERE

  // dequantization
  recv_buf = gradients_recv;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    // Offset of the received chunk
    int their_start_offset = (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    recv_compressed_size = round_to(compressor->BufferSize(recv_num_elems * sizeof(float)), ALIGNMENT_UNIT);
    result = (unsigned char *) output + their_start_offset * sizeof(float);
    if (recv_compressed_size == round_to(recv_num_elems * sizeof(float), ALIGNMENT_UNIT)) {
      // In case of no compression. Only need to copy to the right place.
      cudaMemcpy((void*)result,
                 (void*) recv_buf, recv_compressed_size,
                 cudaMemcpyDeviceToDevice);
    } else {
      compressor->Decompress(recv_buf, (void **) &result, recv_num_elems);
    }
    recv_buf += recv_compressed_size;
  }
  return Status::OK();
}

} // namespace common
} // namespace horovod
