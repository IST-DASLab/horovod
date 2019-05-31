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

#include "mpi_quantized_cuda_operations.h"
#include "../logging.h"

namespace horovod {
namespace common {

#define QUANTIZE_THRESHOLD 100000

bool MPI_Quantized_CUDAAllreduce::AcceptableEntry(const TensorTableEntry& entry) const {
  return entry.tensor->size() >= quantize_threshold * sizeof(float);
}

bool MPI_Quantized_CUDAAllreduce::Enabled(
    const ParameterManager& param_manager,
    const TensorTableEntry& entry,
    const Response& response) const {
  if (global_state_->quantization_bits == 32 ||
      entry.tensor->dtype() != HOROVOD_FLOAT32 || !AcceptableEntry(entry)) {
    return false;
  }
  return CUDAAllreduce::Enabled(param_manager, entry, response);
}

bool MPI_Quantized_CUDAAllreduce::Packed(
    const ParameterManager&  param_manager,
    const TensorTableEntry& entry,
    const Response& response,
    const TensorTableEntry& new_entry,
    const Response& new_response) const {
  if (AcceptableEntry(entry) ^ AcceptableEntry(new_entry))
    return false;
  return CUDAAllreduce::Packed(param_manager, entry, response, new_entry,
      new_response);
}

MPI_Quantized_CUDAAllreduce::MPI_Quantized_CUDAAllreduce(
    MPIContext* mpi_context, CUDAContext* cuda_context,
    HorovodGlobalState* global_state)
    : CUDAAllreduce(cuda_context, global_state), mpi_context_(mpi_context) {
  tensor_fusion_threshold =
      global_state_->param_manager.TensorFusionThresholdBytes();
  chunk_size = (int64_t) ceil(1.0 * tensor_fusion_threshold / global_state->size);
//      (tensor_fusion_threshold + global_state->size - 1) / global_state->size;
  maxmin_size = 2 * ceil(1.0 * chunk_size / (bucket_size * sizeof(float)));
  quantized_buffer_size =
      ceil(1.0 * chunk_size * global_state->quantization_bits / (sizeof(float) * 8));
  num_elems_in_chunk = ceil(((double)chunk_size) / sizeof(float));
  const char *quantize_threshold_str = getenv("HOROVOD_QUANTIZE_THRESHOLD");
  if (quantize_threshold_str == nullptr)
    quantize_threshold = QUANTIZE_THRESHOLD;
  else
    quantize_threshold = std::stol(std::string(quantize_threshold_str));
}

Status MPI_Quantized_CUDAAllreduce::Init(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;
  InitCUDA(entries);
  HERE
  int64_t maxmin_buffers_size =
      maxmin_size * sizeof(float) * (global_state_->size - 1);

  size_t curand_array_size = GPU_get_curand_array_size(num_elems_in_chunk);


  int64_t buffer_size =
        chunk_size + 2 * maxmin_buffers_size +
        +2 * quantized_buffer_size * (global_state_->size - 1) +
            curand_array_size;
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
//    void *buffer_data;
//    cudaMalloc(&buffer_data, buffer_size);
  dequan_buffer = (float*)buffer_data;
  maxandmin_send = (float*)(((char*)dequan_buffer) + chunk_size);
  maxandmin_recv = (float*)(((char*)maxandmin_send) + maxmin_buffers_size);
  quantized_gradients_send =
      (unsigned char*)(maxandmin_recv) + maxmin_buffers_size;
  quantized_gradients_recv =
      quantized_gradients_send +
      quantized_buffer_size * (global_state_->size - 1);

  //Avoid extra rand init.
  auto new_cuda_states = (curandState*) (quantized_gradients_recv +
                         quantized_buffer_size * (global_state_->size - 1));
  HERE
  if (cuda_states != new_cuda_states) {
    cuda_states = new_cuda_states;
    GPU_init_curand(cuda_states, num_elems_in_chunk, time(NULL),
                    cuda_context_->streams[first_entry.device]);
  }
  HERE
  return Status::OK();
}

Status
MPI_Quantized_CUDAAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                                     const Response& response) {
  auto& first_entry = entries[0];
  Status status = Init(entries);
  if (!status.ok())
    return status;

  int rank, num_nodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  int bits = global_state_->quantization_bits; // the amount of bits per value,
                                               // should be 1, 2, 4 or 8
  std::stringstream message;
  message << "Quantizing tensors: " << std::endl;
  for (auto& entry: entries) {
    message << entry.tensor_name << " with size " << entry.tensor->size() << " ";
  }
  LOG(DEBUG, global_state_->rank) << message.str();

  int entries_per_byte = 8 / bits;
  int64_t num_elements = NumElements(entries);
  HERE
  void* buffer_data;
  size_t buffer_len;
  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    auto cuda_result =
        cudaStreamSynchronize(cuda_context_->streams[first_entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*)first_entry.tensor->data();
    buffer_len = (size_t)first_entry.tensor->size();
  }
  HERE
  int num_divisions =
      (buffer_len + tensor_fusion_threshold - 1) / tensor_fusion_threshold;
  int num_elements_division = 0;

  for (int division = 0; division < num_divisions; division++) {
    if (entries.size() > 1) {
      num_elements_division = num_elements;
    } else {
      // There was the bug
      num_elements_division =
          division == num_divisions - 1 &&
                  buffer_len % tensor_fusion_threshold != 0
              ? (buffer_len % tensor_fusion_threshold) / sizeof(float)
              : tensor_fusion_threshold / sizeof(float);
    }
    HERE
    int division_offset = division * (tensor_fusion_threshold / sizeof(float));
    int num_elems_per_node = num_elements_division / num_nodes;
    int residue = num_elements_division % num_nodes;
    int start_elem = num_elems_per_node * rank + std::min(residue, rank);
    int num_elems = num_elems_per_node + (rank < residue ? 1 : 0);
    int num_buckets = (num_elems + bucket_size - 1) / bucket_size;

    std::vector<MPI_Request> request_recv;
    std::vector<MPI_Request> request_send;
    request_recv.reserve(2 * (num_nodes - 1));
    request_send.reserve(2 * (num_nodes - 1));
    int count = 0;
    for (int node_rank = 0; node_rank < num_nodes; node_rank++) {
      if (node_rank == rank) {
        continue;
      }
      // get start offset and length of this chunk
      int start_offset =
          (num_elems_per_node * node_rank) + std::min(residue, node_rank);
      int length = num_elems_per_node + ((node_rank < residue) ? 1 : 0);

      // find max and min of this chunk
      timeline.ActivityStartAll(entries, MPI_QUANTIZED_MAXMIN);
      GPU_find_max_and_min_bucket(
          (float*)buffer_data + division_offset + start_offset,
          maxandmin_send + count * maxmin_size, length, bucket_size,
          cuda_context_->streams[first_entry.device]);
      timeline.ActivityEndAll(entries);

      // quantize this chunk
      timeline.ActivityStartAll(entries, MPI_QUANTIZED_QUANTIZATION);
      GPU_quantize_value_bits(
          quantized_gradients_send + count * quantized_buffer_size,
          (float*)buffer_data + division_offset + start_offset,
          maxandmin_send + count * maxmin_size, length, bits, bucket_size,
          cuda_states, cuda_context_->streams[first_entry.device]);
      timeline.ActivityEndAll(entries);

      int nb = (length + bucket_size - 1) / bucket_size;
      HERE
      request_recv.push_back(MPI_Request());
      MPI_Irecv(quantized_gradients_recv + count * quantized_buffer_size,
                (num_elems + entries_per_byte - 1) / entries_per_byte,
                MPI_UNSIGNED_CHAR, node_rank, 0, MPI_COMM_WORLD,
                &request_recv.back());
      HERE

      request_recv.push_back(MPI_Request());
      MPI_Irecv(maxandmin_recv + count * maxmin_size, num_buckets * 2,
                mpi_context_->GetMPIDataType(first_entry.tensor), node_rank, 0,
                MPI_COMM_WORLD, &request_recv.back());
      HERE

      request_send.push_back(MPI_Request());
      MPI_Isend(quantized_gradients_send + count * quantized_buffer_size,
                (length + entries_per_byte - 1) / entries_per_byte,
                MPI_UNSIGNED_CHAR, node_rank, 0, MPI_COMM_WORLD,
                &request_send.back());
      HERE

      // There was a bug with num_buckets
      request_send.push_back(MPI_Request());
      MPI_Isend(maxandmin_send + count * maxmin_size, nb * 2,
                mpi_context_->GetMPIDataType(first_entry.tensor), node_rank, 0,
                MPI_COMM_WORLD, &request_send.back());
      HERE
      count++;
    }
    // TODO: handling errors!!!
    MPI_Waitall((int)request_recv.size(), &request_recv[0],
                MPI_STATUSES_IGNORE);
    MPI_Waitall((int)request_send.size(), &request_send[0],
                MPI_STATUSES_IGNORE);

    for (int i = 0; i < num_nodes - 1; i++) {
      timeline.ActivityStartAll(entries, MPI_QUANTIZED_DEQUANTIZATION);
      GPU_dequantize_value_bits(
          quantized_gradients_recv + i * quantized_buffer_size,
          maxandmin_recv + i * maxmin_size, dequan_buffer, num_elems, bits,
          bucket_size, cuda_context_->streams[first_entry.device]);
      timeline.ActivityEndAll(entries);

      // add dequantized value to right place of data_buffer
      timeline.ActivityStartAll(entries, MPI_QUANTIZED_SUM);
      GPU_add(num_elems, dequan_buffer,
              (float*)buffer_data + division_offset + start_elem,
              cuda_context_->streams[first_entry.device]);
      timeline.ActivityEndAll(entries);
    }
    HERE

    // Quantize the sum into quantized_gradients_recv[0] and maxandmin_recv[0]
    GPU_find_max_and_min_bucket(
        (float*)buffer_data + division_offset + start_elem, maxandmin_recv,
        num_elems, bucket_size, cuda_context_->streams[first_entry.device]);
    timeline.ActivityStartAll(entries, MPI_QUANTIZED_QUANTIZATION);
    GPU_quantize_value_bits(quantized_gradients_recv,
                            (float*)buffer_data + division_offset + start_elem,
                            maxandmin_recv, num_elems, bits, bucket_size,
                            cuda_states,
                            cuda_context_->streams[first_entry.device]);
    timeline.ActivityEndAll(entries);
    timeline.ActivityStartAll(entries, MPI_QUANTIZED_DEQUANTIZATION);
    // Dequantize the sum in place of the sum itself. This is necessary for
    // models to be the same.
    GPU_dequantize_value_bits(quantized_gradients_recv, maxandmin_recv,
                              dequan_buffer, num_elems, bits, bucket_size,
                              cuda_context_->streams[first_entry.device]);
    timeline.ActivityEndAll(entries);
    cudaMemcpy((void*)((float*)buffer_data + division_offset + start_elem),
               (const void*)dequan_buffer, num_elems * sizeof(float),
               cudaMemcpyDeviceToDevice);
    HERE

    // second round of MPI communication. receive the sums from other nodes
    count = 0;
    for (int node_rank = 0; node_rank < num_nodes; node_rank++) {
      if (node_rank == rank) {
        continue;
      }
      int length = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
      int nb = (length + bucket_size - 1) / bucket_size;
//      MPI_Waitall(2, &request_send[2 * count], MPI_STATUS_IGNORE);

      request_recv[2 * count] = MPI_Request();
      MPI_Irecv(quantized_gradients_send + count * quantized_buffer_size,
                (length + entries_per_byte - 1) / entries_per_byte,
                MPI_UNSIGNED_CHAR, node_rank, 0, MPI_COMM_WORLD,
                &request_recv[2 * count]);
      request_recv[2 * count + 1] = MPI_Request();
      MPI_Irecv(maxandmin_send + count * maxmin_size, nb * 2,
                mpi_context_->GetMPIDataType(first_entry.tensor), node_rank, 0,
                MPI_COMM_WORLD, &request_recv[2 * count + 1]);

      request_send[2 * count] = MPI_Request();
      MPI_Isend(quantized_gradients_recv,
                (num_elems + entries_per_byte - 1) / entries_per_byte,
                MPI_UNSIGNED_CHAR, node_rank, 0, MPI_COMM_WORLD,
                &request_send[2 * count]);

      request_send[2 * count + 1] = MPI_Request();
      MPI_Isend(maxandmin_recv, num_buckets * 2,
                mpi_context_->GetMPIDataType(first_entry.tensor), node_rank, 0,
                MPI_COMM_WORLD, &request_send[2 * count + 1]);
      count++;
    }

    MPI_Waitall((int)request_recv.size(), &request_recv[0],
                MPI_STATUSES_IGNORE);
    HERE

    // dequantization
    count = 0;
    for (int i = 0; i < num_nodes; i++) {
      if (i == rank) {
        continue;
      }
      int start_offset = (num_elems_per_node * i) + std::min(residue, i);
      int length = num_elems_per_node + ((i < residue) ? 1 : 0);

      // dequantize chunk from node i in dequan_buffer
      GPU_dequantize_value_bits(
          quantized_gradients_send + count * quantized_buffer_size,
          maxandmin_send + count * maxmin_size, dequan_buffer, length, bits,
          bucket_size, cuda_context_->streams[first_entry.device]);

      // copy dequantized data to right place of data_buffer
      cudaMemcpy((void*)((float*)buffer_data + division_offset + start_offset),
                 (const void*)dequan_buffer, length * sizeof(float),
                 cudaMemcpyDeviceToDevice);
      count++;
    }
    MPI_Waitall(request_send.size(), &request_send[0], MPI_STATUSES_IGNORE);
  }
  HERE

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);

    auto cuda_result =
        cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  } else {
    if (first_entry.tensor->data() != first_entry.output->data()) {
      cudaMemcpy((void*)first_entry.output->data(), (const void*)buffer_data,
                 first_entry.output->size(), cudaMemcpyDeviceToDevice);
    }
  }
  HERE

  return Status::OK();
}

} // namespace common
} // namespace horovod
