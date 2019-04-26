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

namespace horovod {
namespace common {

Status ExtraBufferManager::InitializeBuffers(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  int64_t tensor_fusion_threshold =
      global_state_->param_manager.TensorFusionThresholdBytes();
  int64_t chunk_size =
      (int64_t)ceil(1.0 * tensor_fusion_threshold / global_state->size);

  // allocate memory for this chunk. We partition the entire gradients into N chunks, where N is the number of MPI nodes.
  Status status =
      first_entry.context->AllocatePersistent(chunk_size, &dequan_buffer_pb);
  dequan_buffer = (float*)accessData(dequan_buffer_pb);

  int num_elems_in_chunk = ceil(((double)chunk_size) / sizeof(float));
  size_t curand_states_size = GPU_get_curand_array_size(num_elems_in_chunk);

  status = first_entry.context->AllocatePersistent(curand_states_size,
                                                   &cuda_states_pb);
  cuda_states = (curandState*)accessData(cuda_states_pb);
  GPU_init_curand(cuda_states, num_elems_in_chunk, time(NULL),
                  cuda_context_->streams[first_entry.device]);

  // Calculate how many buckets in each chunk.
  // Then each bucket has 2 float values: maximum and minimum.
  size_t maxmin_size = ceil(1.0 * chunk_size / (bucket_size * sizeof(float))) *
                       2 * sizeof(float);
  size_t quan_grad_size =
      ceil(1.0 * chunk_size * global_state->quantization_bits / 32);

  for (int i = 0; i < num_nodes - 1; i++) {
    std::shared_ptr<PersistentBuffer> buf;
    status = first_entry.context->AllocatePersistent(maxmin_size, &buf);
    maxandmin_send.push_back((float*)accessData(buf));
    maxandmin_send_buf.push_back(std::move(buf));
    status = first_entry.context->AllocatePersistent(maxmin_size, &buf);
    maxandmin_recv.push_back((float*)accessData(buf));
    maxandmin_recv_buf.push_back(std::move(buf));

    // We quantize values from 32 bits to <bits> bits, so the size of quantized chunk is <bits> / 32 of full precision chunk.
    status = first_entry.context->AllocatePersistent(quan_grad_size, &buf);
    quantized_gradients.push_back((unsigned char*)accessData(buf));
    quantized_gradients_buf.push_back(std::move(buf));
    status = first_entry.context->AllocatePersistent(quan_grad_size, &buf);
    quantized_gradients_recv.push_back((unsigned char*)accessData(buf));
    quantized_gradients_recv_buf.push_back(std::move(buf));
  }
}

template<int bufIndex>
void* ExtraBufferManager::getBuffer(int vInd) {
  auto buffers = buffers_[curFramework];
  if (bufIndex <= CUDA_STATES) {
    return std::get<bufIndex>(bufs_);
  }
  return nullptr;
}

template<int bufIndex, typename T>
std::vector<T>& ExtraBufferManager::getvBuffer(int vInd) {
  auto buffers = buffers_[curFramework];
  if (bufIndex <= CUDA_STATES) {
    throw std::logic_error("No vector of buffer by index ");
  }
  return nullptr;
}




bool MPI_Quantized_CUDAAllreduce::Enabled(
    const horovod::common::ParameterManager& param_manager,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    const horovod::common::Response& response) const {
  if (global_state_->quantization_bits >= 32) {
    return false;
  }
  return CUDAAllreduce::Enabled(param_manager, entries, response);
}

Status MPI_Quantized_CUDAAllreduce::AllocateExtraBuffers(
    const std::vector<horovod::common::TensorTableEntry>& entries) {
  auto accessData = [&first_entry](std::shared_ptr<PersistentBuffer> buf) {
    return const_cast<void*>(buf->AccessData(first_entry.context));
  };
  bufferManager.InitializeBuffers(entries);


}


MPI_Quantized_CUDAAllreduce::MPI_Quantized_CUDAAllreduce(MPIContext* mpi_context,
                                                         CUDAContext* cuda_context,
                                                         HorovodGlobalState* global_state)
    : CUDAAllreduce(cuda_context, global_state),
      mpi_context_(mpi_context) {

}

Status MPI_Quantized_CUDAAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];
//  std::cerr << "Quantized All reduce\n";

  //printf("%d\n", global_state_->quantization_bits);
  auto start = now();
  //printf("Communication at start %lld\n", global_state_->allreduce_time);

  InitCUDA(entries);

//  MPI_Datatype dtype;
  
  int rank, num_nodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  int64_t tensor_fusion_threshold = global_state_->param_manager.TensorFusionThresholdBytes();
//  std::cerr << rank << " " << num_nodes << " " << tensor_fusion_threshold << std::endl;

  //printf("%d\n", global_state_->quantization_bits);
  int bits = global_state_->quantization_bits; // the amount of bits per value, should be 1, 2, 4 or 8
  int entries_per_byte = 8 / bits;
  int64_t num_elements = NumElements(entries);
  int64_t chunk_size = (int64_t) ceil(1.0 * tensor_fusion_threshold / num_nodes);
  char *tf_buf = (char *)accessData(global_state_->fusion_buffer.
      GetBuffer(first_entry.device, first_entry.context->framework()));

  //printf("Before\n");
//  for (auto entry : entries) {
    //GPU_print((float*)entry.output->data(), entry.output->size() / sizeof(float), cuda_context_->streams[first_entry.device]);
//  }

  //printf("Entries %d\n", entries.size());
//  int64_t sum_size = 0;
//  for (auto entry : entries) {
//    sum_size += entry.output->size();
//  }

//  if (sum_size > tensor_fusion_threshold)
//    printf("Size: %d, threshold: %d\n", sum_size, tensor_fusion_threshold);
         
  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    printf("Start MemcpyInFusionBuffer\n");
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    printf("End MemcpyInFusionBuffer\n");

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[first_entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.tensor->data();
    buffer_len = (size_t) first_entry.tensor->size();
  }
  printf("Copied into fusion buffer\n");


  int num_divisions = (buffer_len + tensor_fusion_threshold - 1) / tensor_fusion_threshold;
  int num_elements_division = 0;

  for (int division = 0; division < num_divisions; division++) {
    if (entries.size() > 1) {
      num_elements_division = num_elements;
    } else {
      // There was the bug
      num_elements_division = division == num_divisions - 1 && buffer_len % tensor_fusion_threshold != 0 ?
        (buffer_len % tensor_fusion_threshold) / sizeof(float) : tensor_fusion_threshold / sizeof(float);
    }

    int division_offset = division * (tensor_fusion_threshold / sizeof(float));
    int num_elems_per_node = num_elements_division / num_nodes;
    int residue = num_elements_division % num_nodes;
    int start_elem = num_elems_per_node * rank + std::min(residue, rank);
    int num_elems = num_elems_per_node + (rank < residue ? 1 : 0);
    int num_buckets = (num_elems + bucket_size - 1) / bucket_size;

    //printf("Ready to prepare first round\n");
    std::vector<MPI_Request> request_reduce;
    int counter1 = 0;
    for (int i = 0; i < num_nodes; i++) {
      if (i == rank) { //for each node(chunk)
        continue;
      }
      //get start offset and length of this chunk
      int start_offset = (num_elems_per_node * i) + std::min(residue, i);
      int length = num_elems_per_node + ((i < residue) ? 1 : 0);

      //find max and min of this chunk
      GPU_find_max_and_min_bucket((float*)buffer_data + division_offset + start_offset, maxandmin_send[counter1],
        length, bucket_size, cuda_context_->streams[first_entry.device]);
      printf("Max and min\n");
      //GPU_print(maxandmin_send[counter1], 10, cuda_context_->streams[first_entry.device]);

      //quantize this chunk
      GPU_quantize_value_bits(quantized_gradients[counter1], (float*)buffer_data + division_offset + start_offset,
        maxandmin_send[counter1], length, bits, bucket_size, cuda_states, cuda_context_->streams[first_entry.device]);
      printf("Quantize values\n");
      counter1++;
    }

    auto mpi_start = now();
    counter1 = 0;
    for (int i = 0; i < num_nodes; i++) {
      if (i == rank) { //for each node(chunk)
        continue;
      }
      //get length of this chunk
      int length = num_elems_per_node + ((i < residue) ? 1 : 0);
      int nb = (length + bucket_size - 1) / bucket_size;

      request_reduce.push_back(MPI_Request());
      MPI_Irecv(quantized_gradients_recv[counter1], (num_elems + entries_per_byte - 1) / entries_per_byte,
        MPI_UNSIGNED_CHAR, i, 0,
        MPI_COMM_WORLD, &request_reduce.back());

      request_reduce.push_back(MPI_Request());
      MPI_Irecv(maxandmin_recv[counter1], num_buckets * 2, mpi_context_->GetMPIDataType(first_entry.tensor),
        i, 0, MPI_COMM_WORLD, &request_reduce.back());

      request_reduce.push_back(MPI_Request());
      MPI_Isend(quantized_gradients[counter1], (length + entries_per_byte - 1) / entries_per_byte, 
        MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &request_reduce.back());

      // There was a bug with num_buckets
      request_reduce.push_back(MPI_Request());
      MPI_Isend(maxandmin_send[counter1], nb * 2, mpi_context_->GetMPIDataType(first_entry.tensor),
        i, 0, MPI_COMM_WORLD, &request_reduce.back());

      counter1++;
    }

    MPI_Waitall((int)request_reduce.size(), &request_reduce[0], MPI_STATUSES_IGNORE);
    auto mpi_end = now();
    global_state_->communication_time += mpi_end - mpi_start;
    //printf("First phase is finished\n");

    for (int i = 0; i < num_nodes - 1; i++) {
      //dequantization
      GPU_dequantize_value_bits(quantized_gradients_recv[i], maxandmin_recv[i],
        dequan_buffer, num_elems, bits, bucket_size, cuda_context_->streams[first_entry.device]);

      //add dequantized value to right place of data_buffer 
      GPU_add(num_elems, dequan_buffer, (float*)buffer_data + division_offset + start_elem,
        cuda_context_->streams[first_entry.device]);
    }

    // Quantize the sum into quantized_gradients_recv[0] and maxandmin_recv[0]
    GPU_find_max_and_min_bucket((float*)buffer_data + division_offset + start_elem, maxandmin_recv[0], num_elems, bucket_size,
      cuda_context_->streams[first_entry.device]);
    GPU_quantize_value_bits(quantized_gradients_recv[0], (float*)buffer_data + division_offset + start_elem, 
      maxandmin_recv[0], num_elems, bits, bucket_size, cuda_states, cuda_context_->streams[first_entry.device]);

    // Dequantize the sum in place of the sum itself. This is necessary for models to be the same.
    GPU_dequantize_value_bits(quantized_gradients_recv[0], maxandmin_recv[0], 
      dequan_buffer, num_elems, bits, bucket_size, cuda_context_->streams[first_entry.device]);

    cudaMemcpy((void*)((float*)buffer_data + division_offset + start_elem),
               (const void*) dequan_buffer,
               num_elems * sizeof(float),
               cudaMemcpyDeviceToDevice);
//    GPU_copy_value((float*)buffer_data + division_offset + start_elem, dequan_buffer, num_elems, 
//      cuda_context_->streams[first_entry.device]);
    //printf("Dequantization and quantization of the sum\n");
//    printf("Copied with %d result\n", cuda_result);

    //second round of MPI communication. receive the sums from other nodes
    mpi_start = now();
    int counter2 = 0;
    std::vector<MPI_Request> request_gather;
    for (int i = 0; i < num_nodes; i++) {
      if (i == rank) {
        continue;
      }
      int length = num_elems_per_node + ((i < residue) ? 1 : 0);
      int nb = (length + bucket_size - 1) / bucket_size;

      request_gather.push_back(MPI_Request());
      MPI_Irecv(quantized_gradients[counter2], (length + entries_per_byte - 1) / entries_per_byte,
        MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &request_gather.back());

      request_gather.push_back(MPI_Request());
      MPI_Irecv(maxandmin_send[counter2], nb * 2, mpi_context_->GetMPIDataType(first_entry.tensor),
        i, 0, MPI_COMM_WORLD, &request_gather.back());

      request_gather.push_back(MPI_Request());
      MPI_Isend(quantized_gradients_recv[0], (num_elems + entries_per_byte - 1) / entries_per_byte,
        MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &request_gather.back());

      request_gather.push_back(MPI_Request());
      MPI_Isend(maxandmin_recv[0], num_buckets * 2, mpi_context_->GetMPIDataType(first_entry.tensor),
        i, 0, MPI_COMM_WORLD, &request_gather.back());

      counter2++;
    }

    MPI_Waitall((int)request_gather.size(), &request_gather[0], MPI_STATUSES_IGNORE);
    mpi_end = now();
    global_state_->communication_time += mpi_end - mpi_start;
    printf("Second phase is finished\n");

    //dequantization
    int counter3 = 0;
    for (int i = 0; i < num_nodes; i++) {
      if (i == rank) {
        continue;
      }
      int start_offset = (num_elems_per_node * i) + std::min(residue, i);
      int length = num_elems_per_node + ((i < residue) ? 1 : 0);

      //dequantize chunk from node i in dequan_buffer
      GPU_dequantize_value_bits(quantized_gradients[counter3], maxandmin_send[counter3], 
        dequan_buffer, length, bits, bucket_size, cuda_context_->streams[first_entry.device]);

      //copy dequantized data to right place of data_buffer
      cudaMemcpy((void*)((float*)buffer_data + division_offset + start_offset),
                 (const void*) dequan_buffer,
                 length * sizeof(float),
                 cudaMemcpyDeviceToDevice);
      counter3++;
    }
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  } else {
    if (first_entry.tensor->data() != first_entry.output->data()) {
      cudaMemcpy((void*)first_entry.output->data(),
                 (const void*) buffer_data,
                 first_entry.output->size(),
                 cudaMemcpyDeviceToDevice);
    }
  }

  //printf("Communication at end %lld\n", global_state_->allreduce_time);
  auto end = now();
  auto elapsed = end - start;
  //printf("end-start %lld\n", elapsed);
  //printf("globalstate communication time %lld\n", global_state_->allreduce_time);
  global_state_->allreduce_time += elapsed;
  //printf("Communication at end 2 %lld\n", global_state_->allreduce_time);
  //printf("After\n");
//  for (auto entry : entries) {
//    GPU_print((float*)entry.output->data(), entry.output->size() / sizeof(float), cuda_context_->streams[first_entry.device]);
//  }
  printf("End of func\n");
  return Status::OK();
}

} // namespace common
} // namespace horovod
