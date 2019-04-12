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

MPI_Quantized_CUDAAllreduce::MPI_Quantized_CUDAAllreduce(MPIContext* mpi_context,
                                                         CUDAContext* cuda_context,
                                                         HorovodGlobalState* global_state)
    : CUDAAllreduce(cuda_context, global_state),
      mpi_context_(mpi_context) {}

Status MPI_Quantized_CUDAAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];
//  std::cerr << "Quantized All reduce!!!\n";

  //printf("%d\n", global_state_->quantization_bits);
  auto start = now();
  //printf("Communication at start %lld\n", global_state_->allreduce_time);

  InitCUDA(entries);

  if (global_state_->quantization_bits == 32) {
    void* buffer_data;
    size_t buffer_len;
    int64_t num_elements = NumElements(entries);

    // Copy memory into the fusion buffer.
    auto& timeline = global_state_->timeline;
    if (entries.size() > 1) {
      timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
      const void* fused_input_data;
      MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

      auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
      cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

      timeline.ActivityEndAll(entries);
    } else {
      buffer_data = (void*) first_entry.output->data();
      buffer_len = (size_t) first_entry.output->size();
    }

    // Do allreduce.
    timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
    const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                          ? MPI_IN_PLACE : first_entry.tensor->data();
    auto mpi_start = now();
    int op = MPI_Allreduce(sendbuf, buffer_data,
                           (int) num_elements,
                           mpi_context_->GetMPIDataType(first_entry.tensor),
                           mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                           mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
    auto mpi_end = now();
    global_state_->communication_time += mpi_end - mpi_start;
    if (op != MPI_SUCCESS) {
      throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
    }
    timeline.ActivityEndAll(entries);

    // Copy memory out of the fusion buffer.
    if (entries.size() > 1) {
      timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
      MemcpyOutFusionBuffer(buffer_data, entries);

      auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
      cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

      timeline.ActivityEndAll(entries);
    }
    auto end = now();
    global_state_->allreduce_time += end - start;

    return Status::OK();
  }

//  MPI_Datatype dtype;
  
  int rank, numNodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numNodes);

  int64_t tensor_fusion_threshold = global_state_->param_manager.TensorFusionThresholdBytes();
//  std::cerr << rank << " " << num_nodes << " " << tensor_fusion_threshold << std::endl;

  //printf("%d\n", global_state_->quantization_bits);
  int bits = global_state_->quantization_bits; // the amount of bits per value, should be 1, 2, 4 or 8
  int bucket_size = 512; // the size of the bucket, should be the power of two and does not exceed 1024
  int entries_per_byte = 8 / bits;
  int64_t num_elements = NumElements(entries);
  int64_t chunk_size = (int64_t) ceil(1.0 * tensor_fusion_threshold / numNodes);

  if (dequan_buffer == nullptr) {
    //printf("Dequan buffer is null and its size %d\n", chunk_size);
    cudaMalloc(&dequan_buffer, chunk_size);
    cuda_states = GPU_init_curand(ceil((double)chunk_size / sizeof(float)), time(NULL), cuda_context_->streams[first_entry.device]);
    for (int i = 0; i < numNodes - 1; i++) {
      // allocate memory for this chunk. We partition the entire gradients into N chunks, where N is the number of MPI nodes.
      float* maxandmin_per_node;
      float* maxandmin_per_node_recv; 
      unsigned char* quantized_gradients_per_node;
      unsigned char* quantized_gradients_per_node_recv;

      //calculate how many buckets in each chunk. Then each bucket has 2 float values: maximum and minimum.
      cudaMalloc(&maxandmin_per_node, ceil(1.0 * chunk_size / (bucket_size * sizeof(float))) * 2 * sizeof(float));
      cudaMalloc(&maxandmin_per_node_recv, ceil(1.0 * chunk_size / (bucket_size * sizeof(float))) * 2 * sizeof(float));

      //We quantize values from 32 bits to <bits> bits, so the size of quantized chunk is <bits> / 32 of full precision chunk.
      cudaMalloc(&quantized_gradients_per_node, ceil(1.0 * chunk_size * bits / 32));
      cudaMalloc(&quantized_gradients_per_node_recv, ceil(1.0 * chunk_size * bits / 32));

      maxandmin_send.push_back(maxandmin_per_node);
      quantized_gradients.push_back(quantized_gradients_per_node);
      maxandmin_recv.push_back(maxandmin_per_node_recv);
      quantized_gradients_recv.push_back(quantized_gradients_per_node_recv);
    }
    //printf("Allocation is successful\n");
    //printf("%d\n", dequan_buffer);
    //printf("Num nodes %d\n", num_nodes);
    //printf("max and min size: %d\n", (int)ceil(1.0 * chunk_size / (bucket_size * sizeof(float))) * 2 * sizeof(float));
    //printf("quantized gradients size: %d\n", (int)ceil(1.0 * chunk_size * bits / 32));
  }

  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    //printf("Start MemcpyInFusionBuffer\n");
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    //printf("End MemcpyInFusionBuffer\n");

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[first_entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.tensor->data();
    buffer_len = (size_t) first_entry.tensor->size();
  }
  //printf("Copied into fusion buffer\n");

  int num_divisions = (buffer_len + tensor_fusion_threshold - 1) / tensor_fusion_threshold;
  int num_elements_division = 0;

  for (int division = 0; division < num_divisions; division++) {
    if (entries.size() > 1) {
      num_elements_division = num_elements;
    } else {
      // There was the bug
        num_elements_division = (division == (num_divisions - 1)) ? 
        (buffer_len % tensor_fusion_threshold) / sizeof(float) : tensor_fusion_threshold / sizeof(float);
    }

      int division_offset = division * (tensor_fusion_threshold / sizeof(float));
      //partition the data_buffer into chunks
      int numElemsPerNode = num_elements_division / numNodes;
      int residue = num_elements_division % numNodes;
      int startElem = (numElemsPerNode * rank) + std::min(residue, rank);
      int numElems = numElemsPerNode + ((rank < residue) ? 1 : 0);
      int numBuckets = ceil(numElemsPerNode / (double)bucket_size); 

      //first round of MPI communication
      std::vector<MPI_Request> request_reduce;
      int counter1 = 0;
      for(int i = 0; i < numNodes; i++)
      {
        if(i != rank) //for each node(chunk)
        {
          //get start offset and length of this chunk
          int start_offset = (numElemsPerNode * i) + std::min(residue, i);
          int length = numElemsPerNode + ((i < residue) ? 1 : 0);

          request_reduce.push_back(MPI_Request());
          MPI_Irecv(quantized_gradients_recv[counter1], numElems, MPI_UNSIGNED_CHAR, i, 0, 
            MPI_COMM_WORLD, &request_reduce.back());

          request_reduce.push_back(MPI_Request());
          MPI_Irecv(maxandmin_recv[counter1], numBuckets * 2, mpi_context_->GetMPIDataType(first_entry.tensor),
            i, 0, MPI_COMM_WORLD, &request_reduce.back());

          //find max and min of this chunk
          GPUFindMaxAndMin((float*)buffer_data + division_offset + start_offset, maxandmin_send[counter1], 
            length, cuda_context_->streams[first_entry.device]);
          
          //quantize this chunk
          GPUQuantizeValue(quantized_gradients[counter1], (float*)buffer_data + division_offset + start_offset, 
            maxandmin_send[counter1], length, cuda_states, cuda_context_->streams[first_entry.device]);
          
          request_reduce.push_back(MPI_Request());
          MPI_Isend(quantized_gradients[counter1], numElemsPerNode + ((i < residue) ? 1 : 0), 
            MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &request_reduce.back());

          request_reduce.push_back(MPI_Request());
          MPI_Isend(maxandmin_send[counter1], numBuckets * 2, mpi_context_->GetMPIDataType(first_entry.tensor),
            i, 0, MPI_COMM_WORLD, &request_reduce.back());
          
          counter1++;

        }
      }
      
      MPI_Waitall((int)request_reduce.size(), &request_reduce[0], MPI_STATUSES_IGNORE);

      
      for(int i = 0; i < (numNodes - 1); i++)
      {
        //dequantization
        GPUDequantizeValue(quantized_gradients_recv[i], maxandmin_recv[i], 
          dequan_buffer, numElems, cuda_context_->streams[first_entry.device]);
        
        //add dequantized value to right place of data_buffer 
        GPU_add(numElems, dequan_buffer, (float*)buffer_data + division_offset + startElem, 
          cuda_context_->streams[first_entry.device]); 
      }
      
      //quantize aggregated value  
      GPUFindMaxAndMin((float*)buffer_data + division_offset + startElem, maxandmin_recv[0], numElems, 
        cuda_context_->streams[first_entry.device]);

      GPUQuantizeValue(quantized_gradients_recv[0], (float*)buffer_data + division_offset + startElem, 
        maxandmin_recv[0], numElems, cuda_states, cuda_context_->streams[first_entry.device]);

      //second round of MPI communication
      int counter2 = 0;
      std::vector<MPI_Request> request_gather;
      for(int i = 0; i < numNodes; i++)
      {
        if (i != rank)
        {
          request_gather.push_back(MPI_Request());
          MPI_Irecv(quantized_gradients[counter2], numElemsPerNode + ((i < residue) ? 1 : 0), 
            MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &request_gather.back());

          request_gather.push_back(MPI_Request());
          MPI_Irecv(maxandmin_send[counter2], numBuckets * 2, mpi_context_->GetMPIDataType(first_entry.tensor),
            i, 0, MPI_COMM_WORLD, &request_gather.back());  

          request_gather.push_back(MPI_Request());
          MPI_Isend(quantized_gradients_recv[0], numElems, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &request_gather.back());

          request_gather.push_back(MPI_Request());
          MPI_Isend(maxandmin_recv[0], numBuckets * 2, mpi_context_->GetMPIDataType(first_entry.tensor),
            i, 0, MPI_COMM_WORLD, &request_gather.back());      

          counter2++;
        }
      }

      MPI_Waitall((int)request_gather.size(), &request_gather[0], MPI_STATUSES_IGNORE);

    //dequantization
      //dequantization
      int counter3 = 0;
      for(int i = 0; i < numNodes; i++)
      {
        if(i != rank)
        {

          int start_offset = (numElemsPerNode * i) + std::min(residue, i);
          int length = numElemsPerNode + ((i < residue) ? 1 : 0);

          //dequantize chunk from node i in dequan_buffer     
          GPUDequantizeValue(quantized_gradients[counter3], maxandmin_send[counter3], 
            dequan_buffer, length, cuda_context_->streams[first_entry.device]);
          
          //copy dequantized data to right place of data_buffer
          GPU_copy_value((float*)buffer_data + division_offset + start_offset, dequan_buffer, length, 
            cuda_context_->streams[first_entry.device]);
          counter3++;
        }   
      }

    //printf("Third phase is finished\n");
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
      //printf("Memcpy\n");
//      memcpy((void*)first_entry.output->data(), (const void*)first_entry.tensor->data(),
//        first_entry.tensor->size());
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

  return Status::OK();
}

} // namespace common
} // namespace horovod
