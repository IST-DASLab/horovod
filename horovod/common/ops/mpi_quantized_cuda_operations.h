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

#ifndef HOROVOD_MPI_QUANTIZED_CUDA_OPERATIONS_H
#define HOROVOD_MPI_QUANTIZED_CUDA_OPERATIONS_H

#include "cuda_functions.h"
#include "cuda_operations.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../common.h"
#include "../mpi_context.h"

namespace horovod {
namespace common {

class MPI_Quantized_CUDAAllreduce : public CUDAAllreduce {
public:
  MPI_Quantized_CUDAAllreduce(MPIContext* mpi_context,
                              CUDAContext* cuda_context,
                              HorovodGlobalState* global_state);
  virtual ~MPI_Quantized_CUDAAllreduce() = default;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  Status Init(const std::vector<horovod::common::TensorTableEntry>& entries);

  bool Enabled(const ParameterManager& param_manager,
               const TensorTableEntry& entry,
               const Response& response) const override;
  bool Packed(const ParameterManager& param_manager,
              const TensorTableEntry& entry, const Response& response,
              const TensorTableEntry& new_entry,
              const Response & new_response) const override;
  bool AcceptableEntry(const TensorTableEntry& entry) const;
protected:
  MPIContext* mpi_context_;

  int quantize_bucket_size; // the size of the bucket, should be the
                            // power of two and does not exceed 1024
  int64_t quantize_threshold;
  int64_t maxmin_size;
  int64_t quantized_buffer_size;
  int64_t tensor_fusion_threshold;
  int64_t chunk_size;
  int num_elems_in_chunk;

  FusionBufferManager bufferManager;

  float* dequan_buffer = nullptr;
  curandState* cuda_states = nullptr;

  float* maxandmin_send;
  float* maxandmin_recv;
  unsigned char* quantized_gradients_send;
  unsigned char* quantized_gradients_recv;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MPI_QUANTIZED_CUDA_OPERATIONS_H
