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
#ifndef HOROVOD_QUANTIZATION_H
#define HOROVOD_QUANTIZATION_H


#include "../common.h"
#include "../global_state.h"
#include "mpi_quantized_cuda_operations.h"

namespace horovod {
namespace common {

struct SimpleQuantizer: public MPI_Quantized_CUDAAllreduce {
  SimpleQuantizer(MPIContext* mpi_context, CUDAContext* cuda_context,
                  HorovodGlobalState* global_state);
  Status Init(std::vector<TensorTableEntry>& entries, int world_size);
  int MPI_Quantized_Allreduce(void* sendbuf, void* recvbuf, int count,
                              MPI_Comm comm, std::vector<TensorTableEntry>& entries, int buffer_len) override;
  bool Enabled(
      const ParameterManager& param_manager,
      const TensorTableEntry& entry,
      const Response& response) const;
  Status Execute(
      std::vector<horovod::common::TensorTableEntry>& entries,
      const horovod::common::Response& response);

  float* dequan_buffer1 = nullptr;
  CurandState* cuda_states1 = nullptr;

  unsigned char* quantized_gradients_send1;
  unsigned char* quantized_gradients_recv1;

};

} // namespace common
} // namespace horovod

#endif //HOROVOD_QUANTIZATION_H