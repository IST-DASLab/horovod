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
#ifndef HOROVOD_ALL_BROADCAST_H
#define HOROVOD_ALL_BROADCAST_H

#include "../../common.h"
#include "../../compressor.h"
#include "../../global_state.h"
#include "compressed_reducer.h"

namespace horovod {
namespace common {

struct MPI_CUDAAllBroadcastReducer: public MPI_CUDACompressedReducer {
  MPI_CUDAAllBroadcastReducer(MPIContext* mpi_context, CUDAContext* cuda_context,
                  HorovodGlobalState* global_state);
  Status AllreduceDivision(void* sendbuf, void* recvbuf, int num_elements,
                   MPI_Comm comm, std::vector<TensorTableEntry>& entries,
                   int buffer_len) override;
  bool Enabled(
      const ParameterManager& param_manager,
      const TensorTableEntry& entry,
      const Response& response) const override;
  Status NonCompressedInit(std::vector<TensorTableEntry>& entries, int num_elements, int world_size);
  Status NonCompressed_Allreduce(void* sendbuf, void* recvbuf, int num_elements,
                          MPI_Comm comm, std::vector<TensorTableEntry>& entries, int buffer_len);

protected:
  virtual Status Init(const std::vector<TensorTableEntry>& entries, int world_size) override;

};

} // namespace common
} // namespace horovod

#endif //HOROVOD_ALL_BROADCAST_H