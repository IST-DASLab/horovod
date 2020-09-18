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

#ifndef HOROVOD_SCATTER_BROADCAST_H
#define HOROVOD_SCATTER_BROADCAST_H

#include "reducer.h"

namespace horovod {
namespace common {

class MPI_Allreduce_ScatterReduceAllgather : public MPIReducer {
public:
  MPI_Allreduce_ScatterReduceAllgather(MPIContext* mpi_context,
                                       HorovodGlobalState* global_state,
                                       Compressor* compressor,
                                       Summator* summator);

  Status AllreduceDivision(int num_elements, MPI_Comm comm,
                           std::vector<TensorTableEntry>& entries,
                           int64_t global_offset) override;

  Status Init(const std::vector<TensorTableEntry>& entries) override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_SCATTER_BROADCAST_H
