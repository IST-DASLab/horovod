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

#include "cuda_operations.h"
#include "cuda_functions.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include "../mpi_context.h"
#include "../common.h"

namespace horovod {
namespace common {


class ExtraBufferManager {
public:
  ExtraBufferManager(HorovodGlobalState* global_state):global_state_(global_state){}

  Status InitializeBuffers(const std::vector<horovod::common::TensorTableEntry>& entries);

  enum {
    DEQUAN,
    CUDA_STATES,
    MAXMIN_SEND,
    MAXMIN_RECV,
    QUAN_SEND,
    QUAN_RECV,
  };

  template <int bufIndex>
  void *getBuffer();
  template <int bufIndex, typename T>
  std::vector<T>& getvBuffer();
private:
  using PB_ptr = std::shared_ptr<PersistentBuffer>;
  using VPB_ptr = std::vector<PB_ptr>;
  using tuplePBufs = std::tuple<PB_ptr,PB_ptr,VPB_ptr,VPB_ptr,VPB_ptr,VPB_ptr>
  using tupleBufs = std::tuple<float *, curandState*, float *, float *, unsigned char*, unsigned char *>;
  // Map of buffers used for quantization indexed by framework.
  // This operation only works with gpu, so no need to index by device.
  std::map<Framework, std::pair<tuplePBufs, tupleBufs>> extra_buffers_;
  HorovodGlobalState* global_state_;
  tupleBufs bufs_;
};

class MPI_Quantized_CUDAAllreduce : public CUDAAllreduce {
public:
  MPI_Quantized_CUDAAllreduce(MPIContext* mpi_context, CUDAContext* cuda_context, HorovodGlobalState* global_state);
  virtual ~MPI_Quantized_CUDAAllreduce()=default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  Status AllocateExtraBuffers(const std::vector<horovod::common::TensorTableEntry>& entries);
protected:
  MPIContext* mpi_context_;

private:
  const int bucket_size = 512; // the size of the bucket, should be the power of two and does not exceed 1024
  float *dequan_buffer = nullptr;
  std::shared_ptr<PersistentBuffer> dequan_buffer_pb;
  std::shared_ptr<PersistentBuffer> cuda_states_pb;
  curandState* cuda_states = nullptr;

  ExtraBufferManager bufferManager;

  std::vector<std::shared_ptr<PersistentBuffer>> maxandmin_send_buf;
  std::vector<std::shared_ptr<PersistentBuffer>> maxandmin_recv_buf;
  std::vector<std::shared_ptr<PersistentBuffer>> quantized_gradients_buf;
  std::vector<std::shared_ptr<PersistentBuffer>> quantized_gradients_recv_buf;

  std::vector<float *> maxandmin_send;
  std::vector<float *> maxandmin_recv;
  std::vector<unsigned char *> quantized_gradients;
  std::vector<unsigned char *> quantized_gradients_recv;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_QUANTIZED_CUDA_OPERATIONS_H
