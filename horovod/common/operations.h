// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef HOROVOD_OPERATIONS_H
#define HOROVOD_OPERATIONS_H

#include <functional>

#include "common.h"
#define OMPI_SKIP_MPICXX
#include "mpi.h"

namespace horovod {
namespace common {

// The number of elements held by fusion buffer and hierarchical
// allreduce size is always a multiple of FUSION_BUFFER_ATOMIC_UNIT
#define FUSION_BUFFER_ATOMIC_UNIT 64

// Horovod knobs.
#define HOROVOD_MPI_THREADS_DISABLE "HOROVOD_MPI_THREADS_DISABLE"
#define HOROVOD_TIMELINE "HOROVOD_TIMELINE"
#define HOROVOD_TIMELINE_MARK_CYCLES "HOROVOD_TIMELINE_MARK_CYCLES"
#define HOROVOD_AUTOTUNE "HOROVOD_AUTOTUNE"
#define HOROVOD_AUTOTUNE_LOG "HOROVOD_AUTOTUNE_LOG"
#define HOROVOD_FUSION_THRESHOLD "HOROVOD_FUSION_THRESHOLD"
#define HOROVOD_CYCLE_TIME "HOROVOD_CYCLE_TIME"
#define HOROVOD_STALL_CHECK_DISABLE "HOROVOD_STALL_CHECK_DISABLE"
#define HOROVOD_HIERARCHICAL_ALLREDUCE "HOROVOD_HIERARCHICAL_ALLREDUCE"
#define HOROVOD_HIERARCHICAL_ALLGATHER "HOROVOD_HIERARCHICAL_ALLGATHER"
#define HOROVOD_COMPRESSOR "HOROVOD_COMPRESSOR"
#define HOROVOD_REDUCTION "HOROVOD_REDUCTION"
#define HOROVOD_COMPRESSION_BUCKET_SIZE "HOROVOD_COMPRESSION_BUCKET_SIZE"
#define HOROVOD_QUANTIZATION "HOROVOD_QUANTIZATION"
#define HOROVOD_QUANTIZATION_TYPE "HOROVOD_QUANTIZATION_TYPE"
#define HOROVOD_TOPK "HOROVOD_TOPK"

// Check that Horovod is initialized.
Status CheckInitialized();

extern "C" {

// C interface to initialize Horovod.
void horovod_init(int quantization_bits, const int *ranks, int nranks);

// C interface to initialize Horovod with the given MPI communicator.
void horovod_init_comm(int quantization_bits, MPI_Comm comm);

// C interface to shut down Horovod.
void horovod_shutdown();

// C interface to get index of current Horovod process.
// Returns -1 if Horovod is not initialized.
int horovod_rank();

// C interface to get index of current Horovod process in the node it is on.
// Returns -1 if Horovod is not initialized.
int horovod_local_rank();

// C interface to return number of Horovod processes.
// Returns -1 if Horovod is not initialized.
int horovod_size();

// C interface to return number of Horovod processes in the node it is on.
// Returns -1 if Horovod is not initialized.
int horovod_local_size();

// C interface to return flag indicating whether MPI multi-threading is
// supported. Returns -1 if Horovod is not initialized.
int horovod_mpi_threads_supported();

// C interface to return the time spent in nanoseconds on allreduce
// Returns 0 if Horovod is not initialized
int64_t horovod_allreduce_time();

// C interface to return the time spent in nanoseconds onto the communication
// Returns 0 if Horovod is not initialized.
int64_t horovod_communication_time();

int64_t horovod_compression_time();

int64_t horovod_metainfo_time();

}

Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback);

Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback);

Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback);

} // namespace common
} // namespace horovod

#endif // HOROVOD_OPERATIONS_H
