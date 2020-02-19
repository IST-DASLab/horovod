#ifndef HOROVOD_ALL_BROADCAST_H
#define HOROVOD_ALL_BROADCAST_H

#include "reducer.h"

namespace horovod {
namespace common {

struct MPI_GPUAllreduce_AllBroadcast : public MPIReducer {
  MPI_GPUAllreduce_AllBroadcast(MPIContext* mpi_context,
                                GPUContext* gpu_context,
                                HorovodGlobalState* global_state,
                                Compressor* compressor, Summator* summator);
  Status AllreduceDivision(int num_elements,
                           MPI_Comm comm,
                           std::vector<TensorTableEntry>& entries,
                           int64_t global_offset) override;
  Status Init(const std::vector<TensorTableEntry>& entries) override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_ALL_BROADCAST_H