#ifndef HOROVOD_ALL_BROADCAST_H
#define HOROVOD_ALL_BROADCAST_H

#include "reducer.h"

namespace horovod {
namespace common {

struct MPI_Allreduce_AllGather : public MPIReducer {
  MPI_Allreduce_AllGather(MPIContext* mpi_context, GPUContext* gpu_context,
                          HorovodGlobalState* global_state,
                          Compressor* compressor, Summator* summator);
  Status AllreduceDivision(int num_elements,
                           std::vector<TensorTableEntry>& entries,
                           int64_t global_offset) override;
  Status Init(const std::vector<TensorTableEntry>& entries,
              MPI_Comm comm) override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_ALL_BROADCAST_H