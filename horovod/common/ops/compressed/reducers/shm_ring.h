#ifndef HOROVOD_SHM_RING_H
#define HOROVOD_SHM_RING_H
#include "reducer.h"

namespace horovod {
namespace common {

struct SHM_Allreduce_Ring : public SHMReducer {
  SHM_Allreduce_Ring(MPIContext* mpi_context, GPUContext* gpu_context,
                     HorovodGlobalState* global_state, Compressor* compressor,
                     CommunicatorType comm_type);
  Status AllreduceDivision(int num_elements,
                           std::vector<TensorTableEntry>& entries,
                           int64_t global_offset) override;

  Status Init(const std::vector<TensorTableEntry>& entries,
              MPI_Comm comm) override;
  size_t GetRequiredFreeSize() override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_SHM_RING_H
