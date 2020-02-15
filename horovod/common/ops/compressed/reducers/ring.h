#ifndef HOROVOD_RING_H
#define HOROVOD_RING_H
#include "reducer.h"

namespace horovod {
namespace common {

class MPI_GPUAllreduce_Ring : public MPIReducer {
public:
  MPI_GPUAllreduce_Ring(MPIContext* mpi_context, GPUContext* gpu_context,
                        HorovodGlobalState* global_state,
                        Compressor* compressor);

  Status AllreduceDivision(void* sendbuf, void* recvbuf, int num_elements,
                           MPI_Comm comm,
                           std::vector<TensorTableEntry>& entries,
                           int64_t glovbal_offset) override;

  Status Init(const std::vector<TensorTableEntry>& entries) override;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_RING_H
