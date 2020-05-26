#ifndef HOROVOD_RING_H
#define HOROVOD_RING_H
#include "reducer.h"

namespace horovod {
namespace common {

class MPI_Allreduce_Ring : public MPIReducer {
public:
  MPI_Allreduce_Ring(MPIContext* mpi_context, HorovodGlobalState* global_state,
                     Compressor* compressor, Summator* summator);

  Status AllreduceDivision(int num_elements, MPI_Comm comm,
                           std::vector<TensorTableEntry>& entries,
                           int64_t glovbal_offset) override;

  Status Init(const std::vector<TensorTableEntry>& entries) override;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_RING_H