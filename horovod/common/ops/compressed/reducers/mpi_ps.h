#ifndef MPI_PS_H
#define MPI_PS_H

#include "reducer.h"

namespace horovod {
namespace common {

struct MPI_Allreduce_PS : public MPIReducer {
  MPI_Allreduce_PS(MPIContext* mpi_context, GPUContext* gpu_context,
                   HorovodGlobalState* global_state, Compressor* compressor);
  Status AllreduceDivision(int num_elements,
                           std::vector<TensorTableEntry>& entries,
                           unsigned char* buffer_ptr,
                           int global_offset) override;
  Status Init(const std::vector<TensorTableEntry>& entries,
              MPI_Comm comm) override;
  size_t GetRequiredFreeSize() override;
};

} // namespace common
} // namespace horovod

#endif // MPI_PS_H
