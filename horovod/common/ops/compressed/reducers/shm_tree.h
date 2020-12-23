#ifndef HOROVOD_SHM_TREE_H
#define HOROVOD_SHM_TREE_H
#include "reducer.h"

namespace horovod {
namespace common {

class SHM_Allreduce_Tree : public SHMReducer {
public:
  SHM_Allreduce_Tree(MPIContext* mpi_context, GPUContext* gpu_context,
                     HorovodGlobalState* global_state, Compressor* compressor,
                     CommunicatorType comm_type);
  Status AllreduceDivision(int num_elements,
                           std::vector<TensorTableEntry>& entries,
                           unsigned char* buffer_ptr) override;

  Status Init(const std::vector<TensorTableEntry>& entries,
              MPI_Comm comm) override;
  size_t GetRequiredFreeSize() override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_SHM_TREE_H
