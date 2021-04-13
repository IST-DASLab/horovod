#ifndef HOROVOD_MPI_TREE_H
#define HOROVOD_MPI_TREE_H
#include "reducer.h"

namespace horovod {
namespace common {

class MPI_Allreduce_Tree : public MPIReducer {
public:
  MPI_Allreduce_Tree(MPIContext* mpi_context, GPUContext* gpu_context,
                     HorovodGlobalState* global_state, Compressor* compressor);
  Status AllreduceDivision(int num_elements,
                           std::vector<TensorTableEntry>& entries,
                           unsigned char* buffer_ptr, int global_offset) override;
  Status Init(const std::vector<TensorTableEntry>& entries,
              MPI_Comm comm) override;
  size_t GetRequiredFreeSize() override;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_MPI_TREE_H
