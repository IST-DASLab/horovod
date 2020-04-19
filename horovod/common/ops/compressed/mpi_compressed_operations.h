#ifndef HOROVOD_MPI_COMPRESSED_OPERATIONS_H
#define HOROVOD_MPI_COMPRESSED_OPERATIONS_H

#include "../mpi_operations.h"
#include "reducers/reducer.h"

namespace horovod {
namespace common {

const int BUFFER_THRESHOLD = 100;
const float QUANTIZE_MULTIPLIER = 0.5;

class MPI_CompressedAllReduce: public MPIAllreduce {
public:
  MPI_CompressedAllReduce(MPIContext *mpi_context,
                             HorovodGlobalState *global_state);
  virtual ~MPI_CompressedAllReduce();
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;
  Status Allreduce(int num_elements,
                   MPI_Comm comm, std::vector<TensorTableEntry>& entries,
                   int buffer_len);
  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
protected:
  MPIReducer *mpiReducer;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MPI_COMPRESSED_OPERATIONS_H
