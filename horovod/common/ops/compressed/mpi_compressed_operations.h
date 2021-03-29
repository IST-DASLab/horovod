#ifndef HOROVOD_MPI_COMPRESSED_OPERATIONS_H
#define HOROVOD_MPI_COMPRESSED_OPERATIONS_H

#include "../mpi_operations.h"
#include "reducers/reducer.h"

namespace horovod {
namespace common {

class MPI_CompressedAllReduce: public MPIAllreduce {
public:
  MPI_CompressedAllReduce(MPIContext *mpi_context, GPUContext* gpu_context,
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
  bool EnabledName(const std::string& name) const;

  bool GlobalEnabled(const ParameterManager& param_manager) const override;
protected:
  MPIReducer *reducer_;
  Compressor *compressor_;
};


} // namespace common
} // namespace horovod

#endif // HOROVOD_MPI_COMPRESSED_OPERATIONS_H
