#ifndef HOROVOD_MPI_GPU_COMPRESSED_OPERATIONS_H
#define HOROVOD_MPI_GPU_COMPRESSED_OPERATIONS_H

#include "reducers/reducer.h"
#include "../mpi_gpu_operations.h"

namespace horovod {
namespace common {

class MPI_GPUCompressedAllReduce: public MPI_GPUAllreduce {
public:
  MPI_GPUCompressedAllReduce(MPIContext *mpi_context,
      GPUContext *gpu_context, HorovodGlobalState *global_state);
  ~MPI_GPUCompressedAllReduce() override;
  Status Execute(std::vector<TensorTableEntry>& entries,
      const Response& response) override;
  Status Allreduce(int64_t num_elements,
                   MPI_Comm comm, std::vector<TensorTableEntry>& entries,
                   int buffer_len);
  bool Enabled(const ParameterManager& param_manager,
                             const std::vector<TensorTableEntry>& entries,
                             const Response& response) const override;
  bool GlobalEnabled(const ParameterManager& param_manager) const override;
  bool EnabledName(const std::string& name) const override;
protected:
  MPIReducer *reducer_;
  Compressor *compressor_;
};

void SetQuantizationLevels(float* levels, int bits);

} // common
} // horovod
#endif // HOROVOD_MPI_GPU_COMPRESSED_OPERATIONS_H
