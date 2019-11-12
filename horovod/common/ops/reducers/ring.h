#ifndef HOROVOD_RING_H
#define HOROVOD_RING_H
#include "compressed_reducer.h"
namespace horovod {
namespace common {

class MPI_CUDARingReducer: public MPI_CUDACompressedReducer {
public:
  MPI_CUDARingReducer(MPIContext* mpi_context,
  CUDAContext* cuda_context,
      HorovodGlobalState* global_state);

  bool Enabled(const ParameterManager& param_manager,
               const TensorTableEntry& entry,
               const Response& response) const override;

  Status AllreduceDivision(void* sendbuf, void* recvbuf, int num_elements,
                           MPI_Comm comm, std::vector<TensorTableEntry>& entries,
                           int64_t glovbal_offset) override;

protected:
  Status Init(const std::vector<horovod::common::TensorTableEntry>& entries, int world_size) override;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_RING_H
