#ifndef HOROVOD_ALL_BROADCAST_H
#define HOROVOD_ALL_BROADCAST_H

#include "../../common.h"
#include "../../compression/compressor.h"
#include "../../global_state.h"
#include "compressed_reducer.h"

namespace horovod {
namespace common {

struct MPI_CUDAAllBroadcastReducer: public MPI_CUDACompressedReducer {
  MPI_CUDAAllBroadcastReducer(MPIContext* mpi_context, CUDAContext* cuda_context,
                  HorovodGlobalState* global_state);
  Status AllreduceDivision(void* sendbuf, void* recvbuf, int num_elements,
                   MPI_Comm comm, std::vector<TensorTableEntry>& entries,
                   int64_t global_offset) override;
  bool Enabled(
      const ParameterManager& param_manager,
      const TensorTableEntry& entry,
      const Response& response) const override;
  Status NonCompressedInit(std::vector<TensorTableEntry>& entries, int num_elements, int world_size);
  Status NonCompressed_Allreduce(void* sendbuf, void* recvbuf, int num_elements,
                          MPI_Comm comm, std::vector<TensorTableEntry>& entries, int buffer_len);
  virtual bool Packed(const ParameterManager& param_manager,
                      const TensorTableEntry& entry, const Response& response,
                      const TensorTableEntry& new_entry, const Response & new_response) const override ;
protected:
  virtual Status Init(const std::vector<TensorTableEntry>& entries, int world_size) override;

};

} // namespace common
} // namespace horovod

#endif //HOROVOD_ALL_BROADCAST_H