#ifndef HOROVOD_COMPRESSED_REDUCER_H
#define HOROVOD_COMPRESSED_REDUCER_H

#include "compression/compressor.h"
#include "compression/error_feedback.h"
#include "reducers/reducer.h"
#include "../mpi_gpu_operations.h"

namespace horovod {
namespace common {


const int BUFFER_THRESHOLD = 1;
const float QUANTIZE_MULTIPLIER = 0.5;

class MPI_GPUCompressedAllReduce: public MPI_GPUAllreduce {
public:

  MPI_GPUCompressedAllReduce(MPIContext *mpi_context,
      GPUContext *gpu_context, HorovodGlobalState *global_state);
  virtual ~MPI_GPUCompressedAllReduce()=default;
  Status Execute(std::vector<TensorTableEntry>& entries,
      const Response& response) override;
  Status Allreduce(void* sendbuf, void* recvbuf, int num_elements,
                   MPI_Comm comm, std::vector<TensorTableEntry>& entries,
                   int buffer_len);
  bool Enabled(const ParameterManager& param_manager,
                             const std::vector<TensorTableEntry>& entries,
                             const Response& response) const override;
protected:
  MPIReducer *mpiReducer;
};

} // common
} // horovod
#endif // HOROVOD_COMPRESSED_REDUCER_H
