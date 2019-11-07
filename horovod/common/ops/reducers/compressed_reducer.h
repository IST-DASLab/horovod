#ifndef HOROVOD_COMPRESSED_REDUCER_H
#define HOROVOD_COMPRESSED_REDUCER_H

#include "../../common.h"
#include "../../compressor.h"
#include "../../mpi_context.h"
#include "../collective_operations.h"
#include "../mpi_cuda_operations.h"
#include "../mpi_operations.h"

namespace horovod {
namespace common {

enum ReductionType {
  AllBroadcast,
  ScatterAllgather,
  Ring,
  // Take an algorithm from horovod (mpi or nccl)
  Horovod
};

class MPI_CUDACompressedReducer: public MPI_CUDAAllreduce {
public:
  MPI_CUDACompressedReducer(MPIContext *mpi_context,
      CUDAContext *cuda_context, HorovodGlobalState *global_state);
  virtual ~MPI_CUDACompressedReducer()=default;
  Status Execute(std::vector<TensorTableEntry>& entries,
      const Response& response) override;
  Status Allreduce(void* sendbuf, void* recvbuf, int num_elements,
                   MPI_Comm comm, std::vector<TensorTableEntry>& entries,
                   int buffer_len);
  virtual Status AllreduceDivision(void* sendbuf, void* recvbuf, int num_elements,
                           MPI_Comm comm, std::vector<TensorTableEntry>& entries,
                           int buffer_len) = 0;
  virtual Status Init(const std::vector<TensorTableEntry>& entries, int world_size) = 0;

protected:
  ReductionType reduction_type;
  Compressor *compressor;

  FusionBufferManager bufferManager;
  unsigned char* gradients_send;
  unsigned char* gradients_recv;
  unsigned char* decompress_buffer;
  int64_t tensor_fusion_threshold;
};

} // common
} // horovod
#endif // HOROVOD_COMPRESSED_REDUCER_H
