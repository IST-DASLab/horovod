#ifndef HOROVOD_COMPRESSED_COMM_H
#define HOROVOD_COMPRESSED_COMM_H
#include <mpi.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"

namespace horovod {
namespace common {

struct Comm {
  Comm(int rank): rank_(rank){}

  // Execute a transfer to the remote buffer.
  // Buf: source buffer.
  // buf_size: size in bytes to send.
  // peer_rank: rank in initial MPI_Comm.
  // offset: offset in internal memory buffers
  // to avoid race conditions on the same piece of internal memory
  virtual void Send(void* buf, size_t buf_size, int peer_rank,
                    cudaStream_t stream, size_t offset = 0) = 0;

  // Get pointer to the memory with notification from sender.
  // non blocking Recv.
  // buf - pointer to the receive buffer.
  // peer_rank: rank in initial MPI_Comm.
  // offset: offset in internal memory buffers
  // returns 0 if buffer is ready, 1 otherwise
  virtual int RecvBufAsync(void** buf, int peer_rank, cudaStream_t stream,
                   size_t offset = 0) = 0;

  // Get pointer to the memory with notification from sender.
  // non blocking Recv.
  virtual void RecvBuf(void** buf, int peer_rank,
                       cudaStream_t stream, size_t offset = 0) = 0;

  virtual void WaitSendAll() = 0;
protected:
  int rank_;
  MPI_Comm comm_;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_COMPRESSED_COMM_H
