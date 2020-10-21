#ifndef HOROVOD_SHM_UTILS_H
#define HOROVOD_SHM_UTILS_H
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>

namespace horovod {
namespace common{

struct shmComm {
  shmComm(int rank);
  ~shmComm();

  void Init(MPI_Comm mpiComm, const std::vector<int>& ranks, size_t buf_size);

  // Execute a transfer to the remote buffer.
  // Buf: source buffer.
  // buf_size: size in bytes to send.
  // shm_offset: offset in shared memory to avoid race conditions on the same piece of shared memory
  // peer_rank: rank in initial MPI_Comm.
  void memcpy(void* buf, size_t buf_size, size_t shm_offset,
                      int peer_rank, cudaStream_t stream);

  // Manual communication with the remote buffer.
  // The pattern of usage is following
  // Get the pointer with get_sendBuf.
  // Launch the kernel using the remote buffer on stream.
  // Call post_sendBuf on the same stream.
  void* get_sendBuf(int peer_rank);
  // Post data transfer.
  void post_sendBuf(int peer_rank, cudaStream_t stream);

  // Get pointer to the shared memory with notification from sender.
  // non blocking Recv.
  // returns 0 if buffer is ready, 1 otherwise
  int recvBufAsync(void** buf, size_t shm_offset, int peer_rank,
      cudaStream_t stream);

  // blocking Recv
  void recvBuf(void** buf, size_t shm_offset, int peer_rank,
               cudaStream_t stream);

  void waitSendAll();
private:
  struct cudaEventSync {
    cudaEvent_t event;
    cudaIpcEventHandle_t eventHandle;
    MPI_Request request;
  };

  struct shmBuffer {
    int shmSize;
    void* hostMem;
    void* devHostMem;
  };

  // Initialize send and receive resources.
  // Calls to sendInit and recvInit
  // must be separated with MPI_Barrier.
  void sendInit(shmBuffer* resource, int peer_rank,
      size_t shm_size);
  void recvInit(shmBuffer* resource, int peer_rank,
                size_t shm_size);
  // Initialize cudaIPC primitives.
  void initEventSend(cudaEventSync* eventSync, int recv_rank,
                     MPI_Request* request);
  void initEventRecv(cudaEventSync* eventSync, int send_rank);


  static void freeBuffer(shmBuffer* buffer);
  static void freeEventSync(cudaEventSync* eventSend);

  std::unordered_map<int, std::pair<shmBuffer, cudaEventSync>> send_resources;
  std::unordered_map<int, std::pair<shmBuffer, cudaEventSync>> recv_resources;
  int rank_;
  MPI_Comm comm_;
};



} // namespace common
} // namespace horovod

#endif // HOROVOD_SHM_UTILS_H
