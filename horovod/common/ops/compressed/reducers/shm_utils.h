#ifndef HOROVOD_SHM_UTILS_H
#define HOROVOD_SHM_UTILS_H
#include "comm.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <vector>

namespace horovod {
namespace common {

struct shmComm : public Comm {
  shmComm(int rank);
  ~shmComm();

  void Init(MPI_Comm mpiComm, const std::vector<int>& ranks, size_t buf_size);

  virtual void Send(void* buf, size_t buf_size, int peer_rank, cudaStream_t stream,
              size_t offset = 0);

  virtual int RecvBufAsync(void** buf, int peer_rank, cudaStream_t stream,
                           size_t offset = 0);
  // blocking Recv
  virtual void RecvBuf(void** buf, int peer_rank,
               cudaStream_t stream, size_t offset);

  // Manual communication with the remote buffer.
  // The pattern of usage is following
  // Get the pointer with get_sendBuf.
  // Launch the kernel using the remote buffer on stream.
  // Call post_sendBuf on the same stream.
  void* get_sendBuf(int peer_rank);
  // Post data transfer.
  void post_sendBuf(int peer_rank, cudaStream_t stream);

  virtual void WaitSendAll();
private:
  struct cudaEventSync {
    cudaEvent_t event;
    cudaIpcEventHandle_t eventHandle;
    MPI_Request request;
    unsigned char dummy;
  };

  struct shmBuffer {
    int shmSize;
    void* hostMem;
    void* devHostMem;
  };

  // Initialize send and receive resources.
  // Calls to sendInit and recvInit
  // must be separated with MPI_Barrier.
  void sendInit(shmBuffer* resource, int peer_rank, size_t shm_size);
  void recvInit(shmBuffer* resource, int peer_rank, size_t shm_size);
  // Initialize cudaIPC primitives.
  void initEventSend(cudaEventSync* eventSync, int recv_rank,
                     MPI_Request* request);
  void initEventRecv(cudaEventSync* eventSync, int send_rank);

  static void freeBuffer(shmBuffer* buffer);
  static void freeEventSync(cudaEventSync* eventSend);

  std::unordered_map<int, std::pair<shmBuffer, cudaEventSync>> send_resources;
  std::unordered_map<int, std::pair<shmBuffer, cudaEventSync>> recv_resources;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_SHM_UTILS_H
