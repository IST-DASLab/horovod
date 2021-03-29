#ifndef HOROVOD_TEST_CUDA_SHARED_MEMORY_H
#define HOROVOD_TEST_CUDA_SHARED_MEMORY_H
#include "shm.h"

struct shmResource {
  int peerRank;
  int shmSize;
  void* hostMem;
  void* devHostMem;
};

struct cudaEventSync {
  cudaEvent_t event;
  cudaIpcEventHandle_t eventHandle;
  MPI_Request request;
};

struct shmComm {
  shmResource* sendResources;
  shmResource* recvResources;
  cudaEventSync* sendEvents;
  cudaEventSync* recvEvents;
  int world_size;
  int rank;
};

void sendInit(shmResource* resource, int sendRank, int recvRank,
    size_t shmSize);
void recvInit(shmResource* resource, int sendRank, int recvRank,
    size_t shmSize);

void initEventSend(cudaEventSync* eventSync, int recv_rank,
                   MPI_Request* request);
void initEventRecv(cudaEventSync* eventSync, int send_rank);

// Copy to remote buffer.
void memcpy2sendBuf(void* buf, size_t buf_size, size_t shm_offset,
                    shmResource* resource, cudaEventSync* eventSync,
                    cudaStream_t stream);

// Get pointer to remote buffer.
void* get_sendBuf(shmResource* resource);
// Post data transfer.
void post_sendBuf(shmResource* resource, cudaEventSync* eventSync,
                  cudaStream_t stream);

// non blocking Recv.
int recvBufAsync(void** buf, size_t shm_offset, shmResource* resource,
                 cudaEventSync* eventSync, cudaStream_t stream);
// blocking Recv
void recvBuf(void** buf, shmResource* resource, cudaEventSync* eventSync,
             cudaStream_t stream);

void freeSendResource(shmResource* resource);
void freeEventSync(cudaEventSync* eventSend);

void initSendConnections(shmResource* resources, int world_size, int rank,
                         int shmSize);
void initRecvConnections(shmResource* resources, int world_size, int rank,
                         int shmSize);
void freeConnections(shmResource* resources, int size);

void initEvents(cudaEventSync* eventsSend, cudaEventSync* eventsRecv,
                int world_size, int rank);
void freeEvents(cudaEventSync* eventsSend, int size);


#endif // HOROVOD_TEST_CUDA_SHARED_MEMORY_H
