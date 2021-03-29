#include <assert.h>
#include <vector>

#include "shared_memory.h"

#define NUM_ELEMS 6
#define MAX_SHM_NAME_LEN 1024

void sendInit(shmResource* resource, int sendRank, int recvRank,
              size_t shmSize) {
  char shmName[MAX_SHM_NAME_LEN];
  resource->peerRank = recvRank;
  resource->shmSize = shmSize;
  sprintf(shmName, "nccl-shm-send-%d-%d", sendRank, recvRank);
  TRIVCHECK(shmOpen(shmName, resource->shmSize, (void**)&resource->hostMem,
                     (void**)&resource->devHostMem, 1));
}

void recvInit(shmResource* resource, int sendRank, int recvRank,
              size_t shmSize) {
  char shmName[MAX_SHM_NAME_LEN];
  resource->peerRank = sendRank;
  resource->shmSize = shmSize;
  sprintf(shmName, "nccl-shm-send-%d-%d", sendRank, recvRank);
  TRIVCHECK(shmOpen(shmName, resource->shmSize, (void**)&resource->hostMem,
                     (void**)&resource->devHostMem, 0));
  TRIVCHECK(shmUnlink(shmName));
}

void memcpy2sendBuf(void* buf, size_t buf_size, size_t shm_offset,
                    shmResource* resource, cudaEventSync* eventSync,
                    cudaStream_t stream) {
  assert(buf_size <= resource->shmSize);
  CUDACHECK(cudaMemcpyAsync((void*)(((char*)resource->devHostMem) + shm_offset),
                            buf, buf_size, cudaMemcpyDeviceToDevice, stream));
  CUDACHECK(cudaEventRecord(eventSync->event, stream));
  char a;
  MPICHECK(MPI_Isend((void*)&a, 1, MPI_UNSIGNED_CHAR, resource->peerRank, 0,
                     MPI_COMM_WORLD, &eventSync->request));
}

void* get_sendBuf(shmResource* resource) { return resource->devHostMem; }

void post_sendBuf(shmResource* resource, cudaEventSync* eventSync,
                  cudaStream_t stream) {
  CUDACHECK(cudaEventRecord(eventSync->event, stream));
  char a;
  MPICHECK(MPI_Isend((void*)&a, 1, MPI_UNSIGNED_CHAR, resource->peerRank, 0,
                     MPI_COMM_WORLD, &eventSync->request));
}

int recvBufAsync(void** buf, size_t shm_offset, shmResource* resource,
                 cudaEventSync* eventSync, cudaStream_t stream) {
  *buf = nullptr;
  if (eventSync->request == MPI_REQUEST_NULL) {
    char a;
    MPICHECK(MPI_Irecv(&a, 1, MPI_UNSIGNED_CHAR, resource->peerRank, 0,
                       MPI_COMM_WORLD, &eventSync->request));
  }
  int flag;
  MPICHECK(MPI_Test(&eventSync->request, &flag, MPI_STATUSES_IGNORE));
  if (!flag)
    return 1;
  CUDACHECK(cudaStreamWaitEvent(stream, eventSync->event, 0));
  *buf = (void*)(((char*)resource->devHostMem) + shm_offset);
  eventSync->request = MPI_REQUEST_NULL;
  return 0;
}

void recvBuf(void** buf, shmResource* resource, cudaEventSync* eventSync,
             cudaStream_t stream) {
  *buf = nullptr;
  char a;
  MPICHECK(MPI_Recv(&a, 1, MPI_UNSIGNED_CHAR, resource->peerRank, 0,
                    MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
  CUDACHECK(cudaStreamWaitEvent(stream, eventSync->event, 0));
  *buf = resource->devHostMem;
}

void freeSendResource(shmResource* resource) {
  TRIVCHECK(
      shmClose(resource->hostMem, resource->devHostMem, resource->shmSize));
}

void initEventSend(cudaEventSync* eventSync, int recv_rank,
                   MPI_Request* request) {
  CUDACHECK(cudaEventCreate(&eventSync->event,
                            cudaEventDisableTiming | cudaEventInterprocess));
  CUDACHECK(cudaIpcGetEventHandle(
      (cudaIpcEventHandle_t*)&eventSync->eventHandle, eventSync->event));
  MPICHECK(MPI_Isend((void*)(&eventSync->eventHandle),
                     sizeof(eventSync->eventHandle), MPI_UNSIGNED_CHAR,
                     recv_rank, 0, MPI_COMM_WORLD, request));
}

void initEventRecv(cudaEventSync* eventSync, int send_rank) {
  MPICHECK(MPI_Recv((void*)(&eventSync->eventHandle),
                    sizeof(eventSync->eventHandle), MPI_UNSIGNED_CHAR,
                    send_rank, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
  CUDACHECK(cudaIpcOpenEventHandle(&eventSync->event, eventSync->eventHandle));
  eventSync->request = MPI_REQUEST_NULL;
}

void initSendConnections(shmResource* resources, int world_size, int rank,
                         int shmSize) {
  int count = 0;
  shmResource* cur_resource;
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (rank == peer_rank)
      continue;
    cur_resource = resources + count;
    sendInit(cur_resource, rank, peer_rank, shmSize);
    count++;
  }
}

void initRecvConnections(shmResource* resources, int world_size, int rank,
                         int shmSize) {
  int count = 0;
  shmResource* cur_resource;
  count = 0;
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (rank == peer_rank)
      continue;
    cur_resource = resources + count;
    recvInit(cur_resource, peer_rank, rank, shmSize);
    count++;
  }
}

void freeConnections(shmResource* resources, int size) {
  for (int i = 0; i < size; i++) {
    freeSendResource(resources + i);
  }
}

void initEvents(cudaEventSync* eventsSend, cudaEventSync* eventsRecv,
                int world_size, int rank) {
  MPI_Request* send_requests = new MPI_Request[world_size - 1];
  int count = 0;
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank)
      continue;
    initEventSend(&eventsSend[count], peer_rank, &send_requests[count]);
    count++;
  }

  count = 0;
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank)
      continue;
    initEventRecv(&eventsRecv[count], peer_rank);
    count++;
  }

  MPI_Waitall(world_size - 1, send_requests, MPI_STATUSES_IGNORE);
  delete[] send_requests;
}

void freeEventSync(cudaEventSync* eventSend) {
  cudaEventDestroy(eventSend->event);
}

void freeEvents(cudaEventSync* eventsSend, int size) {
  for (int i = 0; i < size; i++) {
    freeEventSync(&eventsSend[i]);
  }
}
