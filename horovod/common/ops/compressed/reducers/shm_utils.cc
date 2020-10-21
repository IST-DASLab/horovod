#include "shm_utils.h"
#include <assert.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include "common.h"

/*
 * This code is taken from NCCL library.
 */


#define MAX_SHM_NAME_LEN 1024

namespace horovod {
namespace common {

/*
 * Utility functions.
 */

int shm_allocate(int fd, const int shmsize) {
  int err = posix_fallocate(fd, 0, shmsize);
  if (err) { errno = err; return -1; }
  return 0;
}

int shm_map(int fd, const int shmsize, void** ptr) {
  *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

static int shmSetup(const char* shmname, const int shmsize, int* fd, void** ptr, int create) {
  SYS_CHECKVAL(shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "shm_open", *fd);
  if (create) SYS_CHECK(shm_allocate(*fd, shmsize), "posix_fallocate");
  SYS_CHECK(shm_map(*fd, shmsize, ptr), "mmap");
  close(*fd);
  *fd = -1;
  if (create) memset(*ptr, 0, shmsize);
  return 0;
}

static int shmOpen(const char* shmname, const int shmsize, void** shmPtr, void** devShmPtr, int create) {
  int fd = -1;
  void* ptr = MAP_FAILED;
  int res = 0;

  res = shmSetup(shmname, shmsize, &fd, &ptr, create);
  if (res > 0)
    goto sysError;
  if ((res = cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped)) != cudaSuccess ||
      (res = cudaHostGetDevicePointer(devShmPtr, ptr, 0)) != cudaSuccess)
    goto cudaError;

  *shmPtr = ptr;
  return 0;
  sysError:
  printf("Error while %s shared memory segment %s (size %d)\n", create ? "creating" : "attaching to", shmname, shmsize);
  cudaError:
  if (fd != -1) close(fd);
  if (create) shm_unlink(shmname);
  if (ptr != MAP_FAILED) munmap(ptr, shmsize);
  *shmPtr = NULL;
  return res;
}

static int shmUnlink(const char* shmname) {
  if (shmname != NULL) SYS_CHECK(shm_unlink(shmname), "shm_unlink");
  return 0;
}

static int shmClose(void* shmPtr, void* devShmPtr, const int shmsize) {
  CUDA_CHECK(cudaHostUnregister(shmPtr));
  if (munmap(shmPtr, shmsize) != 0) {
    printf("munmap of shared memory failed\n");
    return 1;
  }
  return 0;
}


/*
 * Shared memory communication functions.
 */
shmComm::shmComm(int rank): rank_(rank) {}

void shmComm::Init(MPI_Comm mpiComm, const std::vector<int>& ranks, size_t buf_size) {
  comm_ = mpiComm;
  // Initialize shared memory buffers.
  for (auto peer_rank: ranks) {
    auto& send_resource = send_resources[peer_rank];
    sendInit(&send_resource.first, peer_rank, buf_size);
  }
  MPI_Barrier(comm_);
  for (auto peer_rank: ranks) {
    auto& recv_resource = recv_resources[peer_rank];
    recvInit(&recv_resource.first, peer_rank, buf_size);
  }

  // Initialize IPC primitives.
  MPI_Request* send_requests = new MPI_Request[ranks.size()];
  int count = 0;
  for (auto peer_rank: ranks) {
    auto& send_resource = send_resources[peer_rank];
    initEventSend(&send_resource.second, peer_rank, &send_requests[count++]);
  }
  for (auto peer_rank: ranks) {
    auto& recv_resource = recv_resources[peer_rank];
    initEventRecv(&recv_resource.second,  peer_rank);
  }
  MPI_Waitall(ranks.size(), send_requests, MPI_STATUSES_IGNORE);
  delete [] send_requests;
}

void shmComm::sendInit(shmBuffer* buffer, int peer_rank,
              size_t shm_size) {
  char shmName[MAX_SHM_NAME_LEN];
  buffer->shmSize = shm_size;
  sprintf(shmName, "nccl-shm-send-%d-%d", rank_, peer_rank);
  TRIV_CHECK(shmOpen(shmName, buffer->shmSize, (void**)&buffer->hostMem,
                    (void**)&buffer->devHostMem, 1));
}

void shmComm::recvInit(shmBuffer* buffer, int peer_rank,
              size_t shm_size) {
  char shmName[MAX_SHM_NAME_LEN];
  buffer->shmSize = shm_size;
  sprintf(shmName, "nccl-shm-send-%d-%d", peer_rank, rank_);
  TRIV_CHECK(shmOpen(shmName, buffer->shmSize, (void**)&buffer->hostMem,
                    (void**)&buffer->devHostMem, 0));
  TRIV_CHECK(shmUnlink(shmName));
}

void shmComm::initEventSend(horovod::common::shmComm::cudaEventSync* eventSync,
                            int recv_rank, MPI_Request* request) {
  CUDA_CHECK(cudaEventCreateWithFlags(&eventSync->event,
                            cudaEventDisableTiming | cudaEventInterprocess));
  CUDA_CHECK(cudaIpcGetEventHandle(
      (cudaIpcEventHandle_t*)&eventSync->eventHandle, eventSync->event));
  MPI_CHECK(MPI_Isend((void*)(&eventSync->eventHandle),
                     sizeof(eventSync->eventHandle), MPI_UNSIGNED_CHAR,
                     recv_rank, 0, comm_, request));
}

void shmComm::initEventRecv(horovod::common::shmComm::cudaEventSync* eventSync,
                            int send_rank)  {
  MPI_CHECK(MPI_Recv((void*)(&eventSync->eventHandle),
                    sizeof(eventSync->eventHandle), MPI_UNSIGNED_CHAR,
                    send_rank, 0, comm_, MPI_STATUSES_IGNORE));
  CUDA_CHECK(cudaIpcOpenEventHandle(&eventSync->event, eventSync->eventHandle));
  eventSync->request = MPI_REQUEST_NULL;
}

void shmComm::memcpy(void* buf, size_t buf_size, size_t shm_offset,
                     int peer_rank, cudaStream_t stream) {
  auto& send_resource = send_resources[peer_rank];
  auto& shm_buf = send_resource.first;
  auto& eventSync = send_resource.second;
  assert(buf_size + shm_offset < shm_buf.shmSize);
  CUDA_CHECK(cudaMemcpyAsync((void*)(((char*)shm_buf.devHostMem) + shm_offset),
                            buf, buf_size, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaEventRecord(eventSync.event, stream));
  char a;
  MPI_CHECK(MPI_Isend((void*)&a, 1, MPI_UNSIGNED_CHAR, peer_rank, 0,
                     comm_, &eventSync.request));
}

void* shmComm::get_sendBuf(int peer_rank) {
  auto &send_resource = send_resources[peer_rank];
  return send_resource.first.devHostMem;
}

void shmComm::post_sendBuf(int peer_rank, cudaStream_t stream) {
  auto& eventSync = send_resources[peer_rank].second;
  CUDA_CHECK(cudaEventRecord(eventSync.event, stream));
  char a;
  MPI_CHECK(MPI_Isend((void*)&a, 1, MPI_UNSIGNED_CHAR, peer_rank, 0,
                     comm_, &eventSync.request));
}

int shmComm::recvBufAsync(void** buf, size_t shm_offset, int peer_rank,
                 cudaStream_t stream) {
  auto& recv_resource = recv_resources[peer_rank];
  auto &shm_buf = recv_resource.first;
  auto &eventSync = recv_resource.second;
  *buf = nullptr;
  if (eventSync.request == MPI_REQUEST_NULL) {
    char a;
    MPI_CHECK(MPI_Irecv(&a, 1, MPI_UNSIGNED_CHAR, peer_rank, 0,
                       comm_, &eventSync.request));
  }
  int flag;
  MPI_CHECK(MPI_Test(&eventSync.request, &flag, MPI_STATUSES_IGNORE));
  if (!flag)
    return 1;
  CUDA_CHECK(cudaStreamWaitEvent(stream, eventSync.event, 0));
  *buf = (void*)(((char*)shm_buf.devHostMem) + shm_offset);
  eventSync.request = MPI_REQUEST_NULL;
  return 0;
}

void shmComm::recvBuf(void** buf, size_t shm_offset, int peer_rank,
             cudaStream_t stream) {
  auto& recv_resource = recv_resources[peer_rank];
  auto& shm_buf = recv_resource.first;
  auto& eventSync = recv_resource.second;
  *buf = nullptr;
  char a;
  MPI_CHECK(MPI_Recv(&a, 1, MPI_UNSIGNED_CHAR, peer_rank, 0,
                    comm_, MPI_STATUSES_IGNORE));
  CUDA_CHECK(cudaStreamWaitEvent(stream, eventSync.event, 0));
  *buf = ((char*)shm_buf.devHostMem) + shm_offset;
}

void shmComm::waitSendAll() {
  for (auto& resource: send_resources) {
    auto& eventSync = resource.second.second;
    MPI_Wait(&eventSync.request, MPI_STATUSES_IGNORE);
  }
}

shmComm::~shmComm() {
  // Can't deallocate buffers here, because at this point destuctor is called
  // CUDA driver is in the middle of deinitialization.
  for (auto& resource: send_resources) {
    freeEventSync(&resource.second.second);
//    freeBuffer(&resource.second.first);
  }
//  for (auto& resource: recv_resources) {
//    freeBuffer(&resource.second.first);
//  }
}

void shmComm::freeBuffer(shmBuffer* buffer) {
  TRIV_CHECK(
      shmClose(buffer->hostMem, buffer->devHostMem, buffer->shmSize));
}


void shmComm::freeEventSync(cudaEventSync* eventSend) {
  cudaEventDestroy(eventSend->event);
}

} // namespace common
} // namespace horovod
