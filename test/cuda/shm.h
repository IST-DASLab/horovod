#ifndef HOROVOD_TEST_CUDA_SHM_H
#define HOROVOD_TEST_CUDA_SHM_H

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include "common.h"

#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    printf("Call to " name " failed : %s", strerror(errno)); \
    return 1; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    printf("Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)


static int shm_allocate(int fd, const int shmsize) {
  int err = posix_fallocate(fd, 0, shmsize);
  if (err) { errno = err; return -1; }
  return 0;
}
static int shm_map(int fd, const int shmsize, void** ptr) {
  *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

static int shmSetup(const char* shmname, const int shmsize, int* fd, void** ptr, int create) {
  SYSCHECKVAL(shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "shm_open", *fd);
  if (create) SYSCHECK(shm_allocate(*fd, shmsize), "posix_fallocate");
  SYSCHECK(shm_map(*fd, shmsize, ptr), "mmap");
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
  if (shmname != NULL) SYSCHECK(shm_unlink(shmname), "shm_unlink");
  return 0;
}

static int shmClose(void* shmPtr, void* devShmPtr, const int shmsize) {
  CUDACHECK(cudaHostUnregister(shmPtr));
  if (munmap(shmPtr, shmsize) != 0) {
    printf("munmap of shared memory failed\n");
    return 1;
  }
  return 0;
}


#endif // HOROVOD_TEST_CUDA_SHM_H
