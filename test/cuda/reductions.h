#ifndef HOROVOD_TEST_REDUCTIONS_H
#define HOROVOD_TEST_REDUCTIONS_H
#include "compression_utils.h"
#include "mpi.h"
#include "nccl.h"
#include "shared_memory.h"
#include <vector>

struct CommData {
  CommData(void* buf_, int rank)
      : buf(buf_), remote_rank(rank), request(MPI_REQUEST_NULL) {}

  cudaEvent_t event;
  cudaIpcMemHandle_t memHandle;
  cudaIpcEventHandle_t eventHandle;
  MPI_Request request;
  void* buf;
  int remote_rank;
  unsigned char dummy;
};

struct p2pComm {
  std::vector<CommData> send_comms;
  std::vector<CommData> recv_comms;
  int rank;
  int world_size;
};

void mpi_reduction_sra(float* buf, unsigned char* comm_buf, int num_elements,
                       int world_size, int rank, CompressionSetup* compSetup,
                       cudaStream_t* streams);

void nccl_reduction_ring(float* buf, unsigned char* comm_buf, int num_elements,
                         int world_size, int rank, CompressionSetup* compSetup,
                         cudaStream_t* streams, ncclComm_t* comm);

void nccl_reduction_sra(float* buf, unsigned char* comm_buf, int num_elements,
                        int world_size, int rank, CompressionSetup* compSetup,
                        cudaStream_t* streams, ncclComm_t* comm);

void shm_SRASetup(shmComm* comm, int rank, int world_size, int num_elems);
void shm_SRA(void* device_buf, shmComm* comm, void* comm_buf, int num_elems,
             CompressionSetup* compSetup, cudaStream_t* streams);
void shm_SRAFree(shmComm* comm);

void p2p_SRASetup(p2pComm* comm, int rank, int world_size, int buf_size,
                  void** recv_comm_buf);

void p2p_allgather(void* device_buf, p2pComm* comm, int num_elems,
                   cudaStream_t stream);

#endif // HOROVOD_TEST_REDUCTIONS_H
