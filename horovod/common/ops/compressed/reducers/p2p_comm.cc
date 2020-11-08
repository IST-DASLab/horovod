#include "p2p_comm.h"
#include "../../../utils/env_parser.h"
#include "common.h"

namespace horovod {
namespace common {

void p2pComm::Init(MPI_Comm mpiComm, const std::vector<int>& ranks,
                   unsigned char* recv_bufs, size_t buf_size) {
  comm_ = mpiComm;
  int rank;
  cudaGetDevice(&rank);
  for (auto peer_rank : ranks) {
    recv_comms.emplace(peer_rank, CommData(recv_bufs));
    send_comms.emplace(peer_rank, CommData(nullptr));
    recv_bufs += buf_size;
  }

  std::vector<MPI_Request> send_requests;
  // Create handles on receiver sides.
  for (auto peer_rank : ranks) {
    CommData& commData = recv_comms[peer_rank];
    CUDA_CHECK(cudaIpcGetMemHandle(&commData.memHandle, (void*)commData.buf));
    send_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend((void*)&commData.memHandle, sizeof(commData.memHandle),
                        MPI_UNSIGNED_CHAR, peer_rank, 0, comm_,
                        &send_requests.back()));
  }

  for (auto peer_rank : ranks) {
    CommData& commData = send_comms[peer_rank];
    MPI_CHECK(MPI_Recv((void*)&commData.memHandle, sizeof(commData.memHandle),
                       MPI_UNSIGNED_CHAR, peer_rank, 0, comm_,
                       MPI_STATUS_IGNORE));
    int canAccessPeer;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, rank, peer_rank));
    if (!canAccessPeer) {
      CUDA_CHECK(cudaSetDevice(peer_rank));
    } else {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer_rank, 0));
    }
    CUDA_CHECK(cudaIpcOpenMemHandle((void**)&commData.buf, commData.memHandle,
                                    cudaIpcMemLazyEnablePeerAccess));
    if (!canAccessPeer) {
      CUDA_CHECK(cudaSetDevice(rank));
    }
    CUDA_CHECK(cudaEventCreate(&commData.event,
                               cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_CHECK(cudaIpcGetEventHandle(&commData.eventHandle, commData.event));
    send_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend((void*)&commData.eventHandle, sizeof(commData.eventHandle),
                        MPI_UNSIGNED_CHAR, peer_rank, 0, comm_,
                        &send_requests.back()));
  }

  for (auto peer_rank : ranks) {
    CommData& commData = recv_comms[peer_rank];
    MPI_CHECK(MPI_Recv((void*)(&commData.eventHandle), sizeof(commData.eventHandle),
                       MPI_UNSIGNED_CHAR, peer_rank, 0, comm_,
                       MPI_STATUS_IGNORE));
    CUDA_CHECK(cudaIpcOpenEventHandle(&commData.event, commData.eventHandle));
  }
  MPI_CHECK(MPI_Waitall(send_requests.size(), send_requests.data(),
                        MPI_STATUS_IGNORE));
}

void p2pComm::Send(void* buf, size_t buf_size, int peer_rank, cudaStream_t stream,
                   size_t offset) {
  CommData& commData = send_comms[peer_rank];
  cudaMemcpyAsync(commData.buf, buf, buf_size, cudaMemcpyDefault, stream);
  cudaEventRecord(commData.event);
  MPI_Isend(&commData.dummy, sizeof(commData.dummy), MPI_UNSIGNED_CHAR,
            peer_rank, 0, comm_, &commData.request);
}

int p2pComm::RecvBufAsync(void** buf, int peer_rank, cudaStream_t stream,
                          size_t offset) {
  CommData& commData = recv_comms[peer_rank];
  *buf = nullptr;
  if (commData.request == MPI_REQUEST_NULL) {
    MPI_CHECK(MPI_Irecv(&commData.dummy, 1, MPI_UNSIGNED_CHAR, peer_rank, 0,
                        MPI_COMM_WORLD, &commData.request));
  }
  int flag;
  MPI_CHECK(MPI_Test(&commData.request, &flag, MPI_STATUSES_IGNORE));
  if (!flag)
    return 0;
  CUDA_CHECK(cudaStreamWaitEvent(stream, commData.event, 0));
  *buf = commData.buf;
  commData.request = MPI_REQUEST_NULL;
  return 1;
}

void p2pComm::RecvBuf(void** buf, int peer_rank, cudaStream_t stream, size_t offset) {
  CommData& commData = recv_comms[peer_rank];
  MPI_CHECK(MPI_Recv(&commData.dummy, 1, MPI_UNSIGNED_CHAR, peer_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
  CUDA_CHECK(cudaStreamWaitEvent(stream, commData.event, 0));
  *buf = commData.buf;
}

void p2pComm::WaitSendAll() {
  for (auto& comm : send_comms) {
    auto& commData = comm.second;
    MPI_Wait(&commData.request, MPI_STATUSES_IGNORE);
  }
}

} // namespace common
} // namespace horovod
