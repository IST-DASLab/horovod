#ifndef HOROVOD_COMPRESSED_P2P_COMM_H
#define HOROVOD_COMPRESSED_P2P_COMM_H
#include <unordered_map>
#include <vector>
#include "comm.h"

namespace horovod {
namespace common {

struct p2pComm : public Comm {
  p2pComm(int rank) : Comm(rank) {}
  // mpiComm: communicator within which we perform reduction.
  // send_ranks: ranks within the communicator used for sending. All ranks must map to GPU devices.
  // recv_ranks: ranks within the communicator used for receiving. All ranks must map to GPU devices.
  // recv_buf: buffer used for receiving in communication.
  // buf_size: buffer size taken from recv_buf for each rank.
  // Assumed that recv_bufs points to buffer with size
  // allocated buf_size * ranks.size() bytes.
  void Init(MPI_Comm mpiComm, const std::vector<int>& send_ranks,
            const std::vector<int>& recv_ranks,
            unsigned char* recv_bufs, size_t buf_size);

  virtual void Send(void* buf, size_t buf_size, int peer_rank,
                    cudaStream_t stream, size_t offset = 0);

  virtual int RecvBufAsync(void** buf, int peer_rank, cudaStream_t stream, size_t offset);
  // blocking Recv
  virtual void RecvBuf(void** buf, int peer_rank, cudaStream_t stream, size_t offset);

  virtual void WaitSendAll();

private:
  struct CommData {
    CommData()
        : buf(nullptr), request(MPI_REQUEST_NULL) {}
    CommData(void* buf_)
        : buf(buf_), request(MPI_REQUEST_NULL) {}

    cudaEvent_t event;
    cudaIpcMemHandle_t memHandle;
    cudaIpcEventHandle_t eventHandle;
    MPI_Request request;
    void* buf;
    unsigned char dummy;
  };

  std::unordered_map<int, CommData> send_comms;
  std::unordered_map<int, CommData> recv_comms;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_COMPRESSED_P2P_COMM_H
