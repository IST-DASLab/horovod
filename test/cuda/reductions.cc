#include "reductions.h"
#include "common.h"

#include <algorithm>
#include <queue>

void mpi_reduction_sra(float* buf, unsigned char* comm_buf, int num_elements,
                       int world_size, int rank, CompressionSetup* compSetup,
                       cudaStream_t* streams) {
  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
  int start_elem = num_elems_per_node * rank + std::min(residue, rank);
  int recv_num_elems = num_elems_per_node + (rank < residue ? 1 : 0);
  int recv_compressed_size =
      round_to(get_compressed_size(recv_num_elems, compSetup), ALIGNMENT_UNIT);
  int send_num_elems = 0;
  int send_compressed_size = 0;
  unsigned char* gradients_send_ = comm_buf;
  unsigned char* gradients_recv_ =
      comm_buf +
      world_size * round_to(get_compressed_size(num_elements, compSetup),
                            ALIGNMENT_UNIT);

  unsigned char* send_buf = gradients_send_;
  unsigned char* recv_buf = gradients_recv_;
  std::vector<MPI_Request> send_requests;
  std::vector<MPI_Request> recv_requests;
  std::queue<int> send_sizes;
  int count = 0;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    send_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    send_compressed_size = round_to(
        get_compressed_size(send_num_elems, compSetup), ALIGNMENT_UNIT);
    compress((unsigned char*)(buf + start_offset), send_buf, send_num_elems,
             compSetup, streams[count]);
    send_buf += send_compressed_size;
    send_sizes.push(send_compressed_size);
    count++;
  }

  count = 0;
  send_buf = gradients_send_;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    recv_requests.push_back(MPI_Request());
    MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              MPI_COMM_WORLD, &recv_requests.back());
    send_compressed_size = send_sizes.front();
    send_requests.push_back(MPI_Request());

    cudaStreamSynchronize(streams[count]);

    MPI_Isend(send_buf, send_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              MPI_COMM_WORLD, &send_requests.back());

    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
    send_sizes.pop();
    count++;
  }
  // TODO: handling errors!!!
  std::vector<int> idx_map;
  for (int i = 0; i < world_size - 1; i++) {
    idx_map.push_back(i);
  }

  while (recv_requests.size() > 0) {
    int req_idx;
    MPI_Waitany((int)recv_requests.size(), recv_requests.data(), &req_idx,
                MPI_STATUSES_IGNORE);
    int idx = idx_map[req_idx];
    recv_requests.erase(recv_requests.begin() + req_idx);
    idx_map.erase(idx_map.begin() + req_idx);
    decompress(gradients_recv_ + idx * recv_compressed_size,
               (unsigned char*)(buf + start_elem), recv_num_elems, compSetup,
               *streams, true);
  }
  MPI_Waitall((int)send_requests.size(), send_requests.data(),
              MPI_STATUSES_IGNORE);
  send_requests.clear();

  compress((unsigned char*)(buf + start_elem), gradients_send_, recv_num_elems,
           compSetup, *streams);
  decompress(gradients_send_, (unsigned char*)(buf + start_elem),
             recv_num_elems, compSetup, *streams, false);

  cudaStreamSynchronize(*streams);
  recv_buf = gradients_recv_;

  // second round of MPI communication. receive the sums from other nodes
  send_compressed_size = recv_compressed_size;
  std::vector<std::pair<int64_t, int>> recv_offsets;
  int64_t recv_acc_size = 0;
  recv_requests.clear();
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int their_start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    recv_compressed_size = round_to(
        get_compressed_size(recv_num_elems, compSetup), ALIGNMENT_UNIT);

    recv_requests.push_back(MPI_Request());
    MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR, node_rank, 0,
              MPI_COMM_WORLD, &recv_requests.back());
    send_requests.push_back(MPI_Request());
    MPI_Isend(gradients_send_, send_compressed_size, MPI_UNSIGNED_CHAR,
              node_rank, 0, MPI_COMM_WORLD, &send_requests.back());
    recv_buf += recv_compressed_size;
    recv_offsets.emplace_back(recv_acc_size, their_start_offset);
    recv_acc_size += recv_compressed_size;
  }
  count = 0;
  while (recv_requests.size() > 0) {
    int req_idx;
    int their_start_offset;
    MPI_Waitany((int)recv_requests.size(), recv_requests.data(), &req_idx,
                MPI_STATUSES_IGNORE);

    std::tie(recv_acc_size, their_start_offset) = recv_offsets[req_idx];
    recv_requests.erase(recv_requests.begin() + req_idx);
    recv_offsets.erase(recv_offsets.begin() + req_idx);
    decompress(gradients_recv_ + recv_acc_size,
               (unsigned char*)(buf + their_start_offset), recv_num_elems,
               compSetup, streams[count++], false);
  }
  for (int i = 0; i < world_size - 1; i++) {
    cudaStreamSynchronize(streams[i]);
  }
  MPI_Waitall((int)send_requests.size(), send_requests.data(),
              MPI_STATUSES_IGNORE);
}

void nccl_reduction_ring(float* buf, unsigned char* comm_buf, int num_elements,
                         int world_size, int rank, CompressionSetup* compSetup,
                         cudaStream_t* streams, ncclComm_t* comm) {
  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;

  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;

  auto segment_size = [num_elems_per_node, residue](int segment) {
    return num_elems_per_node + ((segment < residue) ? 1 : 0);
  };
  unsigned char* gradients_send_ = comm_buf;
  unsigned char* gradients_recv_ =
      comm_buf +
      world_size * round_to(get_compressed_size(num_elements, compSetup),
                            ALIGNMENT_UNIT);

  std::vector<size_t> segment_ends(world_size);
  segment_ends[0] = segment_size(0);
  for (size_t i = 1; i < segment_ends.size(); ++i) {
    segment_ends[i] = segment_size(i) + segment_ends[i - 1];
  }

  int recv_segment_idx, send_segment_idx;
  int64_t buf_send_idx, buf_recv_idx;
  int64_t send_size, recv_size;

  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    buf_send_idx =
        (segment_ends[send_segment_idx] - segment_size(send_segment_idx));
    buf_recv_idx =
        (segment_ends[recv_segment_idx] - segment_size(recv_segment_idx));

    recv_size =
        round_to(get_compressed_size(segment_size(recv_segment_idx), compSetup),
                 ALIGNMENT_UNIT);
    send_size =
        round_to(get_compressed_size(segment_size(send_segment_idx), compSetup),
                 ALIGNMENT_UNIT);
    compress((unsigned char*)(buf + buf_send_idx), gradients_send_,
             segment_size(send_segment_idx), compSetup, streams[0]);
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(gradients_send_, send_size, ncclChar, send_to, *comm,
                       *streams));
    NCCLCHECK(ncclRecv(gradients_recv_, recv_size, ncclChar, recv_from, *comm,
                       *streams));
    NCCLCHECK(ncclGroupEnd());
    decompress(gradients_recv_, (unsigned char*)(buf + buf_recv_idx),
               segment_size(recv_segment_idx), compSetup, streams[0], true);
  }

  send_segment_idx = (rank + world_size + 1) % world_size;
  buf_send_idx =
      (segment_ends[send_segment_idx] - segment_size(send_segment_idx));
  unsigned char* send_buf = gradients_send_;
  send_size =
      round_to(get_compressed_size(segment_size(send_segment_idx), compSetup),
               ALIGNMENT_UNIT);
  compress((unsigned char*)(buf + buf_send_idx), send_buf,
           segment_size(send_segment_idx), compSetup, streams[0]);
  decompress(send_buf, (unsigned char*)(buf + buf_send_idx),
             segment_size(send_segment_idx), compSetup, streams[0], false);
  unsigned char* recv_buf = send_buf + send_size;
  unsigned char* compressed_buf = recv_buf;

  for (int i = 0; i < world_size - 1; i++) {
    NCCLCHECK(ncclGroupStart());
    recv_segment_idx = (rank - i + world_size) % world_size;
    // Segment to send - at every iteration we send segment (r+1-i)
    buf_recv_idx =
        (segment_ends[recv_segment_idx] - segment_size(recv_segment_idx));
    recv_size =
        round_to(get_compressed_size(segment_size(recv_segment_idx), compSetup),
                 ALIGNMENT_UNIT);

    // Segment to recv - at every iteration we receive segment (r-i)
    NCCLCHECK(
        ncclSend(send_buf, send_size, ncclChar, send_to, *comm, *streams));
    NCCLCHECK(
        ncclRecv(recv_buf, recv_size, ncclChar, recv_from, *comm, *streams));

    send_buf += send_size;
    recv_buf += recv_size;
    send_size = recv_size;
    NCCLCHECK(ncclGroupEnd());
  }

  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    buf_recv_idx =
        (segment_ends[recv_segment_idx] - segment_size(recv_segment_idx));

    decompress(compressed_buf, (unsigned char*)(buf + buf_recv_idx),
               segment_size(recv_segment_idx), compSetup, streams[0], false);
    recv_size =
        round_to(get_compressed_size(segment_size(recv_segment_idx), compSetup),
                 ALIGNMENT_UNIT);

    compressed_buf += recv_size;
  }
}

void nccl_reduction_sra(float* buf, unsigned char* comm_buf, int num_elements,
                        int world_size, int rank, CompressionSetup* compSetup,
                        cudaStream_t* streams, ncclComm_t* comm) {

  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
  int start_elem = num_elems_per_node * rank + std::min(residue, rank);
  int recv_num_elems = num_elems_per_node + (rank < residue ? 1 : 0);
  int recv_compressed_size =
      round_to(get_compressed_size(recv_num_elems, compSetup), ALIGNMENT_UNIT);
  int send_num_elems = 0;
  int send_compressed_size = 0;
  unsigned char* gradients_send_ = comm_buf;
  unsigned char* gradients_recv_ =
      comm_buf +
      world_size * round_to(get_compressed_size(num_elements, compSetup),
                            ALIGNMENT_UNIT);

  cudaEvent_t* events = new cudaEvent_t[world_size - 1];

  unsigned char* send_buf = gradients_send_;
  unsigned char* recv_buf = gradients_recv_;
  std::queue<int> send_sizes;
  int count = 0;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    cudaEventCreateWithFlags(&events[count], cudaEventDisableTiming);
    int start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    send_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    send_compressed_size = round_to(
        get_compressed_size(send_num_elems, compSetup), ALIGNMENT_UNIT);
    compress((unsigned char*)(buf + start_offset), send_buf, send_num_elems,
             compSetup, streams[count]);
    cudaEventRecord(events[count], streams[count]);
    send_buf += send_compressed_size;
    send_sizes.push(send_compressed_size);
    count++;
  }
  //  for (int i = 0; i < count; i++) {
  //    cudaStreamSynchronize(streams[i]);
  //  }
  count = 0;
  NCCLCHECK(ncclGroupStart());
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    cudaStreamWaitEvent(streams[0], events[count], 0);
    send_compressed_size = send_sizes.front();
    NCCLCHECK(ncclRecv(recv_buf, recv_compressed_size, ncclChar, node_rank,
                       *comm, streams[0]));
    NCCLCHECK(ncclSend(send_buf, send_compressed_size, ncclChar, node_rank,
                       *comm, streams[0]));
    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
    send_sizes.pop();
    cudaEventRecord(events[count], streams[0]);
    count++;
  }
  NCCLCHECK(ncclGroupEnd());

  recv_buf = gradients_recv_;
  bool* ready_flags = new bool[world_size - 1];
  std::fill(ready_flags, ready_flags + (world_size - 1), false);
  int ready_hosts = 0;
  while (ready_hosts < world_size - 1) {
    for (int i = 0; i < world_size - 1; i++) {
      if (ready_flags[i])
        continue;
      if (cudaEventQuery(events[i]) == cudaSuccess) {
        recv_buf = gradients_recv_ + recv_compressed_size * i;
        decompress(recv_buf, (unsigned char*)(buf + start_elem), recv_num_elems,
                   compSetup, streams[0], true);
        ready_flags[i] = true;
        ready_hosts++;
      }
    }
  }
  compress((unsigned char*)(buf + start_elem), gradients_send_, recv_num_elems,
           compSetup, streams[0]);
  decompress(gradients_send_, (unsigned char*)(buf + start_elem),
             recv_num_elems, compSetup, streams[0], false);
  cudaEventRecord(events[0], streams[0]);

  recv_buf = gradients_recv_;
  // second round of communication. receive the sums from other nodes
  send_compressed_size = recv_compressed_size;

  count = 0;
  cudaStreamWaitEvent(streams[0], events[0], 0);
  NCCLCHECK(ncclGroupStart());
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int their_start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    recv_compressed_size = round_to(
        get_compressed_size(recv_num_elems, compSetup), ALIGNMENT_UNIT);
    NCCLCHECK(ncclRecv(recv_buf, recv_compressed_size, ncclChar, node_rank,
                       *comm, streams[0]));
    NCCLCHECK(ncclSend(gradients_send_, send_compressed_size, ncclChar,
                       node_rank, *comm, streams[0]));
    recv_buf += recv_compressed_size;
    cudaEventRecord(events[count], streams[0]);
    count++;
  }
  NCCLCHECK(ncclGroupEnd());
  std::fill(ready_flags, ready_flags + (world_size - 1), false);
  ready_hosts = 0;
  recv_buf = gradients_recv_;
  int their_start_offset = 0;
  while (ready_hosts < world_size - 1) {
    count = 0;
    for (int node_rank = 0; node_rank < world_size; node_rank++) {
      if (node_rank == rank) {
        continue;
      }
      if (ready_flags[count])
        goto cycleEnd;
      if (cudaEventQuery(events[count]) != cudaSuccess)
        goto cycleEnd;

      // Offset of the received chunk
      their_start_offset =
          (num_elems_per_node * node_rank) + std::min(residue, node_rank);
      recv_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
      recv_buf = gradients_recv_ + recv_compressed_size * count;

      recv_compressed_size = round_to(
          get_compressed_size(recv_num_elems, compSetup), ALIGNMENT_UNIT);
      decompress(recv_buf, (unsigned char*)(buf + their_start_offset),
                 recv_num_elems, compSetup, streams[count], false);
      ready_hosts++;
    cycleEnd:
      count++;
    }
  }
}

void shm_SRASetup(shmComm* comm, int rank, int world_size, int num_elems) {
  comm->rank = rank;
  comm->world_size = world_size;
  comm->sendResources = new shmResource[world_size - 1];
  comm->recvResources = new shmResource[world_size - 1];
  comm->recvEvents = new cudaEventSync[world_size - 1];
  comm->sendEvents = new cudaEventSync[world_size - 1];

  int num_elems_per_node = num_elems / world_size;
  int send_size = (num_elems_per_node + 1) * sizeof(float);
  int count = 0;
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank) {
      continue;
    }
    sendInit(&comm->sendResources[count], rank, peer_rank, 2 * send_size);
    count++;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  count = 0;
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank) {
      continue;
    }
    recvInit(&comm->recvResources[count], peer_rank, rank, 2 * send_size);
    count++;
  }

  initEvents(comm->sendEvents, comm->recvEvents, world_size, rank);
}

void shm_SRA(void* device_buf, shmComm* comm, void* comm_buf, int num_elems,
             CompressionSetup* compSetup, cudaStream_t* streams) {
  int world_size = comm->world_size;
  int rank = comm->rank;
  int residue = num_elems % world_size;
  int num_elems_per_node = num_elems / world_size;
  int start_elem = num_elems_per_node * rank + std::min(residue, rank);
  int recv_num_elems = num_elems_per_node + (rank < residue ? 1 : 0);
  int send_num_elems = 0;
  unsigned char* compressed_buf = (unsigned char*)comm_buf;
  float* buf = (float*)device_buf;
  float* peer_buf = nullptr;
  size_t compressed_size = 0;
  size_t total_compressed_size = 0;
  int count = 0;
  std::vector<int> compressed_offsets;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int start_offset =
        (num_elems_per_node * node_rank) + std::min(residue, node_rank);
    send_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    if (compSetup->type == CompType::Memcpy) {
      memcpy2sendBuf(buf + start_offset, send_num_elems * sizeof(float), 0,
                     &comm->sendResources[count], &comm->sendEvents[count],
                     streams[count]);
    } else {
      compressed_size = round_to(get_compressed_size(send_num_elems, compSetup),
                                 ALIGNMENT_UNIT);
      compress((unsigned char*)(buf + start_offset),
               compressed_buf + total_compressed_size, send_num_elems,
               compSetup, streams[count]);
      memcpy2sendBuf(compressed_buf, compressed_size, (size_t)0,
                     &comm->sendResources[count], &comm->sendEvents[count],
                     streams[count]);
      compressed_offsets.push_back(total_compressed_size);
      total_compressed_size += compressed_size;
    }
    count++;
  }
  compressed_buf += total_compressed_size;
  std::vector<int> indices;
  for (int i = 0; i < comm->world_size - 1; i++)
    indices.push_back(i);
  while (indices.size() > 0) {
    for (auto it = indices.begin(); it != indices.end();) {
      int idx = *it;
      if (recvBufAsync((void**)&peer_buf, 0, &comm->recvResources[idx],
                       &comm->recvEvents[idx], streams[0]) == 0) {
        it = indices.erase(it);
        if (compSetup->type == CompType::Memcpy) {
          add(peer_buf, buf + start_elem, recv_num_elems, streams[0]);
        } else {
          compressed_size = round_to(
              get_compressed_size(recv_num_elems, compSetup), ALIGNMENT_UNIT);
          cudaMemcpyAsync(compressed_buf, peer_buf, compressed_size,
                          cudaMemcpyDeviceToDevice, streams[0]);
          decompress((unsigned char*)compressed_buf,
                     (unsigned char*)(buf + start_elem), recv_num_elems,
                     compSetup, streams[0], true);
          //          decompress((unsigned char*)peer_buf,
          //              (unsigned char*)(buf + start_elem), recv_num_elems,
          //              compSetup, streams[0], true);
        }
      } else {
        it++;
      }
    }
  }

  for (int i = 0; i < comm->world_size - 1; i++) {
    MPI_Wait(&comm->sendEvents[i].request, MPI_STATUSES_IGNORE);
  }

  count = 0;
  if (compSetup->type == CompType::Compress) {
    compress((unsigned char*)(buf + start_elem), (unsigned char*)compressed_buf,
             recv_num_elems, compSetup, streams[0]);
    compressed_size = round_to(get_compressed_size(recv_num_elems, compSetup),
                               ALIGNMENT_UNIT);
    cudaStreamSynchronize(streams[0]);
  }

  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    send_num_elems = num_elems_per_node + ((node_rank < residue) ? 1 : 0);
    if (compSetup->type == CompType::Memcpy) {
      memcpy2sendBuf(buf + start_elem, recv_num_elems * sizeof(float),
                     send_num_elems * sizeof(float),
                     &comm->sendResources[count], &comm->sendEvents[count],
                     streams[count]);
    } else {
      memcpy2sendBuf(compressed_buf, compressed_size,
                     send_num_elems * sizeof(float),
                     &comm->sendResources[count], &comm->sendEvents[count],
                     streams[count]);
    }
    count++;
  }

  for (int i = 0; i < comm->world_size - 1; i++)
    indices.push_back(i);
  while (indices.size() > 0) {
    for (auto it = indices.begin(); it != indices.end();) {

      int idx = *it;
      if (recvBufAsync((void**)&peer_buf, recv_num_elems * sizeof(float),
                       &comm->recvResources[idx], &comm->recvEvents[idx],
                       streams[idx]) == 0) {
        it = indices.erase(it);
        int node_rank = comm->recvResources[idx].peerRank;
        int their_start_offset =
            (num_elems_per_node * node_rank) + std::min(residue, node_rank);
        int their_recv_num_elems =
            num_elems_per_node + ((node_rank < residue) ? 1 : 0);
        if (compSetup->type == CompType::Memcpy) {
          cudaMemcpyAsync(buf + their_start_offset, peer_buf,
                          their_recv_num_elems * sizeof(float),
                          cudaMemcpyDeviceToDevice, streams[idx]);
        } else {
          int compressed_offset = compressed_offsets[idx];
          compressed_size =
              get_compressed_size(their_recv_num_elems, compSetup);
          cudaMemcpyAsync(compressed_buf + compressed_offset, peer_buf,
                          compressed_size, cudaMemcpyDeviceToDevice,
                          streams[idx]);
          decompress(compressed_buf + compressed_offset,
                     (unsigned char*)(buf + their_start_offset),
                     their_recv_num_elems, compSetup, streams[idx], false);
          //          decompress((unsigned char*) peer_buf,
          //              (unsigned char*)(buf + their_start_offset),
          //              their_recv_num_elems, compSetup, streams[idx], false);
        }
      } else {
        it++;
      }
    }
  }
  for (int i = 0; i < comm->world_size - 1; i++) {
    MPI_Wait(&comm->sendEvents[i].request, MPI_STATUSES_IGNORE);
  }
}

void shm_SRAFree(shmComm* comm) {
  freeEvents(comm->sendEvents, comm->world_size - 1);
  freeConnections(comm->sendResources, comm->world_size - 1);
  delete[] comm->sendEvents;
  delete[] comm->recvEvents;
  delete[] comm->sendResources;
  delete[] comm->recvResources;
}

void shm_allgatherSetup(shmComm* comm, int rank, int world_size,
                        int num_elems) {
  comm->rank = rank;
  comm->world_size = world_size;
  comm->sendResources = new shmResource[world_size - 1];
  comm->recvResources = new shmResource[world_size - 1];
  comm->recvEvents = new cudaEventSync[world_size - 1];
  comm->sendEvents = new cudaEventSync[world_size - 1];
  size_t buf_size = num_elems * sizeof(float);
  initSendConnections(comm->sendResources, world_size, rank, buf_size);
  MPI_Barrier(MPI_COMM_WORLD);
  initRecvConnections(comm->recvResources, world_size, rank, buf_size);
  initEvents(comm->sendEvents, comm->recvEvents, world_size, rank);
}

void shm_freeAllgatherSetup(shmComm* comm) {
  freeEvents(comm->sendEvents, comm->world_size - 1);
  freeConnections(comm->sendResources, comm->world_size - 1);
  delete[] comm->sendEvents;
  delete[] comm->recvEvents;
  delete[] comm->sendResources;
  delete[] comm->recvResources;
}

void shm_allgather(void* device_buf, shmComm* comm, int num_elems,
                   cudaStream_t stream) {
  float* peer_buf = nullptr;
  for (int i = 0; i < comm->world_size - 1; i++) {
    memcpy2sendBuf(device_buf, num_elems * sizeof(float), 0,
                   &comm->sendResources[i], &comm->sendEvents[i], stream);
  }

  std::vector<int> indices;
  for (int i = 0; i < comm->world_size - 1; i++)
    indices.push_back(i);
  while (indices.size() > 0) {
    for (auto it = indices.begin(); it != indices.end();) {
      int idx = *it;
      if (recvBufAsync((void**)&peer_buf, 0, &comm->recvResources[idx],
                       &comm->recvEvents[idx], stream) == 0) {
        it = indices.erase(it);
        add(peer_buf, (float*)device_buf, num_elems, stream);
      } else {
        it++;
      }
    }
  }
  for (int i = 0; i < comm->world_size - 1; i++) {
    MPI_Wait(&comm->sendEvents[i].request, MPI_STATUSES_IGNORE);
  }
}

void p2p_SRASetup(p2pComm* comm, int rank, int world_size, int buf_size,
                  void** recv_comm_buf) {
  comm->rank = rank;
  comm->world_size = world_size;
  cudaMalloc(recv_comm_buf, (world_size - 1) * buf_size);
  unsigned char* recv = (unsigned char*)(*recv_comm_buf);
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank) {
      continue;
    }
    comm->recv_comms.push_back(CommData(recv, peer_rank));
    comm->send_comms.push_back(CommData(nullptr, peer_rank));
    recv += buf_size;
  }

  std::vector<MPI_Request> send_requests;
  for (int i = 0; i < world_size - 1; i++) {
    CommData& commData = comm->recv_comms[i];
    CUDACHECK(cudaIpcGetMemHandle(&commData.memHandle, (void*) commData.buf));
    send_requests.push_back(MPI_Request());
    MPICHECK(MPI_Isend((void*)&commData.memHandle, sizeof(commData.memHandle),
                       MPI_UNSIGNED_CHAR, commData.remote_rank, 0,
                       MPI_COMM_WORLD, &send_requests.back()));
  }

  for (int i = 0; i < world_size - 1; i++) {
    CommData& commData = comm->send_comms[i];
    MPICHECK(MPI_Recv((void*)&commData.memHandle, sizeof(commData.memHandle),
                      MPI_UNSIGNED_CHAR, commData.remote_rank, 0, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE));
    cudaSetDevice(commData.remote_rank);
    CUDACHECK(cudaIpcOpenMemHandle((void**)&commData.buf, commData.memHandle,
                                   cudaIpcMemLazyEnablePeerAccess));
    cudaSetDevice(rank);
    CUDACHECK(cudaEventCreate(&commData.event,
                              cudaEventDisableTiming | cudaEventInterprocess));
    CUDACHECK(cudaIpcGetEventHandle(&commData.eventHandle, commData.event));
    send_requests.push_back(MPI_Request());
    MPICHECK(MPI_Isend((void*)&commData.eventHandle,
                       sizeof(commData.eventHandle), MPI_UNSIGNED_CHAR,
                       commData.remote_rank, 0, MPI_COMM_WORLD,
                       &send_requests.back()));
  }

  for (int i = 0; i < world_size - 1; i++) {
    CommData& commData = comm->recv_comms[i];
    MPICHECK(MPI_Recv((void*)(&commData.eventHandle),
                      sizeof(commData.eventHandle), MPI_UNSIGNED_CHAR,
                      commData.remote_rank, 0, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE));
    CUDACHECK(cudaIpcOpenEventHandle(&commData.event, commData.eventHandle));
  }
  MPICHECK(MPI_Waitall(send_requests.size(), send_requests.data(),
                       MPI_STATUS_IGNORE));
}

void p2pSend(CommData& commData, void* buf, int buf_size, cudaStream_t stream) {
  cudaMemcpyAsync(commData.buf, buf, buf_size, cudaMemcpyDefault, stream);
  cudaEventRecord(commData.event);
  MPI_Isend(&commData.dummy, sizeof(commData.dummy), MPI_UNSIGNED_CHAR, commData.remote_rank, 0,
            MPI_COMM_WORLD, &commData.request);
}

int p2pRecvAsync(CommData& commData, void** buf, cudaStream_t stream) {
  *buf = nullptr;
  if (commData.request == MPI_REQUEST_NULL) {
    MPICHECK(MPI_Irecv(&commData.dummy, 1, MPI_UNSIGNED_CHAR, commData.remote_rank, 0,
                       MPI_COMM_WORLD, &commData.request));
  }
  int flag;
  MPICHECK(MPI_Test(&commData.request, &flag, MPI_STATUSES_IGNORE));
  if (!flag)
    return 1;
  CUDACHECK(cudaStreamWaitEvent(stream, commData.event, 0));
  *buf = commData.buf;
  commData.request = MPI_REQUEST_NULL;
  return 0;
}

void p2p_allgather(void* device_buf, p2pComm* comm, int num_elems,
                   cudaStream_t stream) {
  float* peer_buf = nullptr;
  for (int i = 0; i < comm->world_size - 1; i++) {
    CommData& commData = comm->send_comms[i];
    p2pSend(commData, device_buf, num_elems * sizeof(float), stream);
  }

  std::vector<int> indices;
  for (int i = 0; i < comm->world_size - 1; i++)
    indices.push_back(i);
  while (indices.size() > 0) {
    for (auto it = indices.begin(); it != indices.end();) {
      int idx = *it;
      CommData& commData = comm->recv_comms[idx];
      if (p2pRecvAsync(commData, (void**)&peer_buf, stream) == 0) {
        it = indices.erase(it);
        add(peer_buf, (float*)device_buf, num_elems, stream);
      } else {
        it++;
      }
    }
  }
  for (int i = 0; i < comm->world_size - 1; i++) {
    MPI_Wait(&comm->send_comms[i].request, MPI_STATUSES_IGNORE);
  }
}
