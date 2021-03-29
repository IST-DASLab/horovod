#include "cuda_runtime.h"
#include <algorithm>
#include <assert.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "common.h"
#include "reductions.h"

const int NUM_BUFS = 25;
const int BITS = 4;
const int BUCKET_SIZE = 512;
const int MAX_SIZE = 10000000;
const float fake_ratio = 0.125;
const unsigned int SPLIT_BUFFSIZE = 16384 * 16384;

void createStream(cudaStream_t* stream) {
  //  int greatest_priority;
  //  cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority);
  //  cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking,
  //                               greatest_priority);
  cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
}

void initComms(ncclComm_t* comm, int* rank, int* world_size) {
  int myRank, nRanks;

  // initializing MPI
  MPICHECK(MPI_Init(nullptr, nullptr));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  ncclUniqueId id;
  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  CUDACHECK(cudaSetDevice(myRank));

  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(comm, nRanks, id, myRank));
  *rank = myRank;
  *world_size = nRanks;
}

void generate_data(float* buf, int len) {
  for (int i = 0; i < len; i++) {
    buf[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
}

void initCompressionSetup(CompressionSetup* compSetup, CompType compType,
                          int max_num_elems, cudaStream_t stream) {
  compSetup->bucket_size = BUCKET_SIZE;
  compSetup->bits = BITS;
  compSetup->type = compType;

  if (compType == CompType::Compress) {
    CUDACHECK(
        cudaMalloc(&compSetup->rand_states,
                   sizeof(CurandState) * get_curand_array_size(max_num_elems)));
    init_curand(compSetup->rand_states, max_num_elems, time(NULL), stream);
  }
}

void setupReduction(int rank, int world_size, ReductionType reductionType,
                    BackendType backendType, void** device_comm, int num_elems,
                    CompressionSetup* compSetup, shmComm* sComm,
                    p2pComm* pComm) {
  if (compSetup->type == CompType::None)
    return;
  if (backendType == BackendType::SHM_) {
    assert(reductionType == ReductionType::SRA);
    if (reductionType == ReductionType::SRA) {
      shm_SRASetup(sComm, rank, world_size, num_elems);
    }
    CUDACHECK(cudaMalloc(
        device_comm,
        2 * world_size *
            round_to(get_compressed_size(num_elems * sizeof(float), compSetup),
                     ALIGNMENT_UNIT)));
  } else if (backendType == BackendType::P2P_) {
    p2p_SRASetup(
        pComm, rank, world_size,
        round_to(get_compressed_size(num_elems * sizeof(float), compSetup),
                 ALIGNMENT_UNIT), device_comm);
  } else {
    CUDACHECK(cudaMalloc(
        device_comm,
        2 * world_size *
            round_to(get_compressed_size(num_elems * sizeof(float), compSetup),
                     ALIGNMENT_UNIT)));
  }
}

void freeReduction(shmComm* sComm, void* device_comm,
                   CompressionSetup* compSetup, BackendType backendType) {
  if (compSetup->type == CompType::None)
    return;
  if (backendType == BackendType::SHM_) {
    shm_SRAFree(sComm);
    return;
  }
  if (compSetup->type == CompType::Compress) {
    cudaFree(compSetup->rand_states);
  }
  cudaFree(device_comm);
}

// usage
// ./app number_of_iterations compression_type(None - 0, Memcpy - 1, Compress -
// 2) backend (MPI - 0 , NCCL_ - 1, SHM_ - 2, P2P_ - 3) reduction_type (SRA - 0, Ring - 1, Allgather - 2)
int main(int argc, char* argv[]) {
  int num_iters = 10;
  CompType compType = CompType::None;
  BackendType backType;
  ReductionType redType;
  parse_args(argc, argv, num_iters, compType, backType, redType);
  //  set_blocks(1024);
  ncclComm_t comm;
  shmComm sComm;
  p2pComm pComm;
  int rank, world_size;
  initComms(&comm, &rank, &world_size);

  int buf_sizes[NUM_BUFS];
  float* host_bufs[NUM_BUFS];
  cudaStream_t* streams;

  std::srand(time(NULL));
  for (int i = 0; i < NUM_BUFS; i++) {
    int len = std::rand() % MAX_SIZE;
    buf_sizes[i] = len;
  }

  int max_num_elems = 0;
  std::srand(rank);
  for (int i = 0; i < NUM_BUFS; i++) {
    int len = buf_sizes[i];
    host_bufs[i] = new float[len];
    generate_data(host_bufs[i], len);
    max_num_elems = std::max(len, max_num_elems);
  }
  int num_streams = world_size;
  streams = new cudaStream_t[num_streams];
  for (int i = 0; i < num_streams; i++) {
    //    cudaStreamCreate(streams+i);
    createStream(streams + i);
  }

  float* device_inputs[NUM_BUFS];
  unsigned char* device_comm_buf;
  CompressionSetup compSetup;
  initCompressionSetup(&compSetup, compType, max_num_elems, streams[0]);

  for (int i = 0; i < NUM_BUFS; i++) {
    CUDACHECK(cudaMalloc(&device_inputs[i], buf_sizes[i] * sizeof(float)));
    CUDACHECK(cudaMemcpy((void*)device_inputs[i], host_bufs[i],
                         buf_sizes[i] * sizeof(float), cudaMemcpyHostToDevice));
  }
  setupReduction(rank, world_size, redType, backType, (void**)&device_comm_buf,
                 max_num_elems, &compSetup, &sComm, &pComm);
  cudaDeviceSynchronize();
  unsigned long long total_num_elems = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, streams[0]);

  cudaProfilerStart();
  for (int i = 0; i < num_iters; i++) {
    int buf_it = i % NUM_BUFS;
    int buf_size = buf_sizes[buf_it];
    unsigned char* comm_buf = device_comm_buf;
    //    if (backType == BackendType::NCCL_ or backType == BackendType::MPI_) {
    //      comm_buf = device_comm_buf + 2 * world_size * buf_it *
    //                                   round_to(get_compressed_size(max_num_elems
    //                                   * sizeof(float),
    //                                                                &compSetup),
    //                                            ALIGNMENT_UNIT);
    //    }
    for (unsigned int offset = 0; offset < buf_size; offset += SPLIT_BUFFSIZE) {
      int split_size = std::min(SPLIT_BUFFSIZE, buf_size - offset);
      float* split_buf = device_inputs[buf_it] + offset;
      if (compType == CompType::None) {
        int64_t total_num_elems = (int64_t)(split_size * fake_ratio);
        ncclAllReduce((const void*)split_buf, (void*)split_buf, total_num_elems,
                      ncclFloat, ncclSum, comm, *streams);
      } else if (backType == BackendType::NCCL_) {
        if (redType == ReductionType::SRA) {
          nccl_reduction_sra(split_buf, comm_buf, split_size, world_size, rank,
                             &compSetup, streams, &comm);
        } else {
          nccl_reduction_ring(split_buf, comm_buf, split_size, world_size, rank,
                              &compSetup, streams, &comm);
        }
      } else if (backType == BackendType::MPI_) {
        assert(redType == ReductionType::SRA);
        mpi_reduction_sra(split_buf, comm_buf, split_size, world_size, rank,
                          &compSetup, streams);
      } else if (backType == BackendType::SHM_)  {
        assert(redType == ReductionType::SRA);
        shm_SRA(split_buf, &sComm, comm_buf, split_size, &compSetup, streams);
      } else if (backType == BackendType::P2P_) {
        assert(redType == ReductionType::Allgather);
        p2p_allgather(split_buf, &pComm, split_size, streams[0]);
      }
    }

    total_num_elems += buf_sizes[buf_it];
  }
  for (int i = 0; i < world_size; i++)
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  cudaProfilerStop();

  cudaEventRecord(stop, streams[0]);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  if (rank == 0) {
    std::cout << "Throughput: "
              << (total_num_elems * world_size * sizeof(float)) /
                     (milliseconds * 1000 * 1000)
              << " Kbs/sec" << std::endl;
    std::cout << "Average buffer size: " << total_num_elems / num_iters
              << std::endl;
    std::sort(buf_sizes, buf_sizes + NUM_BUFS);
    std::cout << "Median size: " << buf_sizes[(NUM_BUFS - 1) / 2] << std::endl;
    std::cout << "Time elapsed: " << milliseconds << std::endl;
  }

  // free device buffers
  for (int i = 0; i < NUM_BUFS; i++) {
    CUDACHECK(cudaFree(device_inputs[i]));
    delete[] host_bufs[i];
  }

  freeReduction(&sComm, device_comm_buf, &compSetup, backType);
  // finalizing NCCL
  ncclCommDestroy(comm);
  // finalizing MPI
  MPICHECK(MPI_Finalize());
  return 0;
}