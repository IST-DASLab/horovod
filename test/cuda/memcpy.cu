#include <iostream>
#include <algorithm>

#include "common.h"

#define NUM_ELEMS 1024

struct CommData {
  cudaIpcMemHandle_t memHandle;
  cudaIpcEventHandle_t eventHandle;
};

int main() {
  int rank, world_size;
  // initializing MPI
  MPICHECK(MPI_Init(nullptr, nullptr));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  CUDACHECK(cudaSetDevice(rank));

  float* host_buf = new float[NUM_ELEMS];
  float* device_buf, *peer_buf;
  CommData commData;
  cudaEvent_t event;
  for (int i = 0; i <  1; i++) {
    if (rank == 0) {
      if (i == 0) {
        cudaMalloc(&device_buf, NUM_ELEMS * sizeof(float));
      }
      cudaMalloc(&device_buf, NUM_ELEMS * sizeof(float));

      std::fill(host_buf, host_buf + NUM_ELEMS, i + 1);
      CUDACHECK(cudaMemcpy(device_buf, host_buf, NUM_ELEMS * sizeof(float),
                           cudaMemcpyHostToDevice));
      cudaDeviceSynchronize();
//      MPI_Barrier(MPI_COMM_WORLD);

      if (i == 0) {
        MPICHECK(MPI_Recv((void*)(&commData.memHandle),
                          sizeof(commData.memHandle), MPI_UNSIGNED_CHAR, 1, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        cudaSetDevice(1);
        CUDACHECK(cudaIpcOpenMemHandle((void**)&peer_buf, commData.memHandle,
                                       cudaIpcMemLazyEnablePeerAccess));
        cudaSetDevice(0);
        CUDACHECK(cudaEventCreate(&event, cudaEventDisableTiming |
                                          cudaEventInterprocess));
        CUDACHECK(cudaIpcGetEventHandle(
            (cudaIpcEventHandle_t*)&commData.eventHandle, event));
        MPICHECK(MPI_Send((void*)(&commData.eventHandle),
                          sizeof(commData.eventHandle), MPI_UNSIGNED_CHAR, 1, 0,
                          MPI_COMM_WORLD));
      }

      printf("Pointer send to %p\n", peer_buf);
      CUDACHECK(cudaMemcpy(peer_buf, device_buf, NUM_ELEMS * sizeof(float), cudaMemcpyDefault));
//      CUDACHECK(cudaMemcpyPeer(peer_buf, 1, device_buf, 0,
//                               NUM_ELEMS * sizeof(float)));
      cudaEventRecord(event);
      cudaEventSynchronize(event);
//      MPI_Barrier(MPI_COMM_WORLD);
      std::cout << rank << ". Source: " << host_buf[0] << " " << host_buf[NUM_ELEMS - 1] << std::endl;
    } else {
      if (i == 0) {
        cudaMalloc(&peer_buf, NUM_ELEMS * sizeof(float));
      }
      std::fill(host_buf, host_buf + NUM_ELEMS, 0);
      CUDACHECK(cudaMemcpy(peer_buf, host_buf, NUM_ELEMS * sizeof(float),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaDeviceSynchronize());
//      MPI_Barrier(MPI_COMM_WORLD);

      if (i == 0) {
        CUDACHECK(cudaIpcGetMemHandle(
            (&commData.memHandle), (void*)peer_buf));
        MPICHECK(MPI_Send((void*)(&commData.memHandle),
                          sizeof(commData.memHandle), MPI_UNSIGNED_CHAR, 0, 0,
                          MPI_COMM_WORLD));
        MPICHECK(MPI_Recv((void*)(&commData.eventHandle),
                          sizeof(commData.eventHandle), MPI_UNSIGNED_CHAR, 0, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        cudaIpcOpenEventHandle(&event, commData.eventHandle);
      }

      cudaEventSynchronize(event);
//      MPI_Barrier(MPI_COMM_WORLD);
      printf("Pointer recv %p\n", peer_buf);
      CUDACHECK(cudaMemcpy(host_buf, peer_buf, NUM_ELEMS * sizeof(float),
                           cudaMemcpyDeviceToHost));
      CUDACHECK(cudaDeviceSynchronize());
      std::cout << rank << ". Result: " << host_buf[0] << " " << host_buf[NUM_ELEMS - 1] << std::endl;
    }
  }
  CUDACHECK(cudaDeviceSynchronize());
//  if (rank == 0) {
//    cudaSetDevice(1);
//    CUDACHECK(cudaIpcCloseMemHandle(peer_buf));
//    cudaSetDevice(0);
//    CUDACHECK(cudaFree(device_buf));
//    CUDACHECK(cudaEventDestroy(event));
//  } else {
//    CUDACHECK(cudaFree(peer_buf));
//  }
  delete [] host_buf;
  MPICHECK(MPI_Finalize());
  return 0;
}