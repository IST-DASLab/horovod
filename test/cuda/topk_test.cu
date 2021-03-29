#include "../../horovod/common/ops/compressed/compression/cuda/cuda_compression_functions.h"
#include "cuda_runtime.h"
#include <cuda_profiler_api.h>
#include <iostream>
#include <random>

#define NUM_ELEMS 11173962
#define NUM_STREAMS 1
#define TOPK_RATIO 0.9
#define NUM_ITERS 10000

namespace hvd = horovod::common::cuda;

void generate_data(float* buf, int len) {
  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::normal_distribution<> d{0,0.005};
  float sum = 0;
  for (int i = 0; i < len; i++) {
//    buf[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    buf[i] = static_cast<float>(d(gen)) ;
    sum += fabs(buf[i]);
  }
}

__global__ void sum_func(float* input, int num, float* sum) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < num; i+=stride) {
    atomicAdd(sum, fabs(input[i]));
  }
}

void check_atomic() {
  float* host_buf = new float[NUM_ELEMS];
  generate_data(host_buf, NUM_ELEMS);
  float* device_input;
  float* sum;
  size_t buf_size = NUM_ELEMS * sizeof(float);
  cudaStream_t stream = 0;
  cudaMalloc(&device_input, buf_size);
  cudaMalloc(&sum, sizeof(float));
  cudaMemcpyAsync(device_input, host_buf, buf_size,
                  cudaMemcpyHostToDevice, stream);
  dim3 num_threads = 1024;
  dim3 blocks = 2048;
  sum_func<<<blocks, num_threads, 0, stream>>>(device_input, NUM_ELEMS, sum);
  CUDA_CHECK(cudaGetLastError());
  float sum_host;
  cudaMemcpyAsync(&sum_host, sum, sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  printf("Sum: %f\n", sum_host);
}


int get_non_zeros(float* a, int num_elems) {
  int num = 0;
  for (int i = 0; i < num_elems; i++) {
    if (fabs(a[i]) > 1e-6)
      num++;
  }
  return num;
}

int main(int argc, char* argv[]) {
  float* host_input = new float[NUM_ELEMS];
  float* host_output = new float[NUM_ELEMS];
  float* device_input[NUM_STREAMS];
  float** device_output[NUM_STREAMS];
  unsigned char* device_compressed[NUM_STREAMS];
  unsigned char* utility_buf[NUM_STREAMS];
  std::srand(42);
  size_t buf_size = NUM_ELEMS * sizeof(float);
  int num_result = ceil(TOPK_RATIO * NUM_ELEMS);
  size_t compressed_size = num_result * (sizeof(unsigned int) + sizeof(float));
  size_t utility_buf_size = hvd::CUDA_get_topk_utility_buf_size();
  generate_data(host_input, NUM_ELEMS);

//  check_atomic();
//  return 0;

  cudaStream_t streams[NUM_STREAMS];
  cudaStream_t stream = 0;
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    cudaMalloc(&device_input[i], buf_size);
    cudaMalloc(&device_output[i], buf_size);
    cudaMemcpyAsync(device_input[i], host_input, buf_size,
                    cudaMemcpyHostToDevice, stream);
    cudaMalloc(&device_compressed[i], compressed_size);
    cudaMalloc(&utility_buf[i], utility_buf_size);
    cudaMemset(utility_buf[i], 0, utility_buf_size);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  bool benchmark = false;
  if (benchmark) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    cudaProfilerStart();
    for (int i = 0; i < NUM_ITERS; i++) {
      int idx = i % NUM_STREAMS;
      int numel = std::rand() % NUM_ELEMS;
      num_result = ceil(TOPK_RATIO * numel);
      hvd::CUDA_topk_compress<float>(
          (unsigned char*)device_input[idx], device_compressed[idx],
          utility_buf[idx], nullptr, numel, num_result, streams[idx]);
      hvd::CUDA_topk_decompress<float, false>(
          device_compressed[idx], (unsigned char*)device_output[idx], NUM_ELEMS,
          num_result, streams[idx]);
    }
    cudaProfilerStop();
    cudaDeviceSynchronize();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Throughput: "
              << NUM_ELEMS * sizeof(float) * NUM_ITERS /
                     (milliseconds * 1000 * 1000)
              << " Kbs/sec" << std::endl;
  } else {
    CUDA_CHECK(cudaGetLastError());
    hvd::CUDA_topk_compress<float>((unsigned char*)device_input[0],
                                   device_compressed[0], utility_buf[0],
                                   nullptr, NUM_ELEMS, num_result, stream);
    hvd::CUDA_topk_decompress<float, false>(device_compressed[0],
                                            (unsigned char*)device_output[0],
                                            NUM_ELEMS, num_result, stream);

    cudaMemcpyAsync(host_output, device_output[0], buf_size,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::cout << "Non zeros: " << get_non_zeros(host_output, NUM_ELEMS)
              << " total: " << NUM_ELEMS << " expected ratio: " << TOPK_RATIO
              << std::endl;
    std::cout << "Input: ";
    for (int i = 0; i < 10; i++)
      std::cout << host_input[i] << " ";
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < 10; i++)
      std::cout << host_output[i] << " ";
    std::cout << std::endl;
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaFree(device_input[i]);
    cudaFree(device_output[i]);
    cudaFree(device_compressed[i]);
    cudaFree(utility_buf[i]);
  }
  delete[] host_input;
  delete[] host_output;
}