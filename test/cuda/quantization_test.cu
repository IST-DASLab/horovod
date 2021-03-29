#include "../../horovod/common/ops/compressed/compression/cuda/cuda_compression_functions.h"
#include "cuda_runtime.h"
#include <iostream>
#include <cuda_profiler_api.h>

#define NUM_ELEMS 1024 * 1024
#define NUM_STREAMS 2
#define BUCKET_SIZE 512
#define BITS 8
#define NUM_ITERS 10000

namespace hvd = horovod::common::cuda;

void generate_data(float* buf, int len) {
  for (int i = 0; i < len; i++) {
    buf[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
}

float get_L2_error(float* a, float* b, int num_elems) {
  float error = 0.0;
  for (int i = 0; i < num_elems; i++) {
    error += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return error;
}

int main(int argc, char* argv[]) {
  float* host_input = new float[NUM_ELEMS];
  float* host_output = new float[NUM_ELEMS];
  float* device_input[NUM_STREAMS];
  float* *device_output[NUM_STREAMS];
  unsigned char* device_compressed[NUM_STREAMS];
  CurandState* rand_states[NUM_STREAMS];
  std::srand(0);
  size_t buf_size = NUM_ELEMS * sizeof(float);
  int num_buckets = (NUM_ELEMS + BUCKET_SIZE - 1) / BUCKET_SIZE;
  size_t compressed_size =
      ((NUM_ELEMS * BITS) / 8) + (2 * num_buckets * sizeof(float));
  size_t curand_size = hvd::CUDA_get_curand_array_size(NUM_ELEMS);
  generate_data(host_input, NUM_ELEMS);
  cudaStream_t streams[NUM_STREAMS];
  cudaStream_t stream = 0;
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    cudaMalloc(&device_input[i], buf_size);
    cudaMalloc(&device_output[i], buf_size);
    cudaMemcpyAsync(device_input[i], host_input, buf_size, cudaMemcpyHostToDevice,
                    stream);
    cudaMalloc(&device_compressed[i], compressed_size);
    cudaMalloc(&rand_states[i], curand_size);
    hvd::CUDA_init_curand(rand_states[i], NUM_ELEMS, 0, stream);
  }
  cudaDeviceSynchronize();
//  cudaMalloc(&device_input[0], buf_size);
//  cudaMalloc(&device_output[0], buf_size);
//  cudaMalloc(&device_compressed[0], compressed_size);
  bool benchmark = false;
  if (benchmark) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    cudaProfilerStart();
    for (int i = 0; i < NUM_ITERS; i++) {
      int idx = i % NUM_STREAMS;
      hvd::CUDA_quantize_maxmin<float>((unsigned char*)device_input[idx], device_compressed[idx], nullptr,
            NUM_ELEMS, BITS, BUCKET_SIZE, rand_states[idx], streams[idx]);
      hvd::CUDA_dequantize_maxmin<float, false>(device_compressed[idx],
                                           (unsigned char*)device_output[idx],
                                           NUM_ELEMS, BITS, BUCKET_SIZE, streams[idx]);
    }
    cudaProfilerStop();
    cudaDeviceSynchronize();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Throughput: " << NUM_ELEMS * sizeof(float) * NUM_ITERS / (milliseconds * 1000 * 1000) << " Kbs/sec" << std::endl;
  } else {
    hvd::CUDA_quantize_maxmin<float>((unsigned char*)device_input[0], device_compressed[0], nullptr,
                               NUM_ELEMS, BITS, BUCKET_SIZE, rand_states[0], stream);
    hvd::CUDA_dequantize_maxmin<float, false>(device_compressed[0],
                                         (unsigned char*)device_output[0],
                                         NUM_ELEMS, BITS, BUCKET_SIZE, stream);

    cudaMemcpyAsync(host_output, device_output[0], buf_size, cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);

    std::cout << "L2 error: " << get_L2_error(host_input, host_output, NUM_ELEMS) << std::endl;
    std::cout << "Input: ";
    for (int i = 0; i < 8; i++)
      std::cout << host_input[NUM_ELEMS - 8 + i] << " ";
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < 8; i++)
      std::cout << host_output[NUM_ELEMS - 8 + i] << " ";
    std::cout << std::endl;
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaFree(device_input[i]);
    cudaFree(device_output[i]);
    cudaFree(device_compressed[i]);
    cudaFree(rand_states[i]);
  }
  delete[] host_input;
  delete[] host_output;
}