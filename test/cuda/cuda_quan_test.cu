#include "../../horovod/common/ops/compressed/compression/cuda/cuda_compression_functions.h"
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

using hvd = horovod::common::cuda;

const int NUM_BUFS = 10;
const int BITS = 4;
const int BUCKET_SIZE = 512;
const int MAX_SIZE = 100000;

void createStream(cudaStream_t* stream) {
  int greatest_priority;
  cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority);
  cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking,
                               greatest_priority);
}

void generate_data(float* buf, int len) {
  for (int i = 0; i < len; i++) {
    buf[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
}

void test(int argc, char* argv[]) {
  int len = MAX_SIZE;
  if (argc > 1) {
    len = atoi(argv[1]);
  }
  float* host_buf;
  float* device_input;
  unsigned char* device_output;
  CurandState* rand_states;

  std::srand(std::time(nullptr));
  int num_buckets = (len + BUCKET_SIZE - 1) / BUCKET_SIZE;
  host_buf = new float[len];
  float* result_buf = new float[2 * num_buckets];
  float* device_result_copy = new float[2 * num_buckets];
  generate_data(host_buf, len);

  for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
    int idx = bucket_idx * BUCKET_SIZE;
    result_buf[2 * bucket_idx] = host_buf[idx];
    result_buf[2 * bucket_idx + 1] = host_buf[idx];
    for (int i = idx + 1; i < len && i < idx + BUCKET_SIZE; i++) {
      result_buf[2 * bucket_idx] =
          fmaxf(result_buf[2 * bucket_idx], host_buf[i]);
      result_buf[2 * bucket_idx + 1] =
          fminf(result_buf[2 * bucket_idx + 1], host_buf[i]);
    }
  }

  cudaStream_t stream;
  createStream(&stream);
  cudaMalloc(&device_input, len * sizeof(float));
  cudaMemcpy((void*)device_input, host_buf, len * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&device_output, 2 * sizeof(float) * num_buckets);
  cudaMalloc(&rand_states,
             sizeof(CurandState) * hvd::CUDA_get_curand_array_size(len));
  hvd::CUDA_init_curand(rand_states, len, time(NULL), stream);

  hvd::CUDA_quantize_maxmin<float>((unsigned char*)device_input, device_output, nullptr, len, BITS,
              BUCKET_SIZE, stream);

  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  cudaMemcpy((void*)device_result_copy, device_output,
             2 * num_buckets * sizeof(float), cudaMemcpyDeviceToHost);
  int count = 0;
  for (int i = 0; i < 2 * num_buckets; i++) {
    if (device_result_copy[i] != result_buf[i]) {
      std::cout << i << std::endl;
      std::cout << device_result_copy[i] << " vs " << result_buf[i]
                << std::endl;
      count++;
    }
  }
  std::cout << "Number of errors: " << count << " out of " << 2 * num_buckets
            << std::endl;
}

void bench(int argc, char* argv[]) {
  int num_iters = 100;
  CompType type = get_comp_type(argc, argv, num_iters);

  float* host_bufs[NUM_BUFS];
  int buf_sizes[NUM_BUFS];
  float* device_inputs[NUM_BUFS];
  unsigned char* device_outputs[NUM_BUFS];
  CurandState* rand_states;

  std::srand(std::time(nullptr));
  int max_num_elems = 0;
  for (int i = 0; i < NUM_BUFS; i++) {
    int len = std::rand() % MAX_SIZE;
    std::cout << "Size " << len << std::endl;
    buf_sizes[i] = len;
    host_bufs[i] = new float[len];
    generate_data(host_bufs[i], len);
    max_num_elems = std::max(len, max_num_elems);
  }
  cudaStream_t stream;
  createStream(&stream);

  cudaMalloc(&rand_states,
             sizeof(CurandState) * hvd::CUDA_get_curand_array_size(max_num_elems));

  for (int i = 0; i < NUM_BUFS; i++) {
    cudaMalloc(&device_inputs[i], buf_sizes[i] * sizeof(float));
    cudaMemcpy((void*)device_inputs[i], host_bufs[i],
               buf_sizes[i] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&device_outputs[i],
               get_compressed_size(buf_sizes[i], type, BITS, BUCKET_SIZE));
  }


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);

  for (int iter = 0; iter < num_iters; iter++) {
    int buf_num = iter % NUM_BUFS;
    hvd::CUDA_quantize_maxmin<float>((unsigned char*)(device_inputs[buf_num]), device_outputs[buf_num], nullptr,
             buf_sizes[buf_num], BITS, BUCKET_SIZE, rand_states, stream)
  }

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Elapsed time %f seconds\n", milliseconds / 1000);
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
//  test(argc, argv);
   bench(argc, argv);
}