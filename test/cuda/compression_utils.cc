#include "compression_utils.h"
#include "cuda_functions.h"
#include "cuda_vector_operations.h"

int64_t round_to(int64_t x, int64_t m) { return x + ((m - x % m) % m); }

void parse_args(int argc, char** argv, int& num_iters, CompType& comp_type,
                BackendType& back_type, ReductionType& red_type) {
  if (argc >= 2) {
    num_iters = std::atoi(argv[1]);
  }
  int parsed = 0;
  if (argc >= 3) {
    parsed = std::atoi(argv[2]);
    assert(parsed < 3);
    comp_type = (CompType) parsed;
  }
  if (argc >= 4) {
    parsed = std::atoi(argv[3]);
    assert(parsed < 4);
    back_type = (BackendType) parsed;
  }
  if (argc >= 5) {
    parsed = std::atoi(argv[4]);
    assert(parsed < 3);
    red_type = (ReductionType) parsed;
  }
}

int get_compressed_size(int num_elems, CompressionSetup* setup) {
  if (setup->type == CompType::Compress) {
    int num_buckets = (num_elems + setup->bucket_size - 1) / setup->bucket_size;
    return ((num_elems * setup->bits) / 8) +
           (2 * num_buckets * sizeof(float));
  }
  else
    return num_elems * sizeof(float);
}

int get_curand_array_size(int num_elems) {
  return CUDA_get_curand_array_size(num_elems);
}

void compress(unsigned char* input, unsigned char* output, int size,
                     CompressionSetup* setup,
                     cudaStream_t stream) {
  if (setup->type == CompType::Compress) {
    CUDA_quantize_maxmin<float>(input, output, nullptr, size, setup->bits, setup->bucket_size,
                              setup->rand_states, stream);
  } else {
    cudaMemcpyAsync(output, input, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }
}

void decompress(unsigned char* input, unsigned char* output, int size,
                       CompressionSetup* setup,
                       cudaStream_t stream, bool add) {
  if (setup->type == CompType::Compress) {
    if (add) {
      CUDA_dequantize_maxmin<float, true>(input, output, size, setup->bits,
                                    setup->bucket_size, stream);
    } else {
      CUDA_dequantize_maxmin<float, false>(input, output, size, setup->bits,
                                    setup->bucket_size, stream);
    }
  } else {
    if (add) {
      CUDA_add(size, (float*) input, (float*) output, (float*) output, stream);
    } else {
      cudaMemcpyAsync(output, input, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
  }
}

void add(float* input, float* output, int size, cudaStream_t stream) {
  CUDA_add(size, input, output, output, stream);
}

void init_curand(CurandState* states, int num_elems, unsigned int seed,
                 cudaStream_t stream) {
  CUDA_init_curand(states, num_elems, seed, stream);
}