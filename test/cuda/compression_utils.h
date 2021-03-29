#ifndef ORIG_HOROVOD_COMPRESSION_UTILS_H
#define ORIG_HOROVOD_COMPRESSION_UTILS_H
#include "cuda_def.h"

enum CompType { None, Memcpy, Compress };
enum BackendType { MPI_, NCCL_, SHM_, P2P_};
enum ReductionType { SRA, Ring, Allgather };

struct CompressionSetup {
  int bits;
  int bucket_size;
  CurandState* rand_states;
  CompType type;
};

const int ALIGNMENT_UNIT = 2 * sizeof(float);

#define ALIGNED_SIZE(size) round_to(size, ALIGNMENT_UNIT)

int64_t round_to(int64_t x, int64_t m);

void parse_args(int argc, char** argv, int& num_iters, CompType& comp_type,
                BackendType& back_type, ReductionType& red_type);

int get_compressed_size(int num_elems, CompressionSetup* setup);

int get_curand_array_size(int num_elems);

void compress(unsigned char* input, unsigned char* output, int size,
                     CompressionSetup* setup,
                     cudaStream_t stream);

void decompress(unsigned char* input, unsigned char* output, int size,
                       CompressionSetup* setup,
                       cudaStream_t stream, bool add);

void add(float* input, float* output, int size, cudaStream_t stream);

void init_curand(CurandState* states, int num_elems, unsigned int seed,
                 cudaStream_t stream);
#endif // ORIG_HOROVOD_COMPRESSION_UTILS_H
