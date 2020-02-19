#include "mpi_gpu_compressed_operations.h"
#include "reducers/all_broadcast.h"
#include "reducers/ring.h"
#include "reducers/scatter_allgather.h"
#include "utils.h"

namespace horovod {
namespace common {

MPI_GPUCompressedAllReduce::MPI_GPUCompressedAllReduce(
    MPIContext* mpi_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state)
    : MPI_GPUAllreduce(mpi_context, gpu_context, global_state) {
  auto reduction_type = GetEnumEnvOrDefault<ReductionType>(
      HOROVOD_REDUCTION, ReductionType::NoneReduction);
  auto compression_type = GetEnumEnvOrDefault<CompressionType>(
      HOROVOD_COMPRESSION, CompressionType::NoneCompression);

  auto quantization_bits = GetIntEnvOrDefault(HOROVOD_QUANTIZATION_BITS, 32);
  Compressor* compressor;
  if (quantization_bits == 32 || compression_type == NoneCompression) {
    compressor = new GPUDummyCompressor(gpu_context, global_state);
  } else {
    switch (compression_type) {
    case CompressionType::MaxMin:
      compressor =
          new GPUMaxMinQuantizer(gpu_context, global_state, quantization_bits);
      break;
    case CompressionType::ExpL2:
      compressor = new GPUNormL2Quantizer(
          gpu_context, global_state, quantization_bits, QUANTIZE_MULTIPLIER);
      break;
    case CompressionType::Uni:
      compressor = new GPUNormLinfQuantizer(gpu_context, global_state,
                                            quantization_bits);
      break;
    case CompressionType::ExpLinf:
      compressor = new GPUNormLinfQuantizer(
          gpu_context, global_state, quantization_bits, QUANTIZE_MULTIPLIER);
      break;
    default:
      throw std::logic_error("Invalid compression type.");
    }
  }
  auto summator = new GPUSummator(global_state, gpu_context);
  switch (reduction_type) {
  case ReductionType::AllBroadcast:
    mpiReducer = new MPI_GPUAllreduce_AllBroadcast(mpi_context, gpu_context,
                                                   global_state, compressor, summator);
    break;
  case ReductionType::Ring:
    mpiReducer = new MPI_GPUAllreduce_Ring(mpi_context, gpu_context,
                                           global_state, compressor, summator);
    break;
  case ReductionType::ScatterAllgather:
    mpiReducer = new MPI_GPUAllreduce_ScatterReduceAllgather(
        mpi_context, gpu_context, global_state, compressor, summator);
    break;
  case ReductionType::NoneReduction:
    mpiReducer = nullptr;
    break;
  }
}

Status MPI_GPUCompressedAllReduce::Allreduce(
    int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int buffer_len) {
  Status status = mpiReducer->Init(entries);
  if (!status.ok()) {
    return status;
  }
  int64_t tensor_fusion_threshold =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  if (buffer_len > tensor_fusion_threshold) {
    int num_divisions =
        (buffer_len + tensor_fusion_threshold - 1) / tensor_fusion_threshold;
    int num_elements_division = 0;
    int64_t global_offset = 0;
    for (int division = 0; division < num_divisions; division++) {
      num_elements_division =
          (division == num_divisions - 1 &&
           buffer_len % tensor_fusion_threshold != 0)
              ? (buffer_len % tensor_fusion_threshold) / sizeof(float)
              : tensor_fusion_threshold / sizeof(float);
      mpiReducer->AllreduceDivision(num_elements_division, comm, entries,
                                    global_offset);
      global_offset += (tensor_fusion_threshold / sizeof(float));
    }
  } else {
    mpiReducer->AllreduceDivision(num_elements, comm, entries, 0l);
  }
}

Status
MPI_GPUCompressedAllReduce::Execute(std::vector<TensorTableEntry>& entries,
                                    const Response& response) {
  gpu_op_context_.InitGPU(entries);
  int64_t num_elements = NumElements(entries);
  //  void* buffer_data;
  size_t buffer_len = num_elements * sizeof(float);
  auto start = clock_::now();
  Allreduce(num_elements, MPI_COMM_WORLD, entries, buffer_len);
  global_state_->allreduce_time += time_since(start);

  return Status::OK();
}

bool MPI_GPUCompressedAllReduce::Enabled(
    const horovod::common::ParameterManager& param_manager,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    const horovod::common::Response& response) const {
  if (mpiReducer == nullptr ||
      NumElements(entries) * sizeof(float) < BUFFER_THRESHOLD ||
      entries[0].tensor->dtype() != HOROVOD_FLOAT32) {
    return false;
  }
  return GPUAllreduce::Enabled(param_manager, entries, response);
}

} // namespace common
} // namespace horovod
