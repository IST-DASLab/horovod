#include "mpi_compressed_operations.h"
#include "common.h"
#include "reducers/mpi_allgather.h"
#include "reducers/mpi_ps.h"
#include "reducers/mpi_ring.h"
#include "reducers/mpi_scatter_allgather.h"
#include "utils.h"

namespace horovod {
namespace common {

MPI_CompressedAllReduce::MPI_CompressedAllReduce(
    MPIContext* mpi_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state)
    : MPIAllreduce(mpi_context, global_state) {
  auto reduction_type = GetEnumEnvOrDefault<ReductionType>(
      HOROVOD_REDUCTION, ReductionType::NoneReduction);
  auto compression_type = GetEnumEnvOrDefault<CompressionType>(
      HOROVOD_COMPRESSION, CompressionType::NoneCompression);
  auto communicator_type = GetEnumEnvOrDefault<CommunicatorType>(
      HOROVOD_COMMUNICATOR, CommunicatorType::NoneCommunicator);
  if (communicator_type != CommunicatorType::MPI) {
    mpiReducer = nullptr;
    return;
  }
  auto norm_type = GetEnumEnvOrDefault<NormType>(HOROVOD_COMPRESSION_NORM_TYPE,
                                                 NormType::Linf);
  auto levels_type = GetEnumEnvOrDefault<LevelsType>(
      HOROVOD_COMPRESSION_LEVELS_TYPE, LevelsType::Pos);
  auto quantization_bits = GetIntEnvOrDefault(HOROVOD_QUANTIZATION_BITS, 32);
  Compressor* compressor;
  if (quantization_bits == 32 ||
      compression_type == CompressionType::NoneCompression) {
    compressor = new CPUDummyCompressor(global_state);
  } else {
    switch (compression_type) {
    case CompressionType::MaxMin:
      compressor = new CPUMaxMinQuantizer(global_state, quantization_bits);
      break;
    case CompressionType::Exp:
    case CompressionType::Uni:
      compressor =
          new CPUNormalizedQuantizer(global_state, quantization_bits,
                                     compression_type, norm_type, levels_type);
      break;
    default:
      throw std::logic_error("Invalid compression type.");
    }
  }
  auto summator = new CPUSummator();
  switch (reduction_type) {
  case ReductionType::AllGather:
    mpiReducer = new MPI_Allreduce_AllGather(
        mpi_context, gpu_context, global_state, compressor, summator);
    break;
  case ReductionType::Ring:
    mpiReducer = new MPI_Allreduce_Ring(mpi_context, gpu_context, global_state,
                                        compressor, summator);
    break;
  case ReductionType::ScatterAllgather:
    mpiReducer = new MPI_Allreduce_ScatterReduceAllgather(
        mpi_context, gpu_context, global_state, compressor, summator);
    break;
  case ReductionType::PS:
    mpiReducer = new MPI_Allreduce_PS(mpi_context, gpu_context, global_state,
                                      compressor, summator);
    break;
  default:
    mpiReducer = nullptr;
    break;
  }
}

MPI_CompressedAllReduce::~MPI_CompressedAllReduce() { delete mpiReducer; }

Status MPI_CompressedAllReduce::Allreduce(
    int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int buffer_len) {
  Status status = mpiReducer->Init(entries, comm);
  if (!status.ok()) {
    return status;
  }
  mpiReducer->ApplyErrorFeedback(entries);
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
      status = mpiReducer->AllreduceDivision(num_elements_division, entries,
                                             global_offset);
      if (!status.ok())
        break;
      global_offset += (tensor_fusion_threshold / sizeof(float));
    }
  } else {
    status = mpiReducer->AllreduceDivision(num_elements, entries, 0l);
  }
  return status;
}

Status MPI_CompressedAllReduce::Execute(std::vector<TensorTableEntry>& entries,
                                        const Response& response) {
  int64_t num_elements = NumElements(entries);
  size_t buffer_len = num_elements * sizeof(float);
  if (response.prescale_factor() != 1.0) {
    ScaleEntriesInPlace(response.prescale_factor(), entries, false);
  }
  auto status = Allreduce(
      num_elements, mpi_context_->GetMPICommunicator(Communicator::GLOBAL),
      entries, buffer_len);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
  }
  if (response.postscale_factor() != 1.0) {
    ScaleEntriesInPlace(response.postscale_factor(), entries, true);
  }
  return status;
}

bool MPI_CompressedAllReduce::Enabled(
    const horovod::common::ParameterManager& param_manager,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    const horovod::common::Response& response) const {
  if (mpiReducer == nullptr ||
      NumElements(const_cast<std::vector<TensorTableEntry>&>(entries)) *
              sizeof(float) <
          BUFFER_THRESHOLD ||
      entries[0].tensor->dtype() != HOROVOD_FLOAT32 ||
      entries[0].device != CPU_DEVICE_ID) {
    return false;
  }
  return MPIAllreduce::Enabled(param_manager, entries, response);
}

} // namespace common
} // namespace horovod
