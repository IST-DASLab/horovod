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
      HOROVOD_COMMUNICATOR, CommunicatorType::MPI);
  if (communicator_type != CommunicatorType::MPI) {
    reducer_ = nullptr;
    return;
  }
  auto norm_type = GetEnumEnvOrDefault<NormType>(HOROVOD_COMPRESSION_NORM_TYPE,
                                                 NormType::Linf);
  auto levels_type = GetEnumEnvOrDefault<LevelsType>(
      HOROVOD_COMPRESSION_LEVELS_TYPE, LevelsType::Pos);
  auto quantization_bits = GetIntEnvOrDefault(HOROVOD_QUANTIZATION_BITS, 32);
  auto summator = new CPUSummator();
  if (quantization_bits == 32 ||
      compression_type == CompressionType::NoneCompression) {
    compressor_ = new CPUDummyCompressor(global_state, summator);
  } else {
    switch (compression_type) {
    case CompressionType::MaxMin:
      compressor_ =
          new CPUMaxMinQuantizer(global_state, summator, quantization_bits);
      break;
    case CompressionType::Exp:
    case CompressionType::Uni:
      compressor_ =
          new CPUNormalizedQuantizer(global_state, summator, quantization_bits,
                                     compression_type, norm_type, levels_type);
      break;
    case CompressionType::TopK:
      LOG(INFO) << "TopK is not supported for CPU-based tensors";
      return;
    default:
      throw std::logic_error("Invalid compression type.");
    }
  }
  switch (reduction_type) {
  case ReductionType::AllGather:
    reducer_ = new MPI_Allreduce_AllGather(mpi_context, gpu_context,
                                           global_state, compressor_);
    break;
  case ReductionType::Ring:
    reducer_ = new MPI_Allreduce_Ring(mpi_context, gpu_context, global_state,
                                      compressor_);
    break;
  case ReductionType::ScatterAllgather:
    reducer_ = new MPI_Allreduce_ScatterReduceAllgather(
        mpi_context, gpu_context, global_state, compressor_);
    break;
  case ReductionType::PS:
    reducer_ = new MPI_Allreduce_PS(mpi_context, gpu_context, global_state,
                                    compressor_);
    break;
  default:
    reducer_ = nullptr;
    break;
  }
}

MPI_CompressedAllReduce::~MPI_CompressedAllReduce() { delete reducer_; }

Status MPI_CompressedAllReduce::Allreduce(
    int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int buffer_len) {
  Status status = reducer_->Init(entries, comm);
  if (!status.ok()) {
    return status;
  }
  compressor_->ApplyErrorFeedback(entries);
  const void* fused_input_data;
  void* buffer_ptr = nullptr;
  if (compressor_->GetCompressionMode() != CompressionMode::NonFused) {
    if (entries.size() == 1) {
      buffer_ptr = (void*) entries[0].output->data();
    } else {
      size_t dummy;
      MemcpyInFusionBuffer(entries, fused_input_data, buffer_ptr,
                           dummy);
    }
  }
  status = reducer_->AllreduceDivision(num_elements, entries, (unsigned char*) buffer_ptr);
  if (!status.ok()) {
    return status;
  }
  if (compressor_->GetCompressionMode() != CompressionMode::NonFused and
      entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_ptr, entries);
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

bool MPI_CompressedAllReduce::GlobalEnabled(
    const ParameterManager& param_manager) const {
  return reducer_ != nullptr and MPIAllreduce::GlobalEnabled(param_manager);
}

bool MPI_CompressedAllReduce::EnabledName(const std::string& name) const {
  if (GlobalEnabled(global_state_->parameter_manager)) {
    for (auto& ignore : compressor_->GetIgnoreModules()) {
      if (name.find(ignore) != std::string::npos)
        return false;
    }
  }
  return true;
}

bool MPI_CompressedAllReduce::Enabled(
    const horovod::common::ParameterManager& param_manager,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    const horovod::common::Response& response) const {
  return false;
//  if (reducer_ == nullptr ||
//      NumElements(const_cast<std::vector<TensorTableEntry>&>(entries)) <
//          BUFFER_THRESHOLD ||
//      entries[0].tensor->dtype() != HOROVOD_FLOAT32 ||
//      entries[0].device != CPU_DEVICE_ID) {
//    return false;
//  }
//  return MPIAllreduce::Enabled(param_manager, entries, response);
}

} // namespace common
} // namespace horovod
