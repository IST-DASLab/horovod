#include "mpi_gpu_compressed_operations.h"
#include "common.h"
#include "compression/gpu_compressor.h"
#include "mpi_compressed_operations.h"
#include "reducers/mpi_allgather.h"
#include "reducers/mpi_ps.h"
#include "reducers/mpi_ring.h"
#include "reducers/mpi_scatter_allgather.h"
#include "reducers/shm_scatter_allgather.h"
#include "utils.h"

#include <chrono>
#include <thread>

namespace horovod {
namespace common {

Compressor* global_compressor;

MPI_GPUCompressedAllReduce::MPI_GPUCompressedAllReduce(
    MPIContext* mpi_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state)
    : MPI_GPUAllreduce(mpi_context, gpu_context, global_state) {
  auto reduction_type = GetEnumEnvOrDefault<ReductionType>(
      HOROVOD_REDUCTION, ReductionType::NoneReduction);
  Compressor* compressor = CreateGPUCompressor(gpu_context, global_state);
  global_compressor = compressor;
  auto summator = new GPUSummator(global_state, gpu_context);
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
  case ReductionType::SHM_ScatterAllgather:
    if (global_state->controller->GetSize() !=
           global_state->controller->GetLocalSize()) {
      throw std::logic_error("SHM_Allreduce_ScatterReduceAllgather is not available in multi-node setting.");
    }
    mpiReducer = new SHM_Allreduce_ScatterReduceAllgather(
        mpi_context, gpu_context, global_state, compressor, summator);
    break;
  default:
    mpiReducer = nullptr;
    break;
  }
}

MPI_GPUCompressedAllReduce::~MPI_GPUCompressedAllReduce() { delete mpiReducer; }

Status MPI_GPUCompressedAllReduce::Allreduce(
    int64_t num_elements, MPI_Comm comm,
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

Status
MPI_GPUCompressedAllReduce::Execute(std::vector<TensorTableEntry>& entries,
                                    const Response& response) {
  gpu_op_context_.InitGPU(entries);
  int64_t num_elements = NumElements(entries);
  int buffer_len = num_elements * sizeof(float);
  if (response.prescale_factor() != 1.0) {
    ScaleEntriesInPlace(response.prescale_factor(), entries, false);
  }
  auto status = Allreduce(num_elements, MPI_COMM_WORLD, entries, buffer_len);
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

bool MPI_GPUCompressedAllReduce::Enabled(
    const horovod::common::ParameterManager& param_manager,
    const std::vector<TensorTableEntry>& entries,
    const horovod::common::Response& response) const {
  if (mpiReducer == nullptr ||
      NumElements(const_cast<std::vector<TensorTableEntry>&>(entries)) *
              sizeof(float) <
          BUFFER_THRESHOLD ||
      !EnabledName(entries[0].tensor_name) ||
      !(entries[0].tensor->dtype() == HOROVOD_FLOAT32 ||
        entries[0].tensor->dtype() == HOROVOD_FLOAT16) ||
      entries[0].device == CPU_DEVICE_ID) {
    return false;
  }
  return GPUAllreduce::Enabled(param_manager, entries, response);
}

bool MPI_GPUCompressedAllReduce::EnabledName(const std::string& name) const {
  return name.find("bias") == std::string::npos;
}

void SetQuantizationLevels(float* levels) {
  global_compressor->SetQuantizationLevels(levels);
}

} // namespace common
} // namespace horovod
