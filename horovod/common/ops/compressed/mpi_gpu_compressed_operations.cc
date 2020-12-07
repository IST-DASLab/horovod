#include "mpi_gpu_compressed_operations.h"
#include "common.h"
#include "compression/gpu_compressor.h"
#include "mpi_compressed_operations.h"
#include "reducers/mpi_allgather.h"
#include "reducers/mpi_ps.h"
#include "reducers/mpi_ring.h"
#include "reducers/mpi_scatter_allgather.h"
#include "reducers/shm_ring.h"
#include "reducers/shm_scatter_allgather.h"
#include "reducers/shm_tree.h"
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
  auto communicator_type = GetEnumEnvOrDefault<CommunicatorType>(
      HOROVOD_COMMUNICATOR, CommunicatorType::MPI);
  if (communicator_type != CommunicatorType::MPI &&
      communicator_type != CommunicatorType::P2P &&
      communicator_type != CommunicatorType::SHM) {
    reducer_ = nullptr;
    return;
  }
  Compressor* compressor = CreateGPUCompressor(gpu_context, global_state);
  global_compressor = compressor;
  compressor_ = compressor;
  auto summator = new GPUSummator(global_state, gpu_context);
  if (communicator_type == CommunicatorType::MPI) {
    switch (reduction_type) {
    case ReductionType::AllGather:
      reducer_ = new MPI_Allreduce_AllGather(
          mpi_context, gpu_context, global_state, compressor, summator);
      break;
    case ReductionType::Ring:
      reducer_ = new MPI_Allreduce_Ring(mpi_context, gpu_context, global_state,
                                        compressor, summator);
      break;
    case ReductionType::ScatterAllgather:
      reducer_ = new MPI_Allreduce_ScatterReduceAllgather(
          mpi_context, gpu_context, global_state, compressor, summator);
      break;
    case ReductionType::PS:
      reducer_ = new MPI_Allreduce_PS(mpi_context, gpu_context, global_state,
                                      compressor, summator);
      break;
    default:
      reducer_ = nullptr;
      break;
    }
  } else {
    if (global_state->controller->GetSize() !=
        global_state->controller->GetLocalSize()) {
      throw std::logic_error("SHM_Allreduce_ScatterReduceAllgather is not "
                             "available in multi-node setting.");
    }
    switch (reduction_type) {
    case ReductionType::ScatterAllgather:
      reducer_ = new SHM_Allreduce_ScatterReduceAllgather(
          mpi_context, gpu_context, global_state, compressor, summator,
          communicator_type);
      break;
    case ReductionType::Ring:
      reducer_ =
          new SHM_Allreduce_Ring(mpi_context, gpu_context, global_state,
                                 compressor, summator, communicator_type);
      break;
    case ReductionType::Tree:
      reducer_ =
          new SHM_Allreduce_Tree(mpi_context, gpu_context, global_state,
                                 compressor, summator, communicator_type);
      break;
    default:
      reducer_ = nullptr;
      break;
    }
  }
}

MPI_GPUCompressedAllReduce::~MPI_GPUCompressedAllReduce() { delete reducer_; }

Status MPI_GPUCompressedAllReduce::Allreduce(
    int64_t num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int buffer_len) {
  Status status = reducer_->Init(entries, comm);
  if (!status.ok()) {
    return status;
  }
  reducer_->ApplyErrorFeedback(entries);
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
      status = reducer_->AllreduceDivision(num_elements_division, entries,
                                           global_offset);
      if (!status.ok())
        break;
      global_offset += (tensor_fusion_threshold / sizeof(float));
    }
  } else {
    status = reducer_->AllreduceDivision(num_elements, entries, 0l);
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
  size_t free = 0, total;
  cuMemGetInfo(&free, &total);
  size_t need_free = (reducer_ != nullptr and !reducer_->isInitialized())
                         ? reducer_->GetRequiredFreeSize()
                         : 0;
  need_free += (compressor_ != nullptr and !compressor_->isInitialized())
                   ? compressor_->GetRequiredFreeSize()
                   : 0;
  int result = 1;
  if (reducer_ == nullptr ||
      NumElements(const_cast<std::vector<TensorTableEntry>&>(entries)) *
              sizeof(float) <
          BUFFER_THRESHOLD ||
      !EnabledName(entries[0].tensor_name) ||
      !(entries[0].tensor->dtype() == HOROVOD_FLOAT32 ||
        entries[0].tensor->dtype() == HOROVOD_FLOAT16) ||
      entries[0].device == CPU_DEVICE_ID || need_free > free) {
    result = 0;
  }
  MPI_Allreduce((void*)&result, (void*)&result, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  if (result < global_state_->controller->GetSize()) {
    if (need_free > free) {
      LOG(DEBUG) << "Switch to nccl due to lack of memory";
    }
    return false;
  }
  return GPUAllreduce::Enabled(param_manager, entries, response);
}

bool MPI_GPUCompressedAllReduce::EnabledName(const std::string& name) const {
  if (reducer_ != nullptr and compressor_ != nullptr) {
    for (auto& ignore : compressor_->GetIgnoreModules()) {
      if (name.find(ignore) != std::string::npos)
        return false;
    }
  }
  return true;
}

void SetQuantizationLevels(float* levels, int bits) {
  global_compressor->SetQuantizationLevels(levels, bits);
}

} // namespace common
} // namespace horovod
