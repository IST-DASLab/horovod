#include "nccl_compressed_operations.h"
#include "common.h"
#include "reducers/nccl_allgather.h"
#include "reducers/nccl_ring.h"
#include "reducers/nccl_scatter_allgather.h"

#include <string>

namespace horovod {
namespace common {

NCCL_CompressedAllreduce::NCCL_CompressedAllreduce(
    NCCLContext* nccl_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state,
    horovod::common::Communicator mpi_communicator)
    : NCCLAllreduce(nccl_context, gpu_context, global_state, mpi_communicator) {
  auto summator = new GPUSummator(global_state, gpu_context);
  compressor_ = CreateGPUCompressor(gpu_context, global_state, summator);
  auto reduction_type = GetEnumEnvOrDefault<ReductionType>(
      HOROVOD_REDUCTION, ReductionType::NoneReduction);
  auto communicator_type = GetEnumEnvOrDefault<CommunicatorType>(
      HOROVOD_COMMUNICATOR, CommunicatorType::MPI);
  if (communicator_type != CommunicatorType::NCCL) {
    reducer_ = nullptr;
    return;
  }

  switch (reduction_type) {
  case ReductionType::AllGather:
    reducer_ = new NCCL_Allreduce_AllGather(
        nccl_context, gpu_context, &gpu_op_context_, global_state, compressor_);
    break;
  case ReductionType::ScatterAllgather:
#ifdef NCCL_P2P_SUPPORTED
    reducer_ = new NCCL_Allreduce_ScatterAllgather(
        nccl_context, gpu_context, &gpu_op_context_, global_state, compressor_);
    break;
#else
    throw std::logic_error("NCCL_ScatterAllgather is not supported because of "
                           "older NCCL Version.");
#endif
  case ReductionType::Ring:
#ifdef NCCL_P2P_SUPPORTED
    reducer_ = new NCCL_Allreduce_Ring(
        nccl_context, gpu_context, &gpu_op_context_, global_state, compressor_);
    break;
#else
    throw std::logic_error(
        "NCCL_Ring is not supported because of older NCCL Version.: " +
        std::to_string(NCCL_VERSION_CODE));
#endif
  default:
    reducer_ = nullptr;
    break;
  }
}

NCCL_CompressedAllreduce::~NCCL_CompressedAllreduce() { delete reducer_; }

Status NCCL_CompressedAllreduce::Allreduce(
    int num_elements, std::vector<horovod::common::TensorTableEntry>& entries,
    int buffer_len) {
  Status status = reducer_->Init(entries);
  if (!status.ok()) {
    return status;
  }
  compressor_->ApplyErrorFeedback(entries);
  const void* fused_input_data;
  void* buffer_ptr = nullptr;
  if (compressor_->GetCompressionMode() != CompressionMode::NonFused) {
    if (entries.size() == 1) {
      buffer_ptr = (void*)entries[0].output->data();
    } else {
      size_t dummy;
      MemcpyInFusionBuffer(entries, fused_input_data, buffer_ptr, dummy);
    }
  }
  status =
      reducer_->AllreduceDivision(num_elements, nccl_op_context_.nccl_comm_,
                                  entries, (unsigned char*)buffer_ptr);
  if (!status.ok()) {
    return status;
  }
  if (compressor_->GetCompressionMode() != CompressionMode::NonFused and
      entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_ptr, entries);
  }
  return status;
}

Status NCCL_CompressedAllreduce::Execute(
    std::vector<horovod::common::TensorTableEntry>& entries,
    const horovod::common::Response& response) {
  int64_t num_elements = NumElements(entries);
  size_t buffer_len = num_elements * sizeof(float);

  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, response.devices());
  gpu_op_context_.InitGPUQueue(entries, response);
  if (response.prescale_factor() != 1.0) {
    ScaleEntriesInPlace(response.prescale_factor(), entries, false);
  }
  auto status = Allreduce(num_elements, entries, buffer_len);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
  if (response.postscale_factor() != 1.0) {
    ScaleEntriesInPlace(response.postscale_factor(), entries, true);
  }
  return gpu_op_context_.FinalizeGPUQueue(entries);
}

bool NCCL_CompressedAllreduce::GlobalEnabled(
    const ParameterManager& param_manager) const {
  return reducer_ != nullptr and GPUAllreduce::GlobalEnabled(param_manager);
}

bool NCCL_CompressedAllreduce::EnabledName(const std::string& name) const {
  if (GlobalEnabled(global_state_->parameter_manager)) {
    for (auto& ignore : compressor_->GetIgnoreModules()) {
      if (name.find(ignore) != std::string::npos)
        return false;
    }
  }
  return true;
}

bool NCCL_CompressedAllreduce::Enabled(
    const ParameterManager& param_manager,
    const std::vector<TensorTableEntry>& entries,
    const Response& response) const {
  //  size_t free = 0, total;
  //  cuMemGetInfo(&free, &total);
  //  size_t need_free = (reducer_ != nullptr and !reducer_->isInitialized())
  //                         ? reducer_->GetRequiredFreeSize()
  //                         : 0;
  //  need_free += (compressor_ != nullptr and !compressor_->isInitialized())
  //                   ? compressor_->GetRequiredFreeSize()
  //                   : 0;
  //  int result = 1;
  if (reducer_ == nullptr ||
      NumElements(const_cast<std::vector<TensorTableEntry>&>(entries)) *
              sizeof(float) <
          BUFFER_THRESHOLD ||
      !EnabledName(entries[0].tensor_name) ||
      !(entries[0].tensor->dtype() == HOROVOD_FLOAT32 ||
        entries[0].tensor->dtype() == HOROVOD_FLOAT16) ||
      entries[0].device == CPU_DEVICE_ID) {
    return false;
  }
  //  if (need_free > free) {
  //    result = 0;
  //  }
  //  MPI_Allreduce((void*)&result, (void*)&result, 1, MPI_INT, MPI_SUM,
  //                MPI_COMM_WORLD);
  //  if (result < global_state_->controller->GetSize()) {
  //    if (need_free > free) {
  //      LOG(DEBUG) << "Switch to nccl due to lack of memory";
  //    }
  //    return false;
  //  }
  return GPUAllreduce::Enabled(param_manager, entries, response);
}

} // namespace common
} // namespace horovod
