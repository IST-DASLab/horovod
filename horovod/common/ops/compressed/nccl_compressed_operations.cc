#include "nccl_compressed_operations.h"
#include "common.h"
#include "reducers/nccl_allgather.h"
#include "reducers/nccl_ring.h"
#include "reducers/nccl_scatter_allgather.h"

namespace horovod {
namespace common {

NCCL_CompressedAllreduce::NCCL_CompressedAllreduce(
    NCCLContext* nccl_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state,
    horovod::common::Communicator communicator_type)
    : NCCLAllreduce(nccl_context, gpu_context, global_state,
                    communicator_type) {
  Compressor* compressor = CreateGPUCompressor(gpu_context, global_state);
  auto summator = new GPUSummator(global_state, gpu_context);
  reduction_type_ = GetEnumEnvOrDefault<ReductionType>(
      HOROVOD_REDUCTION, ReductionType::NoneReduction);
  switch (reduction_type_) {
  case ReductionType::NCCL_Allgather:
    reducer = new NCCL_Allreduce_AllGather(nccl_context, gpu_context,
                                           &gpu_op_context_, global_state,
                                           compressor, summator);
    break;
  case ReductionType::NCCL_ScatterAllgather:
#if NCCL_VERSION_CHECK(2, 7, 0)
    reducer = new NCCL_Allreduce_ScatterAllgather(
        nccl_context, gpu_context, &gpu_op_context_, global_state, compressor,
        summator);
    break;
#else
    throw std::logic_error("NCCL_ScatterAllgather is not supported because of older NCCL Version.");
#endif
  case ReductionType::NCCL_Ring:
#if NCCL_VERSION_CHECK(2, 7, 0)
    reducer =
        new NCCL_Allreduce_Ring(nccl_context, gpu_context, &gpu_op_context_,
                                global_state, compressor, summator);
    break;
#else
    throw std::logic_error("NCCL_Ring is not supported because of older NCCL Version.");
#endif
    default:
    reducer = nullptr;
    break;
  }
}

NCCL_CompressedAllreduce::~NCCL_CompressedAllreduce() {
  delete reducer;
}

Status NCCL_CompressedAllreduce::Allreduce(
    int num_elements, std::vector<horovod::common::TensorTableEntry>& entries,
    int buffer_len) {
  Status status = reducer->Init(entries);
  if (!status.ok()) {
    return status;
  }
  reducer->ApplyErrorFeedback(entries);
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
      status = reducer->AllreduceDivision(num_elements_division, nccl_op_context_.nccl_comm_, entries,
                                          global_offset);
      if (!status.ok())
        return status;
      global_offset += (tensor_fusion_threshold / sizeof(float));
    }
  } else {
    status = reducer->AllreduceDivision(num_elements, nccl_op_context_.nccl_comm_, entries, 0l);
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

  auto status = Allreduce(num_elements, entries, buffer_len);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
  return gpu_op_context_.FinalizeGPUQueue(entries);
}

bool NCCL_CompressedAllreduce::EnabledName(const std::string& name) const {
  return name.find("bias") == std::string::npos;
}

bool NCCL_CompressedAllreduce::Enabled(
    const ParameterManager& param_manager,
    const std::vector<TensorTableEntry>& entries,
    const Response& response) const {
  if (reducer == nullptr ||
      NumElements(entries) * sizeof(float) < BUFFER_THRESHOLD ||
      !EnabledName(entries[0].tensor_name) ||
      !(entries[0].tensor->dtype() == HOROVOD_FLOAT32 ||
        entries[0].tensor->dtype() == HOROVOD_FLOAT16) ||
      entries[0].device == CPU_DEVICE_ID) {
    return false;
  }
  return GPUAllreduce::Enabled(param_manager, entries, response);
}

} // namespace common
} // namespace horovod
