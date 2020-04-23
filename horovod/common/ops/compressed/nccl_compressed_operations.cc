#include "nccl_compressed_operations.h"
#include "common.h"
#include "reducers/nccl_allgather.h"

namespace horovod {
namespace common {

NCCL_CompressedAllreduce::NCCL_CompressedAllreduce(
    NCCLContext* nccl_context, GPUContext* gpu_context,
    HorovodGlobalState* global_state,
    horovod::common::Communicator communicator_type):
    NCCLAllreduce(nccl_context, gpu_context, global_state, communicator_type) {
  Compressor* compressor = CreateGPUCompressor(gpu_context, global_state);
  auto summator = new GPUSummator(global_state, gpu_context);
  auto reduction_type = GetEnumEnvOrDefault<ReductionType>(
      HOROVOD_REDUCTION, ReductionType::NoneReduction);
  switch (reduction_type) {
  case ReductionType::NCCL_Allgather:
    reducer = new NCCL_Allreduce_AllGather(nccl_context, gpu_context, global_state, compressor, summator);
    break;
  default:
    reducer = nullptr;
    break;
  }
}

NCCL_CompressedAllreduce::~NCCL_CompressedAllreduce() {
  delete reducer;
}

Status NCCL_CompressedAllreduce::Allreduce(
    int num_elements, ncclComm_t* comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int buffer_len) {
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
      status = reducer->AllreduceDivision(num_elements_division, comm, entries,
                                    global_offset);
      if (!status.ok())
        return status;
      global_offset += (tensor_fusion_threshold / sizeof(float));
    }
  } else {
    status = reducer->AllreduceDivision(num_elements, comm, entries, 0l);
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

  auto status = Allreduce(num_elements, nccl_op_context_.nccl_comm_, entries, buffer_len);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return status;
  }
  return gpu_op_context_.FinalizeGPUQueue(entries);
}

bool NCCL_CompressedAllreduce::Enabled(
    const ParameterManager& param_manager,
    const std::vector<TensorTableEntry>& entries,
    const Response& response) const {
  if (reducer == nullptr ||
      entries[0].tensor->dtype() != HOROVOD_FLOAT32 ||
      entries[0].device == CPU_DEVICE_ID) {
    return false;
  }
  return GPUAllreduce::Enabled(param_manager, entries, response);
}

} // namespace common
} // namespace horovod
