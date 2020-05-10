#include "mpi_gpu_compressed_operations.h"
#include "mpi_compressed_operations.h"
#include "reducers/all_broadcast.h"
#include "reducers/ring.h"
#include "reducers/scatter_allgather.h"
#include "compression/gpu_compressor.h"
#include "utils.h"
#include "common.h"

namespace horovod {
namespace common {

Compressor *global_compressor;

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
  case ReductionType::AllBroadcast:
    mpiReducer = new MPI_Allreduce_AllBroadcast(mpi_context,
                                                   global_state, compressor, summator);
    break;
  case ReductionType::Ring:
    mpiReducer = new MPI_Allreduce_Ring(mpi_context,
                                           global_state, compressor, summator);
    break;
  case ReductionType::ScatterAllgather:
    mpiReducer = new MPI_Allreduce_ScatterReduceAllgather(
        mpi_context, global_state, compressor, summator);
    break;
  default:
    mpiReducer = nullptr;
    break;
  }
}

MPI_GPUCompressedAllReduce::~MPI_GPUCompressedAllReduce() {
  delete mpiReducer;
}

Status MPI_GPUCompressedAllReduce::Allreduce(
    int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int buffer_len) {
  Status status = mpiReducer->Init(entries);
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
      status = mpiReducer->AllreduceDivision(num_elements_division, comm, entries,
                                    global_offset);
      if (!status.ok())
        break;
      global_offset += (tensor_fusion_threshold / sizeof(float));
    }
  } else {
    status = mpiReducer->AllreduceDivision(num_elements, comm, entries, 0l);
  }
  return status;
}

Status
MPI_GPUCompressedAllReduce::Execute(std::vector<TensorTableEntry>& entries,
                                    const Response& response) {
  gpu_op_context_.InitGPU(entries);
  int64_t num_elements = NumElements(entries);
  //  void* buffer_data;
  size_t buffer_len = num_elements * sizeof(float);
  auto start = clock_::now();
  auto status = Allreduce(num_elements, MPI_COMM_WORLD, entries, buffer_len);
  global_state_->allreduce_time += time_since(start);
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
  }
  return status;
}

bool MPI_GPUCompressedAllReduce::Enabled(
    const horovod::common::ParameterManager& param_manager,
    const std::vector<horovod::common::TensorTableEntry>& entries,
    const horovod::common::Response& response) const {
  if (mpiReducer == nullptr ||
      NumElements(entries) * sizeof(float) < BUFFER_THRESHOLD ||
      entries[0].tensor->dtype() != HOROVOD_FLOAT32 ||
      entries[0].device == CPU_DEVICE_ID) {
    return false;
  }
  return GPUAllreduce::Enabled(param_manager, entries, response);
}

void SetQuantizationLevels(float* levels) {
  global_compressor->SetQuantizationLevels(levels);
}

} // namespace common
} // namespace horovod
