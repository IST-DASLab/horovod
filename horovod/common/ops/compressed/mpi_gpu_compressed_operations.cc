#include "mpi_gpu_compressed_operations.h"
#include "utils.h"
#include "reducers/ring.h"
#include "reducers/scatter_allgather.h"
#include "reducers/all_broadcast.h"

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

  switch (reduction_type) {
  case ReductionType::AllBroadcast:
    mpiReducer = new MPI_GPUAllreduce_AllBroadcast(mpi_context, gpu_context,
                                                global_state, compressor);
    break;
  case ReductionType::Ring:
    mpiReducer = new MPI_GPUAllreduce_Ring(mpi_context, gpu_context, global_state,
                                        compressor);
    break;
  case ReductionType::ScatterAllgather:
    mpiReducer = new MPI_GPUAllreduce_ScatterReduceAllgather(
        mpi_context, gpu_context, global_state, compressor);
    break;
  case ReductionType::NoneReduction:
    mpiReducer = nullptr;
    break;
  }
}

Status MPI_GPUCompressedAllReduce::Allreduce(
    void* sendbuf, void* recvbuf, int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int buffer_len) {
  Status status = mpiReducer->Init(entries);
  if (!status.ok()) {
    return status;
  }
  if (sendbuf == MPI_IN_PLACE) {
    sendbuf = recvbuf;
  }
  int64_t tensor_fusion_threshold =
      global_state_->parameter_manager.TensorFusionThresholdBytes();
  if (buffer_len > tensor_fusion_threshold) {
    float* sendbuf_offset = (float*)sendbuf;
    float* recvbuf_offset = (float*)recvbuf;
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
      mpiReducer->AllreduceDivision((void*)(sendbuf_offset + global_offset),
                                 (void*)(recvbuf_offset + global_offset),
                                 num_elements_division, comm, entries,
                                 global_offset);
      global_offset += (tensor_fusion_threshold / sizeof(float));
    }
  } else {
    mpiReducer->AllreduceDivision(sendbuf, recvbuf, num_elements, comm, entries,
                               0l);
  }
}

Status
MPI_GPUCompressedAllReduce::Execute(std::vector<TensorTableEntry>& entries,
                                    const Response& response) {
  auto& first_entry = entries[0];
  gpu_op_context_.InitGPU(entries);
  int64_t num_elements = NumElements(entries);
  void* buffer_data;
  size_t buffer_len;
  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    gpu_context_->StreamSynchronize(
        gpu_context_
            ->streams[global_state_->current_nccl_stream][first_entry.device]);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*)first_entry.tensor->data();
    buffer_len = (size_t)first_entry.tensor->size();
  }

  auto start = clock_::now();
  Allreduce(MPI_IN_PLACE, buffer_data, num_elements, MPI_COMM_WORLD, entries,
            buffer_len);
  global_state_->allreduce_time += time_since(start);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);

    gpu_context_->StreamSynchronize(
        gpu_context_
            ->streams[global_state_->current_nccl_stream][entries[0].device]);

    timeline.ActivityEndAll(entries);
  }
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
