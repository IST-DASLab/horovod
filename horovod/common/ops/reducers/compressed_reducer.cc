#include "compressed_reducer.h"

namespace horovod {
namespace common {
MPI_CUDACompressedReducer::MPI_CUDACompressedReducer(MPIContext *mpi_context,
                                                     CUDAContext *cuda_context, HorovodGlobalState *global_state):
    MPI_CUDAAllreduce(mpi_context, cuda_context, global_state){
  const char *env_str = getenv(HOROVOD_REDUCTION);
  if (env_str == nullptr)
    reduction_type = ReductionType::Horovod;
  else {
    switch (*env_str) {
    case 'a':
      reduction_type = ReductionType::AllBroadcast;
      break;
    case 's':
      reduction_type = ReductionType::ScatterAllgather;
      break;
    case 'r':
      reduction_type = ReductionType::Ring;
      break;
    default:
      reduction_type = ReductionType::Horovod;
      break;
    }
  }
  if (global_state->quantization_bits == 32) {
    compressor = new DummyCompressor();
  } else {
    env_str = getenv(HOROVOD_COMPRESSOR);
    if (env_str == nullptr)
      compressor = new CUDAMaxMinQuantizer(cuda_context, global_state);
    else {
      switch (*env_str) {
      case 'm':
        compressor = new CUDAMaxMinQuantizer(cuda_context, global_state);
        break;
      case 'n':
        compressor = CreateCUDANormalized(cuda_context, global_state);
        break;
      default:
        compressor = new CUDAMaxMinQuantizer(cuda_context, global_state);
        break;
      }
    }
  }
  tensor_fusion_threshold =
      global_state_->param_manager.TensorFusionThresholdBytes();
}

Status MPI_CUDACompressedReducer::Allreduce(
    void* sendbuf, void* recvbuf, int num_elements, MPI_Comm comm,
    std::vector<horovod::common::TensorTableEntry>& entries, int buffer_len) {
  int world_size;
  MPI_Comm_size(comm, &world_size);

  Status status = Init(entries, world_size);
  if (!status.ok()) {
    return status;
  }
  if (sendbuf == MPI_IN_PLACE) {
    sendbuf = recvbuf;
  }
  if (buffer_len > tensor_fusion_threshold) {
    float *sendbuf_offset = (float *) sendbuf;
    float *recvbuf_offset = (float *) recvbuf;
    int num_divisions =
        (buffer_len + tensor_fusion_threshold - 1) / tensor_fusion_threshold;
    int num_elements_division = 0;
    for (int division = 0; division < num_divisions; division++) {
      num_elements_division =
          (division == num_divisions - 1 &&
           buffer_len % tensor_fusion_threshold != 0)
          ? (buffer_len % tensor_fusion_threshold) / sizeof(float)
          : tensor_fusion_threshold / sizeof(float);
      buffer_len = num_elements_division * sizeof(float);
      AllreduceDivision((void *)sendbuf_offset, (void *) recvbuf_offset,
                        num_elements_division, comm, entries, buffer_len);
      sendbuf_offset += (tensor_fusion_threshold / sizeof(float));
      recvbuf_offset += (tensor_fusion_threshold / sizeof(float));
    }
  } else {
    AllreduceDivision(sendbuf, recvbuf, num_elements, comm, entries, buffer_len);
  }
}

Status
MPI_CUDACompressedReducer::Execute(std::vector<TensorTableEntry>& entries,
                                         const Response& response) {
  auto& first_entry = entries[0];
//  if (!status.ok())
//    return status;
  InitCUDA(entries);
  int64_t num_elements = NumElements(entries);
//  if (global_state_->rank == 0) {
//    std::cout << "Started processing " << num_elements << std::endl;
//    for (auto& entry: entries) {
//      std::cout << entry.tensor_name << " ";
//    }
//    std::cout << std::endl;
//  }

  HERE
  void* buffer_data;
  size_t buffer_len;
  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    HERE
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    auto cuda_result =
        cudaStreamSynchronize(cuda_context_->streams[first_entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*)first_entry.tensor->data();
    buffer_len = (size_t)first_entry.tensor->size();
  }
  HERE
  Allreduce(MPI_IN_PLACE, buffer_data, num_elements, MPI_COMM_WORLD, entries, buffer_len);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    HERE
    MemcpyOutFusionBuffer(buffer_data, entries);

    auto cuda_result =
        cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  } else {
    if (first_entry.tensor->data() != first_entry.output->data()) {
      cudaMemcpy((void*)first_entry.output->data(), (const void*)buffer_data,
                 first_entry.output->size(), cudaMemcpyDeviceToDevice);
    }
  }
//  if (global_state_->rank == 0) {
//    std::cout << "Finished processing " << std::endl;
//    for (auto& entry: entries) {
//      std::cout << entry.tensor_name << " ";
//    }
//    std::cout << std::endl;
//  }

  HERE
  return Status::OK();
}


} // common
} // horovod
