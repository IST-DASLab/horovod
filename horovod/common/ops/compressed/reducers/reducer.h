#ifndef HOROVOD_REDUCER_H
#define HOROVOD_REDUCER_H

#include "../../../hashes.h"
#include "../../nccl_operations.h"
#include "../compression/compressor.h"
#include "../compression/error_feedback.h"
#include "../compression/vector_operations.h"
#include "common.h"
#include "comm.h"

namespace horovod {
namespace common {

class Reducer {
public:
  Reducer(HorovodGlobalState* global_state, Compressor* compressor,
          Summator* summator)
      : global_state_(global_state), compressor_(compressor),
        error_feedback_(summator) {
    tensor_fusion_threshold_ =
        global_state->parameter_manager.TensorFusionThresholdBytes();
  }

  virtual ~Reducer() {
    delete compressor_;
  }

  void ApplyErrorFeedback(std::vector<TensorTableEntry>& entries) {
    error_feedback_.Apply(entries);
  }

protected:
  HorovodGlobalState* global_state_;

  Compressor* compressor_;
  ErrorFeedback error_feedback_;

  // We only need some framework agnostic Buffer Manager so we reuse
  // FussionBufferManager. Our usage of it is not related to tensor fusion
  // buffer.
  FusionBufferManager bufferManager_;
  unsigned char* gradients_send_ = nullptr;
  unsigned char* gradients_recv_ = nullptr;
  unsigned char* decompress_buffer_ = nullptr;
  int64_t tensor_fusion_threshold_;
};

class MPIReducer : public Reducer {
public:
  MPIReducer(MPIContext* mpi_context, GPUContext* gpu_context,
             HorovodGlobalState* global_state, Compressor* compressor,
             Summator* summator)
      : mpi_context_(mpi_context), gpu_context_(gpu_context),
        Reducer(global_state, compressor, summator) {}

  virtual Status Init(const std::vector<TensorTableEntry>& entries,
                      MPI_Comm comm) = 0;

  virtual Status AllreduceDivision(int num_elements,
                                   std::vector<TensorTableEntry>& entries,
                                   int64_t global_offset) = 0;

protected:
  MPIContext* mpi_context_;
  GPUContext* gpu_context_;
  MPI_Comm comm_;
};

#define NCCL_CALL_CHECK(name, op, comm)                                        \
  nccl_context_->ErrorCheck(name, op, comm)

class NCCLReducer : public Reducer {
public:
  NCCLReducer(NCCLContext* nccl_context, GPUContext* gpu_context,
              GPUOpContext* gpu_op_context, HorovodGlobalState* global_state,
              Compressor* compressor, Summator* summator)
      : Reducer(global_state, compressor, summator),
        nccl_context_(nccl_context), gpu_context_(gpu_context),
        gpu_op_context_(gpu_op_context) {}

  virtual Status Init(const std::vector<TensorTableEntry>& entries) = 0;

  virtual Status AllreduceDivision(int num_elements, ncclComm_t* nccl_comm,
                                   std::vector<TensorTableEntry>& entries,
                                   int64_t global_offset) = 0;

protected:
  NCCLContext* nccl_context_;
  GPUContext* gpu_context_;
  GPUOpContext* gpu_op_context_;
  gpuStream_t* stream_;
};

class SHMReducer : public MPIReducer {
public:
  SHMReducer(MPIContext* mpi_context, GPUContext* gpu_context,
             HorovodGlobalState* global_state, Compressor* compressor,
             Summator* summator, CommunicatorType comm_type)
      : MPIReducer(mpi_context, gpu_context, global_state, compressor,
                   summator), comm_type_(comm_type) {};

  virtual Status AllreduceDivision(int num_elements,
                                   std::vector<TensorTableEntry>& entries,
                                   int64_t global_offset) = 0;
  virtual ~SHMReducer() {
    hcomm_.reset();
  }

protected:
  std::shared_ptr<Comm> hcomm_;
  CommunicatorType comm_type_;
};
void printDebug(float* bf, int num_elems, int device, std::string prefix);

} // namespace common
} // namespace horovod
#endif // HOROVOD_REDUCER_H
