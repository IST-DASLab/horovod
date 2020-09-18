#ifndef HOROVOD_REDUCER_H
#define HOROVOD_REDUCER_H

#include "../../../mpi/mpi_context.h"
#include "../../nccl_operations.h"
#include "../compression/compressor.h"
#include "../compression/error_feedback.h"
#include "../compression/vector_operations.h"

namespace horovod {
namespace common {

#define MPI_CHECK(condition)                                                   \
  do {                                                                         \
    int op = condition;                                                        \
    if (op != MPI_SUCCESS) {                                                   \
      throw std::runtime_error(std::string(#condition) + " on line " +         \
                               std::to_string(__LINE__) + " failed: ");        \
    }                                                                          \
  } while (0)

class Reducer {
public:
  Reducer(HorovodGlobalState* global_state, Compressor* compressor,
          Summator* summator)
      : global_state_(global_state), compressor_(compressor),
        summator_(summator), error_feedback_(summator, global_state->controller->GetRank() == 0) {
    tensor_fusion_threshold_ =
        global_state->parameter_manager.TensorFusionThresholdBytes();
  }

  virtual ~Reducer() {
    delete compressor_;
    delete summator_;
  }

  virtual Status Init(const std::vector<TensorTableEntry>& entries) = 0;

  void ApplyErrorFeedback(std::vector<TensorTableEntry>& entries) {
    error_feedback_.Apply(entries);
  }

protected:
  HorovodGlobalState* global_state_;

  Compressor* compressor_;
  Summator* summator_;
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
  MPIReducer(MPIContext* mpi_context, HorovodGlobalState* global_state,
             Compressor* compressor, Summator* summator)
      : mpi_context_(mpi_context), Reducer(global_state, compressor, summator) {
  }

  virtual Status Init(const std::vector<TensorTableEntry>& entries) = 0;

  virtual Status AllreduceDivision(int num_elements, MPI_Comm comm,
                                   std::vector<TensorTableEntry>& entries,
                                   int64_t global_offset) = 0;

protected:
  MPIContext* mpi_context_;
};

#ifdef __JETBRAINS_IDE__
#define NCCL_VERSION(X,Y,Z) ((X) * 1000 + (Y) * 100 + (Z))
#endif

#define NCCL_VERSION_CHECK(major, minor, patch) \
(NCCL_VERSION_CODE >= NCCL_VERSION(major, minor, patch))

#define NCCL_CALL_CHECK(name, op) \
nccl_context_->ErrorCheck(name, op)

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

void printDebug(float* bf, int num_elems, int device, std::string prefix);

} // namespace common
} // namespace horovod
#endif // HOROVOD_REDUCER_H
