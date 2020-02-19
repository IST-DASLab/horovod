#ifndef HOROVOD_REDUCER_H
#define HOROVOD_REDUCER_H

#include "../../../mpi/mpi_context.h"
#include "../../gpu_operations.h"
#include "../compression/compressor.h"
#include "../compression/vector_operations.h"
#include "../compression/error_feedback.h"

namespace horovod {
namespace common {

class MPIReducer {
public:
  MPIReducer(MPIContext* mpi_context, GPUContext* gpu_context,
             HorovodGlobalState* global_state, Compressor* compressor,
             Summator* summator)
      : gpu_context_(gpu_context),
        global_state_(global_state), mpi_context_(mpi_context),
        compressor_(compressor), summator_(summator),
        errorFeedbackManager_(global_state, gpu_context) {
    tensor_fusion_threshold_ =
        global_state->parameter_manager.TensorFusionThresholdBytes();
  }

  ~MPIReducer() = default;

  virtual Status Init(const std::vector<TensorTableEntry>& entries) = 0;

  virtual Status AllreduceDivision(int num_elements, MPI_Comm comm,
                                   std::vector<TensorTableEntry>& entries,
                                   int64_t global_offset) = 0;

protected:
  GPUContext* gpu_context_;

  HorovodGlobalState* global_state_;
  MPIContext* mpi_context_;

  Compressor* compressor_;
  Summator* summator_;
  ErrorFeedback errorFeedbackManager_;

  // We only need some framework agnostic Buffer Manager so we reuse
  // FussionBufferManager. Our usage of it is not related to tensor fusion
  // buffer.
  FusionBufferManager bufferManager_;
  unsigned char* gradients_send_ = nullptr;
  unsigned char* gradients_recv_ = nullptr;
  unsigned char* decompress_buffer_ = nullptr;
  int64_t tensor_fusion_threshold_;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_REDUCER_H
