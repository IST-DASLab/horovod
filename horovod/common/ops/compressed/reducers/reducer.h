#ifndef HOROVOD_REDUCER_H
#define HOROVOD_REDUCER_H

#include "../../../mpi/mpi_context.h"
#include "../compression/compressor.h"
#include "../compression/vector_operations.h"
#include "../compression/error_feedback.h"

namespace horovod {
namespace common {

class MPIReducer {
public:
  MPIReducer(MPIContext* mpi_context,
             HorovodGlobalState* global_state, Compressor* compressor,
             Summator* summator)
      : global_state_(global_state), mpi_context_(mpi_context),
        compressor_(compressor), summator_(summator), error_feedback_(summator) {
    tensor_fusion_threshold_ =
        global_state->parameter_manager.TensorFusionThresholdBytes();
  }

  virtual ~MPIReducer() {
    delete compressor_;
    delete summator_;
  }

  virtual Status Init(const std::vector<TensorTableEntry>& entries) = 0;

  virtual Status AllreduceDivision(int num_elements, MPI_Comm comm,
                                   std::vector<TensorTableEntry>& entries,
                                   int64_t global_offset) = 0;
  void ApplyErrorFeedback(std::vector<TensorTableEntry>& entries) {
    error_feedback_.Apply(entries);
  }

protected:
  HorovodGlobalState* global_state_;
  MPIContext* mpi_context_;

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

} // namespace common
} // namespace horovod
#endif // HOROVOD_REDUCER_H
