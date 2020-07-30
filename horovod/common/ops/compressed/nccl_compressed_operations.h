#ifndef HOROVOD_NCCL_COMPRESSED_OPERATIONS_H
#define HOROVOD_NCCL_COMPRESSED_OPERATIONS_H

#include "../nccl_operations.h"
#include "reducers/reducer.h"
#include "common.h"


namespace horovod {
namespace common {

class NCCL_CompressedAllreduce : public NCCLAllreduce {
public:
  NCCL_CompressedAllreduce(
      NCCLContext* nccl_context, GPUContext* gpu_context,
      HorovodGlobalState* global_state,
      horovod::common::Communicator communicator_type = Communicator::GLOBAL);

  ~NCCL_CompressedAllreduce() override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;
  Status Allreduce(int num_elements,
                   std::vector<TensorTableEntry>& entries,
                   int buffer_len);
  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
protected:
  void InitNCCLCommunicator_All(const std::vector<TensorTableEntry>& entries);
  void InitNCCLCommunicator_Ring(const std::vector<TensorTableEntry>& entries);
  NCCLReducer *reducer;
  ReductionType reduction_type_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_NCCL_COMPRESSED_OPERATIONS_H
