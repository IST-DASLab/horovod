#ifndef NCCL_RING_H
#define NCCL_RING_H
#include "reducer.h"

#ifdef NCCL_P2P_SUPPORTED

namespace horovod {
namespace common {

class NCCL_Allreduce_Ring : public NCCLReducer {
public:
  NCCL_Allreduce_Ring(NCCLContext* nccl_context, GPUContext* gpu_context,
                      GPUOpContext* gpu_op_context,
                      HorovodGlobalState* global_state, Compressor* compressor);
  Status
  Init(const std::vector<horovod::common::TensorTableEntry>& entries) final;
  Status
  AllreduceDivision(int num_elements, ncclComm_t* nccl_comm_,
                    std::vector<horovod::common::TensorTableEntry>& entries,
                    int64_t global_offset) override;
  size_t GetRequiredFreeSize() override;
};

} // namespace common
} // namespace horovod

#endif
#endif // NCCL_RING_H
