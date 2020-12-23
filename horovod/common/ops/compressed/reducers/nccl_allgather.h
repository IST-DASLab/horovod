#ifndef HOROVOD_COMPRESSED_NCCL_ALLGATHER_H
#define HOROVOD_COMPRESSED_NCCL_ALLGATHER_H
#include "reducer.h"
namespace horovod {
namespace common {

class NCCL_Allreduce_AllGather : public NCCLReducer {
public:
  NCCL_Allreduce_AllGather(NCCLContext* nccl_context, GPUContext* gpu_context,
                           GPUOpContext* gpu_op_context,
                           HorovodGlobalState* global_state,
                           Compressor* compressor);
  Status
  Init(const std::vector<horovod::common::TensorTableEntry>& entries) final;
  Status
  AllreduceDivision(int num_elements, ncclComm_t* nccl_comm_,
                    std::vector<horovod::common::TensorTableEntry>& entries,
                    unsigned char* buffer_ptr) final;

  size_t GetRequiredFreeSize() override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_COMPRESSED_NCCL_ALLGATHER_H
