#ifndef NCCL_SCATTER_ALLGATHER_H
#define NCCL_SCATTER_ALLGATHER_H
#include "reducer.h"

#if NCCL_VERSION_CHECK(2, 7, 0)
namespace horovod {
namespace common {
class NCCL_Allreduce_ScatterAllgather : public NCCLReducer {
public:
  NCCL_Allreduce_ScatterAllgather(NCCLContext* nccl_context, GPUContext* gpu_context,
                           GPUOpContext* gpu_op_context,
                           HorovodGlobalState* global_state,
                           Compressor* compressor, Summator* summator);
  Status
  Init(const std::vector<horovod::common::TensorTableEntry>& entries) final;
  Status
  AllreduceDivision(int num_elements, ncclComm_t* nccl_comm_,
                    std::vector<horovod::common::TensorTableEntry>& entries,
                    int64_t global_offset) final;

};

} // namespace common
} // namespace horovod

#endif
#endif // NCCL_SCATTER_ALLGATHER_H
