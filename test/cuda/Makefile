.PHONY: app quantization_test topk_test memcpy
#objects = cuda_test.o cuda_functions.o
#files = cuda_test.cu cuda_functions.cu.cc
horovod_cuda_home = ../../horovod/common/ops/compressed/compression/cuda/
horovod_home = ../../horovod/common
allreduce_files = nccl_allreduce.cc reductions.cc shared_memory.cu.cc $(horovod_cuda_home)/cuda_compression_functions.cu $(horovod_home)/common.cc $(horovod_home)/logging.cc
allreduce_files = nccl_allreduce.cc reductions.cc shared_memory.cu.cc compression_utils.cc
quantization_test_files = quantization_test.cu $(horovod_cuda_home)/cuda_compression_functions.cu $(horovod_home)/common.cc $(horovod_home)/logging.cc
topk_test_files = topk_test.cu $(horovod_cuda_home)/topk_compression.cu $(horovod_home)/common.cc $(horovod_home)/logging.cc

CUDA_HOME ?= /usr/local/cuda
MPI_HOME ?= /usr/local/openmpi
VERBOSE ?= 0
DEBUG ?= 0

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC = $(CUDA_HOME)/bin/nvcc

NVCC_GENCODE = -gencode arch=compute_70,code=sm_70 \

NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11
NVLDFLAGS  := -L${CUDA_LIB} -lcudart -lrt

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3 -g
else
NVCUFLAGS += -O0 -G -g
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
else
.SILENT:
endif

.PHONY: build clean

ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

ifneq ($(MPI_HOME), "")
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif
LIBRARIES += nccl
NVLDFLAGS += $(LIBRARIES:%=-l%)


ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

app:
	nvcc -x cu  -o app $(NVCUFLAGS) $(allreduce_files) $(NVLDFLAGS)

memcpy:
	nvcc -x cu -o memcpy $(NVCUFLAGS) memcpy.cu $(NVLDFLAGS)
#%.o: %.cu: %.cc.cu
#	nvcc -x cu -gencode arch=compute_75,code=sm_75 -I. -dc $< -o $@

quantization_test:
	nvcc -x cu -o quantization_test $(NVCUFLAGS) $(quantization_test_files) $(NVLDFLAGS)

topk_test:
	nvcc -x cu -o topk_test $(NVCUFLAGS) $(topk_test_files) $(NVLDFLAGS)

clean:
	rm -f *.o app shared_memory memcpy