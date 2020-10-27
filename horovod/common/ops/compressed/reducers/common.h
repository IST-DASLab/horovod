#ifndef HOROVOD_COMPRESSION_REDUCTION_COMMON_H
#define HOROVOD_COMPRESSION_REDUCTION_COMMON_H
#include "../../../mpi/mpi_context.h"
#include "../common.h"

#define MPI_CHECK(condition)                                                   \
  do {                                                                         \
    int op = condition;                                                        \
    if (op != MPI_SUCCESS) {                                                   \
      throw std::runtime_error(std::string(#condition) + " on line " +         \
                               std::to_string(__LINE__) + " failed: ");        \
    }                                                                          \
  } while (0)

#define SYS_CHECK(call, name)                                                  \
  do {                                                                         \
    int retval;                                                                \
    SYS_CHECKVAL(call, name, retval);                                           \
  } while (false)

#define SYS_CHECKSYNC(call, name, retval)                                      \
  do {                                                                         \
    retval = call;                                                             \
    if (retval == -1 &&                                                        \
        (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {         \
      printf("Call to " name " returned %s, retrying", strerror(errno));       \
    } else {                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (true)

#define SYS_CHECKVAL(call, name, retval)                                       \
  do {                                                                         \
    SYS_CHECKSYNC(call, name, retval);                                         \
    if (retval == -1) {                                                        \
      printf("Call to " name " failed : %s", strerror(errno));                 \
      return 1;                                                                \
    }                                                                          \
  } while (false)

#define TRIV_CHECK(cmd)                                                        \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != 0) {                                                              \
      printf("Failed: Error %s:%d'\n", __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif // HOROVOD_COMPRESSION_REDUCTION_COMMON_H
