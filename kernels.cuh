#include <assert.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "Error: %s\nFile %s, line %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

#define MAX_THREADS_PER_BLOCK 256

#define FALSE 0u
#define  TRUE 1u

__device__ unsigned *activeMask;
__managed__ unsigned terminate = TRUE;
__managed__ unsigned numActiveThreads;
