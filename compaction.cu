#include "scan.cu"

extern __managed__ unsigned numActiveThreads;

__global__
void compactSIMD(unsigned *prefixSums, unsigned *activeMask, size_t blockSize) {
    size_t blockStart = blockIdx.x * blockSize;
    size_t blockEnd = (blockIdx.x + 1) * blockSize;

    unsigned validElemsNumber = prefixSums[blockEnd] - prefixSums[blockStart];
    
}