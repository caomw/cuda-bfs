__host__
void preallocBlockSums(unsigned maxNumElements);

__host__
void deallocBlockSums();

__host__
void prescanArray(unsigned *outArray, unsigned *inArray, int numElements);

__global__
void compactSIMD(unsigned *prefixSums, unsigned *activeMask);