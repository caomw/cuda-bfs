__host__
void preallocBlockSums(unsigned int maxNumElements);

__host__
void deallocBlockSums();

__host__
void prescanArray(unsigned *outArray, unsigned *inArray, int numElements);