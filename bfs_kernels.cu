#include "bfs_kernels.cuh"

__managed__ unsigned terminate;
__managed__ unsigned numActiveThreads;

__global__
void BFSKernel1(
    size_t graphSize, unsigned *V, unsigned *E, unsigned *F, 
    unsigned *X, unsigned *C, unsigned *Fu) {

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // If vertex v exists and is active at current iteration
    if (v < graphSize and F[v]) {

        // Remove v from current frontier
        F[v] = FALSE;

        // Iterate over v's neighbors
        for (unsigned edge = V[v]; edge < V[v+1]; ++edge) {
            unsigned neighbor = E[edge];

            // If neighbor wasn't visited
            if (not X[neighbor]) {
                C[neighbor] = C[v] + 1;
                Fu[neighbor] = TRUE;
            }
        }
    }
}

__global__
void BFSKernel2(size_t graphSize, unsigned *F, unsigned *X, unsigned *Fu) {

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // If vertex v exists and has recently joined the frontier
    if (v < graphSize and Fu[v]) {
        // Copy the new frontier into F
        F[v] = TRUE;
        // Set v as visited
        X[v] = TRUE;
        // Clean up the new frontier
        Fu[v] = FALSE;
        terminate = FALSE;
    }
}

__global__
void getActiveMaskTemp(size_t graphSize, unsigned *F, unsigned *activeMask) {
    printf("K3, F[] = ");
    for (int i = 0; i < graphSize; ++i) {
        printf("%u ", F[i]);
    }
    printf("\n");

    numActiveThreads = 0;
    for (int i = 0; i < graphSize; ++i) {
        if (F[i]) {
            activeMask[numActiveThreads] = i;
            ++numActiveThreads;
        }
    }

    printf("K3, activeMask[AFTER] = ");
    for (int i = 0; i < numActiveThreads; ++i) {
        printf("%u ", activeMask[i]);
    }
    printf("\n");

    printf("K3, nAT = %u\n", numActiveThreads);
}
