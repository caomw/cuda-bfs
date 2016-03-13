#include "bfs.hpp"
#include "bfs_kernels.cuh"
#include "compaction.cuh"
#include <stdio.h>

extern __managed__ unsigned terminate;
extern __managed__ unsigned numActiveThreads;

__host__
void setUInt(unsigned *address, unsigned value) {
    gpuErrchk(cudaMemcpy(address, &value, sizeof(unsigned), cudaMemcpyHostToDevice));
}

__global__
void output(int N, unsigned *ptr) {
    printf("Prefix sums: ");
    for (int i = 0; i < N; ++i) {
        printf("%u ", ptr[i]);
    }
    printf("\n");
}

__host__
void BFS(Graph & graph, unsigned sourceVertex, std::vector<unsigned> & distances) {

    assert(sizeof(unsigned) == 4);
    
    distances.clear();
    distances.resize(graph.size());

    // Convert the graph to GPU representation

    size_t totalEdges = 0;
    for (auto & neighborsList : graph) {
        totalEdges += neighborsList.size();
    }

    std::vector<unsigned> V(graph.size() + 1);
    std::vector<unsigned> E;
    E.reserve(totalEdges);

    for (size_t v = 0; v < graph.size(); ++v) {
        V[v] = E.size();
        for (int neighbor : graph[v]) {
            E.push_back(neighbor);
        }
    }
    V[graph.size()] = totalEdges;

printf("V: ");
for (int x : V) printf("%u ", x);
printf("\n");

printf("E: ");
for (int x : E) printf("%u ", x);
printf("\n");

    // Memory allocation and setup

    unsigned *d_V, *d_E;
    unsigned *d_F, *d_X, *d_C, *d_Fu;
    unsigned *activeMask, *prefixSums;

    size_t memSize = (graph.size() + 1) * sizeof(unsigned);
    
    gpuErrchk(cudaMalloc(&d_F, memSize));
    gpuErrchk(cudaMemset(d_F, FALSE, memSize));
    setUInt(d_F + sourceVertex, TRUE); // add source to frontier

    gpuErrchk(cudaMalloc(&d_X, memSize));
    gpuErrchk(cudaMemset(d_X, FALSE, memSize));
    setUInt(d_X + sourceVertex, TRUE); // set source as visited

    gpuErrchk(cudaMalloc(&d_C, memSize));
    gpuErrchk(cudaMemset(d_C, 255, memSize));
    setUInt(d_C + sourceVertex, FALSE); // set zero distance to source

    gpuErrchk(cudaMalloc(&d_Fu, memSize));

    gpuErrchk(cudaMalloc(&d_V, memSize));
    gpuErrchk(cudaMemcpy(d_V, V.data(), memSize, cudaMemcpyHostToDevice));

    size_t memSizeE = totalEdges * sizeof(unsigned);
    gpuErrchk(cudaMalloc(&d_E, memSizeE));
    gpuErrchk(cudaMemcpy(d_E, E.data(), memSizeE, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&activeMask, memSize));
    setUInt(activeMask + 0, sourceVertex); // set thread #source as active
    numActiveThreads = 1;

    gpuErrchk(cudaMalloc(&prefixSums, memSize));
    preallocBlockSums(graph.size() + 1);

    // Main loop

    printf("Settled\n"); fflush(stdout);

    const size_t prefixSumGridSize = 
        (graph.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    while (true) {

        terminate = TRUE;

        const size_t gridSize = 
            (numActiveThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        printf("Kernel 1, <<<%d, %d>>>\n", gridSize, BLOCK_SIZE); fflush(stdout);
        // launch kernel 1
        BFSKernel1 <<<gridSize, BLOCK_SIZE>>> (graph.size(), d_V, d_E, d_F, d_X, d_C, d_Fu);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        printf("Kernel 2, <<<%d, %d>>>...", gridSize, BLOCK_SIZE); fflush(stdout);
        // launch kernel 2
        BFSKernel2 <<<gridSize, BLOCK_SIZE>>> (graph.size(), d_F, d_X, d_Fu);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        printf("done\n"); fflush(stdout);

        if (terminate) {
            break;
        } else {
            // Get prefix sums of F
            prescanArray(prefixSums, d_F, graph.size() + 1);
            
            output <<<1,1>>> (graph.size(), prefixSums);
            gpuErrchk(cudaDeviceSynchronize());

            const size_t gridSizeCompaction = (graph.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compactSIMD <<<gridSizeCompaction, BLOCK_SIZE>>> (prefixSums, activeMask);

            printf("Kernel 3, <<<1, 1>>>\n"); fflush(stdout);
            getActiveMaskTemp <<<1, 1>>> (graph.size(), d_F, activeMask);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
    }

    // Download result

    gpuErrchk(cudaMemcpy(distances.data(), d_C, memSize-sizeof(unsigned), cudaMemcpyDeviceToHost));

    // Free memory

    gpuErrchk(cudaFree(d_F));
    gpuErrchk(cudaFree(d_X));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_Fu));
    gpuErrchk(cudaFree(d_V));
    gpuErrchk(cudaFree(d_E));
    gpuErrchk(cudaFree(activeMask));
    deallocBlockSums();
    gpuErrchk(cudaFree(prefixSums));
}
