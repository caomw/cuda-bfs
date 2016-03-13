#include "bfs.hpp"
#include "kernels.cuh"

__host__
void setUInt(unsigned *address, unsigned value) {

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

    // Memory allocation and setup

    unsigned *d_V, *d_E;
    unsigned *d_F, *d_X, *d_C, *d_Fu;

    size_t memSize = (graph.size() + 1) * sizeof(unsigned);
    
    gpuErrchk(cudaMalloc(&d_F, memSize));
    gpuErrchk(cudaMemset(d_F, FALSE, memSize));
    setUInt(d_F + sourceVertex, TRUE); // add source to frontier

    gpuErrchk(cudaMalloc(&d_X, memSize));
    gpuErrchk(cudaMemset(d_X, FALSE, memSize));
    setUInt(d_X + sourceVertex, TRUE); // set source as visited

    gpuErrchk(cudaMalloc(&d_C, memSize));
    setUInt(d_C + sourceVertex, FALSE); // set zero distance to source

    gpuErrchk(cudaMalloc(&d_Fu, memSize));

    gpuErrchk(cudaMalloc(&d_V, memSize));
    gpuErrchk(cudaMemcpy(d_V, V.data(), memSize, cudaMemcpyHostToDevice));

    size_t memSizeE = totalEdges * sizeof(unsigned);
    gpuErrchk(cudaMalloc(&d_E, memSizeE));
    gpuErrchk(cudaMemcpy(d_E, E.data(), memSizeE, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&activeMask, memSize));
    gpuErrchk(cudaMemset(activeMask, FALSE, memSize));
    setUInt(activeMask + sourceVertex, TRUE); // set thread #source as active
    numActiveThreads = 1;

    terminate = TRUE;

    // Main loop

    const size_t prefixSumGridSize = 
        (graph.size() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    while (true) {
        const size_t gridSize = 
            (numActiveThreads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        // launch kernel 1
        BFSKernel1 <<<gridSize, MAX_THREADS_PER_BLOCK>>> (graph.size(), d_V, d_E, d_F, d_X, d_C, d_Fu);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // launch kernel 2
        BFSKernel2 <<<gridSize, MAX_THREADS_PER_BLOCK>>> (graph.size(), d_F, d_X, d_Fu);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // copy terminate from GPU

        if (terminate) {
            break;
        } else {
            // Get active threads list
            //prefixSum <<<prefixSumGridSize, MAX_THREADS_PER_BLOCK>>> (d_F, activeMask);
            //gather <<<
            getActiveMaskTemp <<<1, 1>>> (d_F, activeMask);

            //numActiveThreads
        }
    }

    // Download result

    gpuErrchk(cudaMemcpy(distances.data(), d_C, memSize, cudaMemcpyDeviceToHost));

    // Free memory

    gpuErrchk(cudaFree(d_F));
    gpuErrchk(cudaFree(d_X));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_Fu));
    gpuErrchk(cudaFree(d_V));
    gpuErrchk(cudaFree(d_E));
    gpuErrchk(cudaFree(activeMask));
}
