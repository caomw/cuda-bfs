#include "bfs.hpp"
#include "kernels.cuh"

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

    std::vector<unsigned> V(graph.size());
    std::vector<unsigned> E;
    E.reserve(totalEdges);

    for (size_t v = 0; v < graph.size(); ++v) {
        V[v] = E.size();
        for (int neighbor : graph[v]) {
            E.push_back(neighbor);
        }
    }

    // Memory allocation and setup

    unsigned *d_V, *d_E;
    unsigned *d_F, *d_X, *d_C, *d_Fu;
    
    size_t memSize = graph.size() * sizeof(unsigned);
    
    gpuErrchk(cudaMalloc(&d_F, memSize));
    gpuErrchk(cuMemsetD32(d_F, 0u, memSize));
    gpuErrchk(cuMemsetD32(d_F + sourceVertex, 1u, 1)); // add source to frontier

    gpuErrchk(cudaMalloc(&d_X, memSize));
    gpuErrchk(cuMemsetD32(d_X, 0u, memSize));
    gpuErrchk(cuMemsetD32(d_X + sourceVertex, 1u, 1)); // set source as visited

    gpuErrchk(cudaMalloc(&d_C, memSize));
    gpuErrchk(cuMemsetD32(d_C + sourceVertex, 0u, 1)); // set zero distance to source

    gpuErrchk(cudaMalloc(&d_Fu, memSize));

    gpuErrchk(cudaMalloc(&d_V, memSize));
    gpuErrchk(cudaMemcpy(V.data(), d_V, memSize, cudaMemcpyHostToDevice));

    size_t memSizeE = totalEdges * sizeof(unsigned);
    gpuErrchk(cudaMalloc(&d_E, memSizeE));
    gpuErrchk(cudaMemcpy(E.data(), d_E, memSizeE, cudaMemcpyHostToDevice));

    bool terminate = false;

    while (!terminate) {
        // launch kernel 1
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // launch kernel 2
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        terminate = true;
        // copy terminate from GPU
    }

    gpuErrchk(cudaMemcpy(distances.data(), d_C, memSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_F));
    gpuErrchk(cudaFree(d_X));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_Fu));
    gpuErrchk(cudaFree(d_V));
    gpuErrchk(cudaFree(d_E));
}