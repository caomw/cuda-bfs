#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "bfs.hpp"

int main(int argc, char **argv) {

    assert(argc >= 3);

    std::ifstream in(argv[1]);
    int n; in >> n;
    Graph graph(n);
    int m; in >> m;

    const int start = atoi(argv[2]);
    const bool undirected = (argc >= 4 && strcmp(argv[3], "undirected") == 0);

    for (int i = 0; i < m; ++i) {
        int a, b;
        in >> a >> b;
        graph[a].push_back(b);
        if (undirected) graph[b].push_back(a);
    }
    in.close();

    std::vector<unsigned> distances;
    BFS(graph, start, distances);
    
    std::ofstream out("result_gpu.txt");
    for (int i = 0; i < distances.size(); ++i) {
        out << distances[i] << std::endl;
    }

    return 0;
}

