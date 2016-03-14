#include <iostream>
#include <fstream>
#include "bfs.hpp"

int main(int argc, char **argv) {

    std::ifstream in(argv[1]);
    int n; in >> n;
    Graph graph(n);
    int m; in >> m;

    for (int i = 0; i < m; ++i) {
        int a, b;
        in >> a >> b;
        graph[a].push_back(b);
    }
    in.close();

    std::vector<unsigned> distances;
    BFS(graph, 0, distances);
    
    ofstream out("result_gpu.txt");
    for (int i = 0; i < distances.size(); ++i) {
        out << distances[i] << " ";
    }

    return 0;
}

