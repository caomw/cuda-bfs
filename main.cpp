#include <iostream>
#include <fstream>
#include "bfs.hpp"

int main() {

    std::ifstream in("graph.txt");
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
    
    for (int i = 0; i < distances.size(); ++i) {
        std::cout << distances[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

