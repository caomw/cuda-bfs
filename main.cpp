#include <iostream>
#include "bfs.hpp"

int main() {

    Graph graph(2);
    graph[0].push_back(1);
    
    std::vector<unsigned> distances;
    BFS(graph, 0, distances);

    for (int i = 0; i < distances.size(); ++i) {
        std::cout << distances[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

