#include "bfs.hpp"

int main() {

    Graph graph(1);
    std::vector<unsigned> distances;
    BFS(graph, 0, distances);

    return 0;
}

