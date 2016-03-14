#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>

using namespace std;

int main(int argc, char **argv) {
	ifstream in(argv[1]);
	int n, m;
	in >> n >> m;

	const bool undirected = (argc >= 4 && strcmp(argv[3], "undirected") == 0);

	vector< vector<int> > graph(n);

	for (int i = 0; i < m; ++i) {
		int a,b;
		in >> a >> b;
		graph[a].push_back(b);
		if (undirected) graph[b].push_back(a);
	}

	vector<char> visited(n, false);
	vector<unsigned> cost(n, (unsigned)-1);

	const int start = atoi(argv[2]);

	cost[start] = 0;

	queue<int> q;
	q.push(start);

	while (!q.empty()) {
		int curr = q.front();
		q.pop();
		visited[curr] = true;

		for (int i = 0; i < graph[curr].size(); ++i) {
			int next = graph[curr][i];
			if (!visited[next]) {
				q.push(next);
				cost[next] = cost[curr] + 1;
				visited[next] = true;
			}
		}
	}

	ofstream out("result.txt");

	for (int i = 0; i < n; ++i) {
		out << cost[i] << endl;
	}

	return 0;
}