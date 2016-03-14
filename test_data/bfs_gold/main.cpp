#include <iostream>
#include <fstream>
#include <vector>
#include <queue>

using namespace std;

int main(int argc, char **argv) {
	ifstream in(argv[1]);
	int n, m;
	in >> n >> m;

	vector<vector<int>> graph(n);

	for (int i = 0; i < m; ++i) {
		int a,b;
		in >> a >> b;
		graph[a].push_back(b);
		graph[b].push_back(a);
	}

	vector<char> visited(n, 0);
	vector<unsigned> cost(n, (unsigned)-1);

	const int start = 2289;

	cost[start] = 0;

	queue<int> q;
	q.push(2289);

	while (!q.empty()) {
		int curr = q.front();
		q.pop();
		visited[curr] = true;

		for (int i = 0; i < graph[curr].size(); ++i) {
			if (!visited[graph[curr][i]]) {
				q.push(graph[curr][i]);
				cost[graph[curr][i]] = cost[curr] + 1;
			}
		}
	}

	ofstream out("result.txt");

	for (int i = 0; i < n; ++i) {
		out << cost[i] << " ";
	}

	return 0;
}