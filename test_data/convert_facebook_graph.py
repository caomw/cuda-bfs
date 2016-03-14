from sys import argv

edges = []
maxV = -1

with open(argv[1], 'r') as f:
	for line in f:
		edges.append(list(map(int, line.split())))
		maxV = max(maxV, edges[-1][0])
		maxV = max(maxV, edges[-1][1])

with open(argv[2], 'w') as out:
	out.write(str(maxV) + ' ' + str(len(edges)) + '\n')
	for edge in edges:
		out.write(str(edge[0] - 1) + ' ' + str(edge[1] - 1) + '\n')
