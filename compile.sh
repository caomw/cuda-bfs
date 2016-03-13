set -e
nvcc -std=c++11 -arch=sm_35 -dc kernels.cu bfs.cu
nvcc -std=c++11 -arch=sm_35 -dlink kernels.o bfs.o -o dlink.o
g++ kernels.o bfs.o dlink.o main.cpp -lcudart -L/usr/local/cuda/lib64 -o main
