#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Macro per il controllo degli errori CUDA
#define CUDA_CHECK(err) { gpuAssert((err), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void kmeans_cuda(float* d_points, float* d_centroids, int* d_assignments, int numPoints, int numCentroids, int dimensions, int maxIterations, float tolerance, std::vector<float>& h_oldCentroids, std::vector<float>& h_currentCentroids);
