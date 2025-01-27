#include "kernel.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void assign_clusters(float* points, float* centroids, int* assignments, int numPoints, int numCentroids, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    //printf("Thread %d (idx %d): Inizio assegnazione cluster.\n", threadIdx.x, idx);

    float minDist = 1e20;
    int bestCluster = 0;

    for (int c = 0; c < numCentroids; c++) {
        float dist = 0.0;
        for (int d = 0; d < dimensions; d++) {
            float diff = points[idx * dimensions + d] - centroids[c * dimensions + d];
            dist += diff * diff;
        }
        if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
        }
    }
    if (idx >= numPoints) {
        //printf("Errore: Thread %d, idx %d fuori dai limiti (numPoints: %d)\n", threadIdx.x, idx, numPoints);
        return;
    }
    assignments[idx] = bestCluster;

    //printf("Thread %d (idx %d): Assegnato al cluster %d.\n", threadIdx.x, idx, bestCluster);
}

__global__ void update_centroids(float* points, float* centroids, int* assignments, int numPoints, int numCentroids, int dimensions) {
    extern __shared__ float sharedMem[];
    float* sums = sharedMem;
    int* clusterCounts = (int*)&sums[numCentroids * dimensions];

    int centroidIdx = blockIdx.x;
    if (centroidIdx >= numCentroids) return;

    for (int d = threadIdx.x; d < dimensions; d += blockDim.x) {
        sums[centroidIdx * dimensions + d] = 0.0f;
    }
    if (threadIdx.x == 0) clusterCounts[centroidIdx] = 0;

    __syncthreads();

    for (int i = threadIdx.x; i < numPoints; i += blockDim.x) {
        if (assignments[i] == centroidIdx) {
            for (int d = 0; d < dimensions; d++) {
                atomicAdd(&sums[centroidIdx * dimensions + d], points[i * dimensions + d]);
            }
            atomicAdd(&clusterCounts[centroidIdx], 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int d = 0; d < dimensions; d++) {
            if (clusterCounts[centroidIdx] > 0)
                centroids[centroidIdx * dimensions + d] = sums[centroidIdx * dimensions + d] / clusterCounts[centroidIdx];
        }

        //printf("Centroide %d aggiornato: ", centroidIdx);
        for (int d = 0; d < dimensions; d++) {
            //printf("%f ", centroids[centroidIdx * dimensions + d]);
        }
        //printf("\n");
    }
}

void kmeans_cuda(float* d_points, float* d_centroids, int* d_assignments, int numPoints, int numCentroids, int dimensions, int maxIterations, float tolerance, std::vector<float>& h_oldCentroids, std::vector<float>& h_currentCentroids) {
  
    if (d_points == nullptr || d_centroids == nullptr || d_assignments == nullptr) {
        //printf("Errore: Puntatori GPU non validi.\n");
        return;
    }

    //printf("Inizio kmeans_cuda (numPoints: %d, numCentroids: %d, dimensions: %d)\n", numPoints, numCentroids, dimensions);

    // Allocazione memoria per i centroidi precedenti
    float* d_oldCentroids;
    //printf("Allocazione d_oldCentroids (numCentroids: %d, dimensions: %d)...\n", numCentroids, dimensions);
    CUDA_CHECK(cudaMalloc(&d_oldCentroids, numCentroids * dimensions * sizeof(float)));
    //printf("Allocazione d_oldCentroids completata con successo.\n");

    // Configurazione per i kernel
    dim3 threadsPerBlock(128);
    dim3 blocksPerGrid((numPoints + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemSize = sizeof(float) * dimensions * threadsPerBlock.x + sizeof(int) * threadsPerBlock.x;
    //printf("Memoria condivisa allocata per blocco: %lu bytes\n", sharedMemSize);

    for (int iter = 0; iter < maxIterations; iter++) {

        
        //printf("Inizio iterazione %d\n", iter);

        // Copia d_centroids in d_oldCentroids
        CUDA_CHECK(cudaMemcpy(d_oldCentroids, d_centroids, numCentroids * dimensions * sizeof(float), cudaMemcpyDeviceToDevice));
        //printf("Copia di d_centroids in d_oldCentroids completata con successo.\n");

        // Kernel per assegnare i punti ai cluster
        assign_clusters << <blocksPerGrid, threadsPerBlock >> > (d_points, d_centroids, d_assignments, numPoints, numCentroids, dimensions);
        CUDA_CHECK(cudaDeviceSynchronize());
        //printf("Kernel assign_clusters completato senza errori.\n");

        // Kernel per aggiornare i centroidi
        update_centroids << <numCentroids, threadsPerBlock, sharedMemSize >> > (d_points, d_centroids, d_assignments, numPoints, numCentroids, dimensions);
        CUDA_CHECK(cudaDeviceSynchronize());
        //printf("Kernel update_centroids completato senza errori.\n");

        // Copia dei dati dalla GPU per la verifica della convergenza
        std::vector<float> h_oldCentroids(numCentroids * dimensions);
        CUDA_CHECK(cudaMemcpy(h_oldCentroids.data(), d_oldCentroids, numCentroids * dimensions * sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Copia di d_oldCentroids nella memoria host completata con successo.\n");

        std::vector<float> h_currentCentroids(numCentroids * dimensions);
        CUDA_CHECK(cudaMemcpy(h_currentCentroids.data(), d_centroids, numCentroids * dimensions * sizeof(float), cudaMemcpyDeviceToHost));
       // printf("Copia di d_centroids nella memoria host completata con successo.\n");

        
        // Verifica della convergenza
        //float maxChange = 0.0f;
        bool converged = true;

        for (int i = 0; i < numCentroids * dimensions; i++) {

            if (h_currentCentroids[i] != h_oldCentroids[i]) {
                converged = false; // Se anche un solo valore differisce, non c'è convergenza


                //float change = fabs(h_currentCentroids[i] - h_oldCentroids[i]);
                //if (change > maxChange) maxChange = change;
            }
        }
        
        if (converged) {
            std::cout << "Convergenza raggiunta dopo " << iter << " iterazioni.\n";
            break; // Esci dal ciclo principale
        }
       // printf("Massima variazione: %f\n", maxChange);

        //if (maxChange < tolerance) {
        //    printf("Convergenza Cuda raggiunta dopo %d iterazioni.\n", iter + 1);
        //    break;
        //}

        //printf("Iterazione %d completata senza errori.\n", iter);
    }

    // Rilascio della memoria GPU
    //printf("Rilascio della memoria GPU in kmeans_cuda...\n");
    if (d_oldCentroids) {
        CUDA_CHECK(cudaFree(d_oldCentroids));
        //printf("Memoria d_oldCentroids rilasciata con successo.\n");
    }

    //printf("kmeans_cuda completato senza errori.\n");
}
