#include "kernel.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void assign_clusters(double* points, double* centroids, int* assignments, int numPoints, int numCentroids, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    //printf("Thread %d (idx %d): Inizio assegnazione cluster.\n", threadIdx.x, idx);

    double minDist = 1e20;
    int bestCluster = 0;

    for (int c = 0; c < numCentroids; c++) {
        double dist = 0.0;
        for (int d = 0; d < dimensions; d++) {
            double diff = points[idx * dimensions + d] - centroids[c * dimensions + d];
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

/*
__global__ void update_centroids(double* points, double* centroids, int* assignments, int numPoints, int numCentroids, int dimensions) {
    extern __shared__ double sharedMem[];
    double* sums = sharedMem;
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

*/

#include <curand_kernel.h> // Inclusione di cuRAND per generazione numeri casuali

__global__ void update_centroids(double* points, double* centroids, int* assignments, int numPoints, int numCentroids, int dimensions, unsigned long seed) {
    extern __shared__ double sharedMem[];
    double* sums = sharedMem;
    int* clusterCounts = (int*)&sums[numCentroids * dimensions];

    int centroidIdx = blockIdx.x;
    if (centroidIdx >= numCentroids) return;

    // Inizializzazione memoria condivisa
    for (int d = threadIdx.x; d < dimensions; d += blockDim.x) {
        sums[centroidIdx * dimensions + d] = 0.0f;
    }
    if (threadIdx.x == 0) {
        clusterCounts[centroidIdx] = 0;
    }

    __syncthreads();

    // Sommare le coordinate dei punti assegnati a questo centroide
    for (int i = threadIdx.x; i < numPoints; i += blockDim.x) {
        if (assignments[i] == centroidIdx) {
            for (int d = 0; d < dimensions; d++) {
                atomicAdd(&sums[centroidIdx * dimensions + d], points[i * dimensions + d]);
            }
            atomicAdd(&clusterCounts[centroidIdx], 1);
        }
    }

    __syncthreads();

    // Un thread per blocco aggiorna il centroide
    if (threadIdx.x == 0) {
        if (clusterCounts[centroidIdx] > 0) {
            for (int d = 0; d < dimensions; d++) {
                centroids[centroidIdx * dimensions + d] = sums[centroidIdx * dimensions + d] / clusterCounts[centroidIdx];
            }
        }
        else {
            // Se un cluster è vuoto, assegniamo un punto casuale usando cuRAND
            curandState state;
            curand_init(seed + centroidIdx, 0, 0, &state);
            int randomIdx = curand(&state) % numPoints;

            for (int d = 0; d < dimensions; d++) {
                centroids[centroidIdx * dimensions + d] = points[randomIdx * dimensions + d];
            }
            printf("Centroide %d era vuoto e riassegnato al punto %d\n", centroidIdx, randomIdx);
        }
        //printf("Centroide %d aggiornato con %d punti\n", centroidIdx, clusterCounts[centroidIdx]);
    }
}

 
void kmeans_cuda(double* d_points, double* d_centroids, int* d_assignments, int numPoints, int numCentroids, int dimensions, int maxIterations, double tolerance, std::vector<double>& h_oldCentroids, std::vector<double>& h_currentCentroids) {

    if (d_points == nullptr || d_centroids == nullptr || d_assignments == nullptr) {
        //printf("Errore: Puntatori GPU non validi.\n");
        return;
    }

    //printf("Inizio kmeans_cuda (numPoints: %d, numCentroids: %d, dimensions: %d)\n", numPoints, numCentroids, dimensions);

    // Allocazione memoria per i centroidi precedenti
    double* d_oldCentroids;
    //printf("Allocazione d_oldCentroids (numCentroids: %d, dimensions: %d)...\n", numCentroids, dimensions);
    CUDA_CHECK(cudaMalloc(&d_oldCentroids, numCentroids * dimensions * sizeof(double)));
    //printf("Allocazione d_oldCentroids completata con successo.\n");

    // Configurazione per i kernel
    dim3 threadsPerBlock(128);
    dim3 blocksPerGrid((numPoints + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemSize = sizeof(double) * dimensions * threadsPerBlock.x + sizeof(int) * threadsPerBlock.x;
    //printf("Memoria condivisa allocata per blocco: %lu bytes\n", sharedMemSize);

    for (int iter = 0; iter < maxIterations; iter++) {


        //printf("Inizio iterazione %d\n", iter);

        // Copia d_centroids in d_oldCentroids
        CUDA_CHECK(cudaMemcpy(d_oldCentroids, d_centroids, numCentroids * dimensions * sizeof(double), cudaMemcpyDeviceToDevice));
        //printf("Copia di d_centroids in d_oldCentroids completata con successo.\n");

        // Kernel per assegnare i punti ai cluster
        assign_clusters << <blocksPerGrid, threadsPerBlock >> > (d_points, d_centroids, d_assignments, numPoints, numCentroids, dimensions);
        CUDA_CHECK(cudaDeviceSynchronize());
        //printf("Kernel assign_clusters completato senza errori.\n");

        // Kernel per aggiornare i centroidi
        //update_centroids << <numCentroids, threadsPerBlock, sharedMemSize >> > (d_points, d_centroids, d_assignments, numPoints, numCentroids, dimensions);
        //CUDA_CHECK(cudaDeviceSynchronize());
        //printf("Kernel update_centroids completato senza errori.\n");
        unsigned long seed = time(NULL); // Definiamo un seed casuale per cuRAND
        int sharedMemorySize = numCentroids * dimensions * sizeof(double) + numCentroids * sizeof(int);

        update_centroids << <numCentroids, 256, sharedMemorySize >> > (d_points, d_centroids,d_assignments, numPoints, numCentroids, dimensions, seed);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copia dei dati dalla GPU per la verifica della convergenza
        std::vector<double> h_oldCentroids(numCentroids * dimensions);
        CUDA_CHECK(cudaMemcpy(h_oldCentroids.data(), d_oldCentroids, numCentroids * dimensions * sizeof(double), cudaMemcpyDeviceToHost));
        //printf("Copia di d_oldCentroids nella memoria host completata con successo.\n");

        std::vector<double> h_currentCentroids(numCentroids * dimensions);
        CUDA_CHECK(cudaMemcpy(h_currentCentroids.data(), d_centroids, numCentroids * dimensions * sizeof(double), cudaMemcpyDeviceToHost));
        // printf("Copia di d_centroids nella memoria host completata con successo.\n");



         // Verifica della convergenza
        //double maxChange = 0.0f;
        double change = 0.0;
        bool converged = true;

        
        for (int i = 0; i < numCentroids * dimensions; i++) {

            
            //if (h_currentCentroids[i] != h_oldCentroids[i]) {
            //    converged = false; // Se anche un solo valore differisce, non c'è convergenza

                change += (h_currentCentroids[i] - h_oldCentroids[i]) / h_currentCentroids[i] * 100.0;
                double change = fabs(h_currentCentroids[i] - h_oldCentroids[i]);
                //if (change > maxChange) maxChange = change;
            //}
        }

        
        if (fabs(change) < tolerance) {
            printf("Convergenza Cuda raggiunta dopo %d iterazioni.\n", iter + 1);
            break;
        }
        
        
    /*
        if (converged) {
            std::cout << "Convergenza raggiunta dopo " << iter+1 << " iterazioni.\n";
            break; // Esci dal ciclo principale
        }
       */ 

        // printf("Massima variazione: %f\n", maxChange);
        /*
         if (maxChange < tolerance) {
             printf("Convergenza Cuda raggiunta dopo %d iterazioni.\n", iter + 1);
             break;
         }
         */

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
