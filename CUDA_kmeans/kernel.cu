#include "kernel.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <curand_kernel.h> // Inclusione di cuRAND per generazione numeri casuali
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cmath>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA errore in " << __FILE__ << " linea " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void assegnaClusters(const double* punti, const double* centroidi, int* assegnamenti,
    int numPunti, int numCentroidi, int dimensioni) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPunti) return;

    double minDist = INFINITY;
    int migliorCluster = -1;

    for (int c = 0; c < numCentroidi; ++c) {
        double dist = 0.0;
        for (int d = 0; d < dimensioni; ++d) {
            double diff = punti[idx * dimensioni + d] - centroidi[c * dimensioni + d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < minDist) {
            minDist = dist;
            migliorCluster = c;
        }
    }
    assegnamenti[idx] = migliorCluster;
   // printf("Punto %d assegnato al cluster %d con distanza %.4f", idx, bestCluster, minDist);
}

__global__ void aggiornaCentroidi(const double* punti, double* nuoviCentroidi, int* assegnamenti,
    int numPunti, int numCentroidi, int dimensioni, int* grandezzeCluster) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPunti) return;

    int cluster = assegnamenti[idx];
    if (cluster == -1) return;

    for (int d = 0; d < dimensioni; ++d) {
        atomicAdd(&nuoviCentroidi[cluster * dimensioni + d], punti[idx * dimensioni + d]);
    }
    atomicAdd(&grandezzeCluster[cluster], 1);
}

__global__ void normalizzaCentroidi(double* centroidi, double* nuoviCentroidi, int* grandezzeCluster, int numCentroidi, int dimensioni) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCentroidi) return;

    int n = grandezzeCluster[idx];
    if (n > 0) {
        for (int d = 0; d < dimensioni; ++d) {
            centroidi[idx * dimensioni + d] = nuoviCentroidi[idx * dimensioni + d] / n;
        }
    }
}

void kmeans_cuda(double* d_punti, double* d_centroidi, int* d_assegnamenti,
    int numPunti, int numCentroidi, int dimensioni, int maxIter, double tol,
    std::vector<double>& h_centroidiPrecedenti, std::vector<double>& h_centroidiCorrenti) {

    int* d_grandezzeCluster;
    double* d_nuoviCentroidi;
    CUDA_CHECK(cudaMalloc(&d_grandezzeCluster, numCentroidi * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nuoviCentroidi, numCentroidi * dimensioni * sizeof(double)));

    int blockSize = 256;
    int gridSizePunti = (numPunti + blockSize - 1) / blockSize;
    int gridSizeCentroidi = (numCentroidi + blockSize - 1) / blockSize;

    // Inizializza numero di iterazioni e convergenza
    bool convergenza=false;
    int iter = 0;

    while(!convergenza){

        if (iter >= maxIter) {
            break;
        }

        CUDA_CHECK(cudaMemset(d_grandezzeCluster, 0, numCentroidi * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_nuoviCentroidi, 0, numCentroidi * dimensioni * sizeof(double)));

        assegnaClusters << <gridSizePunti, blockSize >> > (d_punti, d_centroidi, d_assegnamenti, numPunti, numCentroidi, dimensioni);
        CUDA_CHECK(cudaDeviceSynchronize());

        aggiornaCentroidi << <gridSizePunti, blockSize >> > (d_punti, d_nuoviCentroidi, d_assegnamenti, numPunti, numCentroidi, dimensioni, d_grandezzeCluster);
        CUDA_CHECK(cudaDeviceSynchronize());

        normalizzaCentroidi << <gridSizeCentroidi, blockSize >> > (d_centroidi, d_nuoviCentroidi, d_grandezzeCluster, numCentroidi, dimensioni);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_centroidiCorrenti.data(), d_centroidi, numCentroidi * dimensioni * sizeof(double), cudaMemcpyDeviceToHost));
 

        //***********************CONTROLLO CONVERGENZA*****************************
        for (int c = 0; c < numCentroidi; ++c) {
            double shift = 0.0;
            convergenza = false;

            for (int d = 0; d < dimensioni; ++d) {
                double diff = h_centroidiCorrenti[c * dimensioni + d] - h_centroidiPrecedenti[c * dimensioni + d];
                shift += diff * diff;
            }

            if ((std::sqrt(shift) / dimensioni) > tol) {
                //printf("Non convergente\n");
                break;
            }
            else {
                convergenza = true;
            }
        }

        
        iter++;

        // Aggiorno i vecchi centroidi per confronto successivo
        h_centroidiPrecedenti = h_centroidiCorrenti;



    }

    // Stampa il numero totale di iterazioni eseguite
    std::cout << "Numero di iterazioni per convergenza: " << iter << " \n";

    CUDA_CHECK(cudaFree(d_grandezzeCluster));
    CUDA_CHECK(cudaFree(d_nuoviCentroidi));
}
