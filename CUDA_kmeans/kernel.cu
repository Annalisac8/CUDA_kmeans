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

//kernel assegnazione dei punti ai cluster
__global__ void assegnaClusters(const double* punti, const double* centroidi, int* assegnamenti,
    int numPunti, int numCentroidi, int dimensioni) {
    
    //id del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPunti) return;

    double minDist = INFINITY;
    int migliorCluster = -1;

    //ogni thread calcola l'assegnamento per un punto 
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
    //assegno il punto al miglior cluster
    assegnamenti[idx] = migliorCluster;
   // printf("Punto %d assegnato al cluster %d con distanza %.4f", idx, bestCluster, minDist);
}
// kernel aggiornamento centroidi
__global__ void aggiornaCentroidi(const double* punti, double* nuoviCentroidi, int* assegnamenti,
    int numPunti, int numCentroidi, int dimensioni, int* grandezzeCluster) {
    //id del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPunti) return;

    //ogni thread aggiorna la somma delle coordinate per il proprio cluster (lavora su un punto aggiungendo i suoi valori al centroide)
    //uso atomicAdd per evitare race condition (più thread scrivono sulla stessa variabile)
    int cluster = assegnamenti[idx];
    if (cluster == -1) return;

    for (int d = 0; d < dimensioni; ++d) {
        atomicAdd(&nuoviCentroidi[cluster * dimensioni + d], punti[idx * dimensioni + d]);
    }
    atomicAdd(&grandezzeCluster[cluster], 1);
}

//kernel normalizza centroidi
__global__ void normalizzaCentroidi(double* centroidi, double* nuoviCentroidi, int* grandezzeCluster, int numCentroidi, int dimensioni) {
    //id del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCentroidi) return;

    //ogni thread lavora su un centroide e calcola la media, evito anche la divisione per zero controllando n
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

    //allocazione della memoria sulla gpu per somme coordinate nuovi centroidi e numero di punti per centroide
    int* d_grandezzeCluster;
    double* d_nuoviCentroidi;
    CUDA_CHECK(cudaMalloc(&d_grandezzeCluster, numCentroidi * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nuoviCentroidi, numCentroidi * dimensioni * sizeof(double)));

    //definisco il numero di blocchi e thread
    int blockSize = 256; //256 valore ottimale 
    int gridSizePunti = (numPunti + blockSize - 1) / blockSize;
    int gridSizeCentroidi = (numCentroidi + blockSize - 1) / blockSize;

    // Inizializza numero di iterazioni e convergenza
    bool convergenza=false;
    int iter = 0;

    //ciclo iterativo del kmeans
    while(!convergenza){

        if (iter >= maxIter) {
            break;
        }
        //reset delle strutture di supporto
        CUDA_CHECK(cudaMemset(d_grandezzeCluster, 0, numCentroidi * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_nuoviCentroidi, 0, numCentroidi * dimensioni * sizeof(double)));

        //assegna i punti ai cluster
        assegnaClusters << <gridSizePunti, blockSize >> > (d_punti, d_centroidi, d_assegnamenti, numPunti, numCentroidi, dimensioni);
        CUDA_CHECK(cudaDeviceSynchronize());

        //aggiorna i centroidi sommando le coordinate
        aggiornaCentroidi << <gridSizePunti, blockSize >> > (d_punti, d_nuoviCentroidi, d_assegnamenti, numPunti, numCentroidi, dimensioni, d_grandezzeCluster);
        CUDA_CHECK(cudaDeviceSynchronize());

        //normalizza i centroidi calcolando la media
        normalizzaCentroidi << <gridSizeCentroidi, blockSize >> > (d_centroidi, d_nuoviCentroidi, d_grandezzeCluster, numCentroidi, dimensioni);
        CUDA_CHECK(cudaDeviceSynchronize());

        //copio i nuovi centroidi sulla cpu per controllare la convergenza
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
    //std::cout << "Numero di iterazioni per convergenza: " << iter << " \n";

    //libero memoria allocata per evitare memory leak
    CUDA_CHECK(cudaFree(d_grandezzeCluster));
    CUDA_CHECK(cudaFree(d_nuoviCentroidi));
}
