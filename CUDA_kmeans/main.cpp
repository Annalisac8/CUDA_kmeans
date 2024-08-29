
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <filesystem>
#include <cstdlib>   // Per exit
#include "sequential_kmeans.h"
#include "Punto.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include "sm_20_atomic_functions.h"
#include "sm_60_atomic_functions.h"
#include <cmath>
#include <iostream>



#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <filesystem>
#include <cstdlib>   // Per exit
#include "sequential_kmeans.h"
#include "Punto.h"
#include "kernel.cuh"

/*int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}*/


#include <iostream>


#include <stdio.h>


// Funzione per ottenere la media dei tempi di esecuzione
double calcolaTempoMedio(const std::vector<double>& tempi) {
    return std::accumulate(tempi.begin(), tempi.end(), 0.0) / tempi.size();
}


int main(int argc, char* argv[]) {

   
    std::vector<std::string> filenames = {
        "1000x10.txt",
        "1000x100.txt",
        "1000x1000.txt",
        "10000x10.txt",
        "10000x100.txt",
        "10000x1000.txt",
        "100000x10.txt",
        "100000x100.txt",
        
        // files
    };

    const int numEsecuzioni = 10; // Numero di esecuzioni per calcolare la media

    // Vettori per memorizzare i risultati finali
    std::vector<std::string> fileNamesVec;
    std::vector<double> tempiMediSeq;
    std::vector<double> tempiMediCuda;

    for (const auto& filename : filenames) {
        std::vector<double> tempiSeq, tempiCuda;

        for (int esecuzione = 0; esecuzione < numEsecuzioni; esecuzione++) {
            std::string ds_path = "ds/" + filename;
            std::ifstream dataset_file(ds_path);
            if (!dataset_file) {
                std::cerr << "Impossibile aprire il file: " << ds_path << std::endl;
                continue;
            }

            std::vector<Punto> ds;
            std::string riga;
            double valore;

            while (getline(dataset_file, riga)) {
                std::istringstream iss(riga);
                Punto punto;
                while (iss >> valore) {
                    punto.dimensioni.push_back(valore);
                }
                ds.push_back(punto);
            }
            dataset_file.close();

            auto numPunti = ds.size();
            auto dimPunti = ds[0].dimensioni.size();

            int k = 10; // Definisco il numero di cluster come fisso o calcolalo in base a qualche regola

            // Verifico che il valore di k sia valido
            if (k <= 0 || k > numPunti) {
                std::cerr << "Numero di cluster non valido: " << k << std::endl;
                continue;
            }

            // Genero K random centroidi
            std::vector<Punto> centroidi(k);
            std::vector<int> v(numPunti);
            std::iota(v.begin(), v.end(), 0);

            std::shuffle(v.begin(), v.end(), std::mt19937(std::random_device()()));

            for (int i = 0; i < k; i++) {
                centroidi[i] = ds[v[i]];
                centroidi[i].cluster_id = i;
            }

            std::vector<Punto> output_centroidi;
            std::vector<Punto> output_ds;

            // SEQUENTIAL Kmeans
            auto start = std::chrono::high_resolution_clock::now();
            std::tie(output_ds, output_centroidi) = sequential_kmeans(ds, centroidi, k);
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            tempiSeq.push_back(elapsed.count());

            // GPU Kmeans
            auto dim_dataset = numPunti * dimPunti * sizeof(double);
            auto dim_centroidi = k * dimPunti * sizeof(double);

            double* hostDataset = (double*)malloc(dim_dataset);
            double* hostCentroidi = (double*)malloc(dim_centroidi);

            double* deviceDataset, * deviceCentroidi;

            for (auto i = 0; i < numPunti; i++) {
                for (auto j = 0; j < dimPunti; j++) {
                    hostDataset[i * dimPunti + j] = ds[i].dimensioni[j];
                }
            }

            for (auto i = 0; i < k; i++) {
                for (auto j = 0; j < dimPunti; j++) {
                    hostCentroidi[i * dimPunti + j] = centroidi[i].dimensioni[j];
                }
            }

            CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceDataset, dim_dataset));
            CUDA_CHECK_RETURN(cudaMemcpy(deviceDataset, hostDataset, dim_dataset, cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceCentroidi, dim_centroidi));
            CUDA_CHECK_RETURN(cudaMemcpy(deviceCentroidi, hostCentroidi, dim_centroidi, cudaMemcpyHostToDevice));

            short* hostAssegnamento = (short*)malloc(numPunti * sizeof(short));

            start = std::chrono::high_resolution_clock::now();
            std::tie(deviceCentroidi, hostAssegnamento) = cuda_KMeans(deviceDataset, deviceCentroidi, numPunti, k, dimPunti);
            finish = std::chrono::high_resolution_clock::now();
            elapsed = finish - start;
            tempiCuda.push_back(elapsed.count());

            CUDA_CHECK_RETURN(cudaMemcpy(hostCentroidi, deviceCentroidi, dim_centroidi, cudaMemcpyDeviceToHost));

            CUDA_CHECK_RETURN(cudaFree(deviceDataset));
            CUDA_CHECK_RETURN(cudaFree(deviceCentroidi));

            free(hostDataset);
            free(hostCentroidi);
            free(hostAssegnamento);
        }

        // Salvo i risultati medi
        fileNamesVec.push_back(filename);
        tempiMediSeq.push_back(calcolaTempoMedio(tempiSeq));
        tempiMediCuda.push_back(calcolaTempoMedio(tempiCuda));
    }

    // Stampo i risultati finali come tabella
    std::cout << std::setw(15) << "File" << std::setw(20) << "Tempo medio Seq (s)" << std::setw(20) << "Tempo medio CUDA (s)" << "\n";
    std::cout << std::string(55, '-') << "\n";
    for (size_t i = 0; i < fileNamesVec.size(); ++i) {
        std::cout << std::setw(15) << fileNamesVec[i]
            << std::setw(20) << tempiMediSeq[i]
            << std::setw(20) << tempiMediCuda[i] << "\n";
    }

    return 0;
}