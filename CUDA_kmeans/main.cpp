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
#include "kernel.cuh"
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA errore in " << __FILE__ << " linea " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void stampaCentroidi(const std::vector<Punto>& centroidi) {
    std::cout << "Centroidi:\n";
    for (size_t i = 0; i < centroidi.size(); ++i) {
        std::cout << "Centroide " << i << ": (";
        for (size_t j = 0; j < centroidi[i].dimensioni.size(); ++j) {
            std::cout << centroidi[i].dimensioni[j];
            if (j < centroidi[i].dimensioni.size() - 1) std::cout << ", ";
        }
        std::cout << ")\n";
    }
}

void stampaTabella(const std::vector<std::tuple<std::string, double, double>>& risultati) {
    std::cout << std::setw(20) << "File"
        << std::setw(20) << "Tempo Seq (s)"
        << std::setw(20) << "Tempo Parallelo (s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (const auto& risultato : risultati) {
        std::cout << std::setw(20) << std::get<0>(risultato) // Nome del file
            << std::setw(20) << std::fixed << std::setprecision(6) << std::get<1>(risultato) // Tempo sequenziale
            << std::setw(20) << std::fixed << std::setprecision(6) << std::get<2>(risultato) // Tempo parallelo
            << std::endl;
    }
}


int main(int argc, char* argv[]) {

    std::vector<std::string> nomi_file= { 
       //"test.txt",
       "ds.txt",
        
        /*
        "s1set.txt",  
        
        "a1.txt",
        "a2.txt",
        "a3.txt",
        
        
        "letter.txt",
        */
        //"birch1.txt",
        
        
        /*
        "1000x10.txt",
        "1000x100.txt",
        "10000x10.txt",
        "10000x100.txt",
        */
        
        //"100000x10.txt",
        //"100000x100.txt"
        
        
        };
    int numEsecuzioni = 1;

    std::vector<std::tuple<std::string, double, double>> risultati;

    //************** Prelevo datasets da files **************************
    for (const auto& file : nomi_file) {
        std::ifstream dataset_file("ds/" + file);
        if (!dataset_file) {
            std::cerr << "Impossibile aprire il file: " << file << std::endl;
            continue;
        }

        std::vector<Punto> dataset;
        std::string riga;
        double valore;
        while (std::getline(dataset_file, riga)) {
            std::istringstream iss(riga);
            Punto punto;
            while (iss >> valore) {
                punto.dimensioni.push_back(valore);
            }
            dataset.push_back(punto);
        }
        dataset_file.close();



        //*********** Definisco numero centroidi da cercare ******************
        int numCentroidi = 10;
        if (file == "test.txt") { numCentroidi = 3; }
        if (file == "ds.txt") { numCentroidi = 3; }
        if (file == "s1set.txt") { numCentroidi = 15; }
        if (file == "a1.txt") { numCentroidi = 20; }
        if (file == "a2.txt") { numCentroidi = 35; }
        if (file == "a3.txt") { numCentroidi = 50; }
        if (file == "letter.txt") { numCentroidi = 26; }
        if (file == "birch1.txt") { numCentroidi = 100; }
        if (file == "birch3.txt") { numCentroidi = 100; }

        //inizializzo e definisco numero punti, dimensioni. definisco vettori per inizializzazione centroidi
        int numPunti = dataset.size();
        int dim = dataset[0].dimensioni.size();
        std::vector<Punto> centroidiIniziali(numCentroidi);
        std::vector<int> indice(numPunti);

        //definisco vettori per salvataggio tempi di esecuzione
        std::vector<double> tempiSeq(numEsecuzioni);
        std::vector<double> tempiPar(numEsecuzioni);


        //************************ Esecuzioni *****************************************
        for(int iter = 0; iter < numEsecuzioni; iter++) {

          
            //**********************************inizializzazione centroidi************************************************

            std::iota(indice.begin(), indice.end(), 0);
            std::shuffle(indice.begin(), indice.end(), std::mt19937(std::random_device()()));
            for (int i = 0; i < numCentroidi; i++) {
                centroidiIniziali[i] = dataset[indice[i]];
            }
            if (file == "ds.txt") {
                centroidiIniziali[0] = Punto(1.5, 1.8);
                centroidiIniziali[1] = Punto(8.0, 8.0);
                centroidiIniziali[2] = Punto(10.0, 2.0);

            }
            
            //std::cout << "Inizializzazione centroidi:\n";
            //stampaCentroidi(centroidiIniziali);

            // **************************************SEQUENZIALE*********************************************************
            
            
            std::cout << "Inizio K-Means sequenziale...\n";
            auto startSeq = std::chrono::high_resolution_clock::now();
            auto [seqDataset, seqCentroidi] = sequential_kmeans(dataset, centroidiIniziali, numCentroidi, 1000, 0.001);
            auto finishSeq = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> enlapsedSeq = finishSeq - startSeq;
            std::cout << "Tempo sequenziale: " << enlapsedSeq.count() << " secondi.\n";
            tempiSeq[iter] = std::chrono::duration<double>(finishSeq - startSeq).count();

            std::cout << "Centroidi sequenziale:";
            stampaCentroidi(seqCentroidi);

            

            //*************************************************CUDA******************************************************

            
            std::vector<double> h_punti(numPunti * dim);
            
            for (int i = 0; i < numPunti; i++) {
                for (int d = 0; d < dim; d++) {
                    h_punti[i * dim + d] = dataset[i].dimensioni[d];
                }
            }

            std::vector<double> h_centroidi(numCentroidi* dim);
            for (int c = 0; c < numCentroidi; c++) {
                for (int d = 0; d < dim; d++) {
                    h_centroidi[c * dim + d] = centroidiIniziali[c].dimensioni[d];
                }
            }

            double* d_punti;
            double* d_centroidi;
            int* d_assegnamenti;
            CUDA_CHECK(cudaMalloc(&d_punti, numPunti * dim * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_centroidi, numCentroidi * dim * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_assegnamenti, numPunti * sizeof(int)));

            CUDA_CHECK(cudaMemcpy(d_punti, h_punti.data(), numPunti * dim * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_centroidi, h_centroidi.data(), numCentroidi * dim * sizeof(double), cudaMemcpyHostToDevice));

            std::vector<double> h_centroidiPrecedenti(numCentroidi * dim);
            std::vector<double> h_centroidiCorrenti(numCentroidi * dim);
            std::vector<int> h_assegnamenti(numPunti);

            std::cout << "Esecuzione kmeans CUDA:\n";
            auto startPar = std::chrono::high_resolution_clock::now();
            kmeans_cuda(d_punti, d_centroidi, d_assegnamenti, numPunti, numCentroidi, dim, 1000, 0.001, h_centroidiPrecedenti, h_centroidiCorrenti);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto finishPar = std::chrono::high_resolution_clock::now();
            
            CUDA_CHECK(cudaMemcpy(h_centroidi.data(), d_centroidi, numCentroidi * dim * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_assegnamenti.data(), d_assegnamenti, numPunti * sizeof(int), cudaMemcpyDeviceToHost));


            CUDA_CHECK(cudaFree(d_punti));
            CUDA_CHECK(cudaFree(d_centroidi));
            CUDA_CHECK(cudaFree(d_assegnamenti));


            /*
            std::vector<int> contatoreCluster(numCentroidi, 0);
            for (int i = 0; i < numPunti; i++) {
                contatoreCluster[h_assignments[i]]++;
            }
           
            std::cout << "\n--- Distribuzione punti nei cluster ---\n";
            for (int c = 0; c < numCentroidi; c++) {
                std::cout << "Cluster " << c << ": " << contatoreCluster[c] << " punti\n";
            }
            */
            
            
              std::cout << "\n--- Centroidi calcolati con CUDA ---\n";
              for (int c = 0; c < numCentroidi; c++) {
                  std::cout << "Centroide " << c << ": (";
                  for (int d = 0; d < dim; d++) {
                      std::cout << h_centroidi[c * dim + d];
                      if (d < dim - 1) std::cout << ", ";
                  }
                  std::cout << ")\n";
              }
              

            std::chrono::duration<double> elapsedCuda = finishPar - startPar;
            std::cout << "Tempo CUDA: " << elapsedCuda.count() << " secondi.\n";
            tempiPar[iter] = std::chrono::duration<double>(finishPar - startPar).count();
       

        }
        /// Calcola la media dei tempi
        double mediaSeq = std::accumulate(tempiSeq.begin(), tempiSeq.end(), 0.0) / numEsecuzioni;
        double mediaPar = std::accumulate(tempiPar.begin(), tempiPar.end(), 0.0) / numEsecuzioni;

        // Aggiungi risultati al vettore
        risultati.push_back(std::make_tuple(file, mediaSeq, mediaPar));
    }

    stampaTabella(risultati);
    return 0;
}

