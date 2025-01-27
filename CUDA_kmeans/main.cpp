
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
        //<< std::setw(20) << "Tempo Seq (s)"
        << std::setw(20) << "Tempo Parallelo (s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (const auto& risultato : risultati) {
        std::cout << std::setw(20) << std::get<0>(risultato) // Nome del file
           // << std::setw(20) << std::fixed << std::setprecision(6) << std::get<1>(risultato) // Tempo sequenziale
            << std::setw(20) << std::fixed << std::setprecision(6) << std::get<2>(risultato) // Tempo parallelo
            << std::endl;
    }
}

void stampaPunti(const std::vector<Punto>& ds) {
    std::cout << "Punti e cluster assegnati:\n";
    for (const auto& punto : ds) {
        std::cout << "(";
        for (size_t j = 0; j < punto.dimensioni.size(); ++j) {
            std::cout << punto.dimensioni[j];
            if (j < punto.dimensioni.size() - 1) std::cout << ", ";
        }
        std::cout << ") -> Cluster " << punto.cluster_id << "\n";
    }
}

int main(int argc, char* argv[]) {

    std::vector<std::string> filenames = {  "ds.txt",
                                            //"a1.txt",
                                            /*
                                            "1000x10.txt",
                                            "1000x100.txt",
                                            "1000x1000.txt",
                                            "10000x10.txt",
                                            "10000x100.txt",
                                            "10000x1000.txt",
                                            "100000x10.txt",
                                            "100000x100.txt",
                                            "100000x1000.txt",
                                            */};
    const int numEsecuzioni = 1;

    std::vector<std::tuple<std::string, double, double>> risultati;

    for (const auto& filename : filenames) {
        std::string ds_path = "ds/" + filename;
        std::ifstream dataset_file(ds_path);
        if (!dataset_file) {
            std::cerr << "Impossibile aprire il file: " << filename << std::endl;
            return -1;
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

        int numPoints = dataset.size();
        int dimensions = dataset[0].dimensioni.size();
        int numCentroids = 10;

        if (filename == "ds.txt") { numCentroids = 3; }
        if (filename == "s1set.txt") { numCentroids = 15; }
        if (filename == "a1.txt") { numCentroids = 20; }
        if (filename == "a2.txt") { numCentroids = 35; }
        if (filename == "a3.txt") { numCentroids = 50; }
        if (filename == "letter.txt") { numCentroids = 26; }
        if (filename == "birch1.txt") { numCentroids = 100; }
        if (filename == "birch3.txt") { numCentroids = 100; }

        // Generazione dei centroidi iniziali
        std::vector<Punto> initialCentroids(numCentroids);
        std::vector<int> indices(numPoints);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
        //std::mt19937 rng(42); // Seed fisso
        //std::shuffle(indices.begin(), indices.end(), rng);
        for (int i = 0; i < numCentroids; i++) {
            initialCentroids[i] = dataset[indices[i]];
        }
        initialCentroids[0] = Punto(8.5, 7.5, 9.5);
        initialCentroids[1] = Punto(2.5, 3.0, 4.5);
        initialCentroids[2] = Punto(9.0, 2.5, 3.5);


        //std::cout << "Centroidi iniziali: ";
        //stampaCentroidi(initialCentroids);

        // Vettori per memorizzare i tempi delle esecuzioni
        std::vector<double> tempiSeq(numEsecuzioni);
        std::vector<double> tempiPar(numEsecuzioni);

        const int numEsecuzioni = 1;


        for (int iter = 0; iter < numEsecuzioni; iter++) {
            // **Esecuzione SEQUENZIALE**

            /*
            std::cout << "Inizio K-Means sequenziale...\n";
            auto startSeq = std::chrono::high_resolution_clock::now();
            auto [seqDataset, seqCentroids] = sequential_kmeans(dataset, initialCentroids, numCentroids);
            auto finishSeq = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsedSeq = finishSeq - startSeq;
            std::cout << "Tempo sequenziale: " << elapsedSeq.count() << " secondi.\n";
            tempiSeq[iter] = std::chrono::duration<double>(finishSeq - startSeq).count();

            std::cout << "Centroidi sequenziale:";
            stampaCentroidi(seqCentroids);

            */

            std::vector<float> h_points(numPoints * dimensions);
            std::vector<float> h_centroids(numCentroids * dimensions);

            for (int i = 0; i < numPoints; i++) {
                for (int d = 0; d < dimensions; d++) {
                    h_points[i * dimensions + d] = dataset[i].dimensioni[d];
                }
            }
            for (int c = 0; c < numCentroids; c++) {
                for (int d = 0; d < dimensions; d++) {
                    //h_centroids[c * dimensions + d] = dataset[c].dimensioni[d];
                    h_centroids[c * dimensions + d] = initialCentroids[c].dimensioni[d];
                }
            }
            // Verifica la dimensione di h_centroids
            //std::cout << "Dimensioni h_centroids: " << h_centroids.size()
            //    << ", atteso: " << (numCentroids * dimensions) << "\n";

            float* d_points;
            float* d_centroids;
            int* d_assignments;

            CUDA_CHECK(cudaMalloc(&d_points, numPoints * dimensions * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_centroids, numCentroids * dimensions * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_assignments, numPoints * sizeof(int)));

            CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), numPoints * dimensions * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), numCentroids * dimensions * sizeof(float), cudaMemcpyHostToDevice));

            std::vector<float> h_oldCentroids(numCentroids * dimensions);
            std::vector<float> h_currentCentroids(numCentroids * dimensions);

            std::cout << "Inizio K-Means su GPU...\n";

            auto start = std::chrono::high_resolution_clock::now();
            kmeans_cuda(d_points, d_centroids, d_assignments, numPoints, numCentroids, dimensions, 100, 0.01, h_oldCentroids, h_currentCentroids);

            CUDA_CHECK(cudaDeviceSynchronize()); // Assicuriamoci che i kernel siano completati
            auto finish = std::chrono::high_resolution_clock::now();

            //std::cout << "Copia dei centroidi dalla GPU...\n";
            if (d_centroids == nullptr) {
                //std::cerr << "Errore: puntatore d_centroids è nullo.\n";
                exit(-1);
            }
            CUDA_CHECK(cudaMemcpy(h_centroids.data(), d_centroids, numCentroids * dimensions * sizeof(float), cudaMemcpyDeviceToHost));
            //std::cout << "Copia completata con successo.\n";

            // Stampa dei dati copiati
            /*
            std::cout << "--- Verifica dei dati copiati dalla GPU ---\n";
            for (int c = 0; c < numCentroids; c++) {
                std::cout << "Centroide " << c << ": ";
                for (int d = 0; d < dimensions; d++) {
                    std::cout << h_centroids[c * dimensions + d] << " ";
                }
                std::cout << "\n";
            }
            */
            
            
            std::cout << "\n--- Centroidi calcolati con CUDA ---\n";           
            for (int c = 0; c < numCentroids; c++) {
                std::cout << "Centroide " << c << ": (";
                for (int d = 0; d < dimensions; d++) {
                    std::cout << h_centroids[c * dimensions + d];
                    if (d < dimensions - 1) std::cout << ", ";
                }
                std::cout << ")\n";
            }
            
            


            //std::cout << "Rilascio della memoria GPU nel main...\n";

            if (d_points) {
                //std::cout << "Rilascio d_points: " << d_points << "...\n";
                CUDA_CHECK(cudaFree(d_points));
                //std::cout << "Memoria d_points rilasciata.\n";
            }
            else {
                std::cerr << "Errore: Puntatore d_points è nullo!\n";
            }

            if (d_centroids) {
                //std::cout << "Rilascio d_centroids: " << d_centroids << "...\n";
                CUDA_CHECK(cudaFree(d_centroids));
                //std::cout << "Memoria d_centroids rilasciata.\n";
            }
            else {
                std::cerr << "Errore: Puntatore d_centroids è nullo!\n";
            }

            if (d_assignments) {
                //std::cout << "Rilascio d_assignments: " << d_assignments << "...\n";
                CUDA_CHECK(cudaFree(d_assignments));
                //std::cout << "Memoria d_assignments rilasciata.\n";
            }
            else {
                std::cerr << "Errore: Puntatore d_assignments è nullo!\n";
            }


            std::chrono::duration<double> elapsedCuda = finish - start;
            tempiPar[iter] = std::chrono::duration<double>(finish - start).count();

            std::cout << "Tempo CUDA: " << elapsedCuda.count() << " secondi.\n";

            /// Calcola la media dei tempi
            //double mediaSeq = std::accumulate(tempiSeq.begin(), tempiSeq.end(), 0.0) / numEsecuzioni;
            double mediaPar = std::accumulate(tempiPar.begin(), tempiPar.end(), 0.0) / numEsecuzioni;

            // Aggiungi risultati al vettore
            risultati.push_back(std::make_tuple(filename, 0, mediaPar));
            std::cout << "--- ********************************************************** ---\n";

            // Confronto tempi
            //std::cout << "--- Confronto tempi ---\n";
            //std::cout << "Tempo sequenziale: " << elapsedSeq.count() << " secondi\n";
            //std::cout << "Tempo CUDA: " << elapsedCuda.count() << " secondi\n";

        }

    }
    // Stampa la tabella
    stampaTabella(risultati);

    std::cout << "Fine del programma, tutti i processi completati con successo.\n";
    return 0;
}