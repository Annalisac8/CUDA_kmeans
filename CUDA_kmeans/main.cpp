
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

/*int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}*/


#include <iostream>



int main(int argc, char* argv[]) {

    std::string riga;
    double valore;
    std::vector<Punto> ds;

    /*
        if (argc != 3) {
            std::cerr << "usage: k_means <data-file> <k>" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    */


    //std::ifstream dataset_file(argv[1]);

    // Richiedi il nome del file all'utente
    std::string filename;
    std::cout << "Inserisci il nome del file (ad esempio, 1000x10.txt): ";
    std::cin >> filename;

    // Costruisci il percorso completo del dataset
    std::string ds_path = "ds/" + filename;

    // Apri il file del dataset
    std::ifstream dataset_file(ds_path);
    if (!dataset_file) {
        std::cerr << "Impossibile aprire il file: " << ds_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    /*
    std::string ds_path = "../ds/1000x10";
    std::ifstream dataset_file(ds_path);
    if (!dataset_file) {
        //std::cerr << "Could not open file: " << argv[1] << std::endl;
        std::cerr << "Could not open file: " << ds_path << std::endl;
        std::exit(EXIT_FAILURE);
    }
     */


    if (dataset_file.is_open()) {
        while (getline(dataset_file, riga)) {
            std::istringstream iss(riga);
            Punto punto;
            while (iss >> valore) {
                punto.dimensioni.push_back(valore);
            }
            ds.push_back(punto);
        }
        dataset_file.close();
    }

    //Number of dataset points
    auto numPunti = ds.size();
    //Points dimension
    auto dimPunti = ds[0].dimensioni.size();
    //Get cluster number from input

    // Richiedi il numero di cluster (k) all'utente
    int k;
    std::cout << "Inserisci il numero di cluster (k): ";
    std::cin >> k;

    // Verifica che il valore di k sia valido
    if (k <= 0 || k > numPunti) {
        std::cerr << "Numero di cluster non valido: " << k << std::endl;
        exit(EXIT_FAILURE);
    }

    /*
    const auto k = std::strtol(argv[2], nullptr, 0);
    if (k == 0) {
        std::cerr << "cluster non ottenuti: " << argv[2] << std::endl;
        exit(EXIT_FAILURE);
    }
     */

     //Generate K random centroids
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


    /*
        //PRINT INITIAL CENTROIDS
        //Print centroids
        std::cout << "PRINT INITIAL CENTROIDS after then dataset is load \n";
        for (int i=0; i < k; i++){
            for(int j=0; j < dimPoint; j++){
                std::cout << centroids[i].dimensions[j] << " ";
            }
            std::cout << std::endl;
        }
    */

    //SEQUENTIAL Kmeans

        //CHRONO START
    auto start = std::chrono::high_resolution_clock::now();

    std::tie(output_ds, output_centroidi) = sequential_kmeans(ds, centroidi, k);


    //CHRONO END
    auto finish = std::chrono::high_resolution_clock::now();
    //CHRONO calcolo tempo e print
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "ENLAPSED TIME SEQUENZIALE  : " << elapsed.count() << " s\n \n";
}