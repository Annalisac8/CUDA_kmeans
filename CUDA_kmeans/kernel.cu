#include "kernel.cuh"
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



__global__ void testAtomicCAS(double* address) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old_value = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old_value;
        old_value = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(__longlong_as_double(assumed) + 1.0));
    } while (assumed != old_value);
}



__constant__ short costanteK;
__constant__ int costanteNumPunti;
__constant__ short costanteDimPunto;

void ControllaErroreCudaAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

void stampa_dispositivo(double* dispositivo, int righe, int colonne) {
    double* host;
    host = (double*)malloc(righe * colonne * sizeof(double));
    cudaMemcpy(host, dispositivo, righe * colonne * sizeof(double), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < righe; i++) {
        for (auto j = 0; j < colonne; j++) {
            std::cout << "- " << host[i * colonne + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}

void stampa_dispositivo(short* dispositivo, int righe, int colonne) {
    short* host;
    host = (short*)malloc(righe * colonne * sizeof(short));
    cudaMemcpy(host, dispositivo, righe * colonne * sizeof(short), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < righe; i++) {
        for (auto j = 0; j < colonne; j++) {
            std::cout << "- " << host[i * colonne + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}

void stampa_dispositivo(int* dispositivo, int righe, int colonne) {
    int* host;
    host = (int*)malloc(righe * colonne * sizeof(int));
    cudaMemcpy(host, dispositivo, righe * colonne * sizeof(int), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < righe; i++) {
        for (auto j = 0; j < colonne; j++) {
            std::cout << "- " << host[i * colonne + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}

//INIZIALIZZA ASSEGNAZIONE CENTROIDE A ZERO PER TUTTI I DATI DEI PUNTI
//Assegno ogni punto al cluster -1
__global__
void inizializza_assegnazione(short* assegnazioneDispositivo) {
    unsigned int idThread = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (idThread < costanteNumPunti) {
        assegnazioneDispositivo[idThread] = -1;
    }
}

__device__ double sommaAtomicaDouble(double* indirizzo, double valore) {
    auto* indirizzoComeULL = (unsigned long long int*) indirizzo;
    unsigned long long int vecchio = *indirizzoComeULL, presunto;
    do {
        presunto = vecchio;
        vecchio = atomicCAS(indirizzoComeULL, presunto, __double_as_longlong(valore + __longlong_as_double((long long int)presunto)));
    } while (presunto != vecchio);
    return __longlong_as_double((long long int)vecchio);
}

__host__
bool controllaAssegnazioneUguale(const short* vecchiaAssegnazioneHost, const short* assegnazioneHost, const int numPunti) {
    for (auto i = 0; i < numPunti; i++) {
        if (vecchiaAssegnazioneHost[i] != assegnazioneHost[i]) {
            return false;
        }
    }
    return true;
}



__global__
void calcola_distanze(const double* datasetDispositivo, const double* centroidiDispositivo, double* distanzeDispositivo) {
    double distanza = 0;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int riga = blockIdx.y * blockDim.y + threadIdx.y;
    if (riga < costanteNumPunti && col < costanteK) {
        for (int i = 0; i < costanteDimPunto; i++) {
            distanza += pow(datasetDispositivo[riga * costanteDimPunto + i] - centroidiDispositivo[col * costanteDimPunto + i], 2);
        }
        distanzeDispositivo[riga * costanteK + col] = sqrt(distanza);
    }
}

__global__
void assegnazione_punti(const double* distanzeDispositivo, short* assegnazioneDispositivo) {
    unsigned int idThread = (blockDim.x * blockIdx.x) + threadIdx.x;
    double minimo = INFINITY;
    short etichettaCluster;
    double distanza;
    if (idThread < costanteNumPunti) {
        for (auto i = 0; i < costanteK; i++) {
            distanza = distanzeDispositivo[idThread * costanteK + i];
            if (distanza < minimo) {
                minimo = distanza;
                etichettaCluster = i;
            }
        }
        assegnazioneDispositivo[idThread] = etichettaCluster;
    }
}


__global__
void inizializza_centroidi(double* centroidiDispositivo) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int riga = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < costanteDimPunto && riga < costanteK) {
        centroidiDispositivo[riga * costanteDimPunto + col] = 0;
    }
}

//Calcolo della somma originale con griglia 2D (meglio con dataset con molte dimensioni)
__global__
void calcola_somma(const double* datasetDispositivo, double* centroidiDispositivo, const short* assegnazioneDispositivo, int* contatoreDispositivo) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int riga = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < costanteDimPunto && riga < costanteNumPunti) {
        short idCluster = assegnazioneDispositivo[riga];
        sommaAtomicaDouble(&centroidiDispositivo[idCluster * costanteDimPunto + col], datasetDispositivo[riga * costanteDimPunto + col]);
        atomicAdd(&contatoreDispositivo[idCluster], 1);
    }
}



//Calcolo della somma con griglia 1D e iterazione sulle dimensioni del punto
__global__
void calcola_somma2(const double* datasetDispositivo, double* centroidiDispositivo, const short* assegnazioneDispositivo, int* contatoreDispositivo) {
    unsigned int riga = blockIdx.x * blockDim.x + threadIdx.x;
    if (riga < costanteNumPunti) {
        short idCluster = assegnazioneDispositivo[riga];
        for (auto i = 0; i < costanteDimPunto; i++) {
            sommaAtomicaDouble(&centroidiDispositivo[idCluster * costanteDimPunto + i], datasetDispositivo[riga * costanteDimPunto + i]);
        }
        atomicAdd(&contatoreDispositivo[idCluster], 1);
    }
}


//Aggiornamento centroidi con griglia 2D (meglio con dataset con molte dimensioni)
__global__
void aggiorna_centroidi(double* centroidiDispositivo, const int* contatoreDispositivo) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int riga = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < costanteDimPunto && riga < costanteK) {
        centroidiDispositivo[riga * costanteDimPunto + col] = centroidiDispositivo[riga * costanteDimPunto + col] / (double(contatoreDispositivo[riga]) / costanteDimPunto);
    }
}


//Aggiornamento centroidi con griglia 1D (non c'è bisogno di dividere il contatore per le dimensioni del punto)
__global__
void aggiorna_centroidi2(double* centroidiDispositivo, const int* contatoreDispositivo) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int riga = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < costanteDimPunto && riga < costanteK) {
        centroidiDispositivo[riga * costanteDimPunto + col] /= contatoreDispositivo[riga];
    }
}






struct KMeansResult {
    double* centroidiDispositivo;
    short* assegnazioneHost;
};


__host__ 
std::tuple<double*, short*> cuda_KMeans(double* datasetDispositivo, double* centroidiDispositivo, const int numPunti, const short k, const short dimPunto) {
    dim3 dimBloccoDistanza(2, 512, 1);
    dim3 dimGrigliaDistanza(ceil(k / 2.0), ceil(numPunti / 512.0), 1);

    dim3 dimBloccoInizializza(16, 16, 1);
    dim3 dimGrigliaInizializza(ceil(dimPunto / 16.0), ceil(k / 16.0), 1);

    dim3 dimBloccoCalcolaSomma(2, 512, 1);
    dim3 dimGrigliaCalcolaSomma(ceil(dimPunto / 2.0), ceil(numPunti / 512.0), 1);

    dim3 dimBloccoAggiornaCentroidi(16, 16, 1);
    dim3 dimGrigliaAggiornaCentroidi(ceil(dimPunto / 16.0), ceil(k / 16.0), 1);

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(costanteK, &k, sizeof(short)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(costanteNumPunti, &numPunti, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(costanteDimPunto, &dimPunto, sizeof(short)));

    bool convergenza = false;

    short* vecchiaAssegnazioneHost;
    vecchiaAssegnazioneHost = (short*)malloc(numPunti * sizeof(short));

    short* assegnazioneHost;
    assegnazioneHost = (short*)malloc(numPunti * sizeof(short));

    short* assegnazioneDispositivo;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&assegnazioneDispositivo, numPunti * sizeof(short)));
    double* distanzeDispositivo;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&distanzeDispositivo, numPunti * k * sizeof(double)));
    int* contatoreDispositivo;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&contatoreDispositivo, k * sizeof(int)));

    while (!convergenza) {
        //ASSEGNAZIONE
        //Trova il centroide più vicino e assegna il punto a quel cluster
        calcola_distanze <<<dimGrigliaDistanza, dimBloccoDistanza >>> (datasetDispositivo, centroidiDispositivo, distanzeDispositivo);
        cudaDeviceSynchronize();
        assegnazione_punti <<<ceil(numPunti / 1024.0), 1024 >>> (distanzeDispositivo, assegnazioneDispositivo);
        cudaDeviceSynchronize();

        //AGGIORNAMENTO CENTROIDI
        //Inizializza i centroidi a 0 e imposta il contatore a 0 (per calcolare le medie)
        inizializza_centroidi <<<dimGrigliaInizializza, dimBloccoInizializza >>> (centroidiDispositivo);

        CUDA_CHECK_RETURN(cudaMemset(contatoreDispositivo, 0, k * sizeof(int)));
        cudaDeviceSynchronize();

        //Calcola tutte le somme per i centroidi
        calcola_somma <<<dimGrigliaCalcolaSomma, dimBloccoCalcolaSomma >>> (datasetDispositivo, centroidiDispositivo, assegnazioneDispositivo, contatoreDispositivo);

        cudaDeviceSynchronize();

        //Calcola la media: divisione per il contatore
        aggiorna_centroidi <<<dimGrigliaAggiornaCentroidi, dimBloccoAggiornaCentroidi >>> (centroidiDispositivo, contatoreDispositivo);

        cudaDeviceSynchronize();

        CUDA_CHECK_RETURN(cudaMemcpy(assegnazioneHost, assegnazioneDispositivo, numPunti * sizeof(short), cudaMemcpyDeviceToHost));

        if (controllaAssegnazioneUguale(vecchiaAssegnazioneHost, assegnazioneHost, numPunti)) {
            convergenza = true;
        }
        else {
            CUDA_CHECK_RETURN(cudaMemcpy(vecchiaAssegnazioneHost, assegnazioneDispositivo, numPunti * sizeof(short), cudaMemcpyDeviceToHost));
        }
    }

    return{ centroidiDispositivo, assegnazioneHost };
}




 




int test() {


    std::string riga;
    double valore;
    std::vector<Punto> ds;


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

    //Numero putni ds
    auto numPunti = ds.size();
    //dimensione punti
    auto dimPunti = ds[0].dimensioni.size();
    

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
        
        //Print centroidi
        std::cout << "PRINT CENTROIDI dataset load \n";
        for (int i=0; i < k; i++){
            for(int j=0; j < dimPunti; j++){
                std::cout << centroidi[i].dimensioni[j] << " ";
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

    /*
    //Print centroidi
    std::cout << "PRINT CENTROIDI \n";
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dimPunti; j++) {
            std::cout << centroidi[i].dimensioni[j] << " ";
        }
        std::cout << std::endl;
    }
    */


    // Calcola la dimensione del dataset e dei centroidi in base al numero di punti, 
// dimensioni dei punti e numero di centroidi (k). 
    auto dim_dataset = numPunti * dimPunti * sizeof(double); // Dimensione totale del dataset in byte
    auto dim_centroidi = k * dimPunti * sizeof(double); // Dimensione totale dei centroidi in byte

    // Alloca memoria sull'host (CPU) per il dataset e i centroidi
    double* hostDataset;
    hostDataset = (double*)malloc(dim_dataset);
    double* hostCentroidi;
    hostCentroidi = (double*)malloc(dim_centroidi);

    double* deviceDataset, * deviceCentroidi; // Dichiarazione dei puntatori per i dati sul device (GPU)

    // Sposta i dati dal vettore alla matrice semplice (hostDataset) per il dataset
    for (auto i = 0; i < numPunti; i++) {
        for (auto j = 0; j < dimPunti; j++) {
            hostDataset[i * dimPunti + j] = ds[i].dimensioni[j];
        }
    }

    // Sposta i dati dal vettore alla matrice semplice (hostCentroidi) per i centroidi
    for (auto i = 0; i < k; i++) {
        for (auto j = 0; j < dimPunti; j++) {
            hostCentroidi[i * dimPunti + j] = centroidi[i].dimensioni[j];
        }
    }

    // ALLOCA E COPIA IL DATASET E I CENTROIDI SULLA GPU

    // Alloca memoria sulla GPU per il dataset
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceDataset, dim_dataset));
    // Copia il dataset dall'host (CPU) al device (GPU)
    CUDA_CHECK_RETURN(cudaMemcpy(deviceDataset, hostDataset, dim_dataset, cudaMemcpyHostToDevice));
    // Alloca memoria sulla GPU per i centroidi
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceCentroidi, dim_centroidi));
    // Copia i centroidi dall'host (CPU) al device (GPU)
    CUDA_CHECK_RETURN(cudaMemcpy(deviceCentroidi, hostCentroidi, dim_centroidi, cudaMemcpyHostToDevice));

    // Alloca memoria sull'host per l'assegnamento dei punti ai centroidi
    short* hostAssegnamento;
    hostAssegnamento = (short*)malloc(numPunti * sizeof(short));

    // INIZIA LA MISURAZIONE DEL TEMPO
    start = std::chrono::high_resolution_clock::now();

    // LANCIA IL KERNEL CUDA PER ESEGUIRE L'ALGORITMO K-MEANS
    std::tie(deviceCentroidi, hostAssegnamento) = cuda_KMeans(deviceDataset, deviceCentroidi, numPunti, k, dimPunti);

    // TERMINA LA MISURAZIONE DEL TEMPO
    finish = std::chrono::high_resolution_clock::now();
    // CALCOLA E STAMPA IL TEMPO TRASCORSO
    elapsed = finish - start;
    std::cout << "CUDA Elapsed time: " << elapsed.count() << " s\n \n";

    // COPIA I CENTROIDI FINALMENTE CALCOLATI DALLA GPU ALL'HOST
    CUDA_CHECK_RETURN(cudaMemcpy(hostCentroidi, deviceCentroidi, dim_centroidi, cudaMemcpyDeviceToHost));

    /*
        // STAMPA I CENTROIDI FINALI 
        std::cout << "STAMPA I CENTROIDI FINALI: \n";
        for(auto i = 0; i<k; i++){
            for(auto j = 0; j<dimPunti; j++){
                std::cout << hostCentroidi[i*dimPunti+j] << " ";
            }
            std::cout << "\n";
        }
    */

    // LIBERA LA MEMORIA ALLOCATA SULLA GPU
    CUDA_CHECK_RETURN(cudaFree(deviceDataset));
    CUDA_CHECK_RETURN(cudaFree(deviceCentroidi));

    // LIBERA LA MEMORIA ALLOCATA SULL'HOST
    free(hostDataset);
    free(hostCentroidi);
    free(hostAssegnamento);

    return 0;
}


