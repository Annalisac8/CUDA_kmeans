#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"

#include <stdio.h>

#include <cmath>
#include <iostream>

// variabili globali utilizzate nei kernel CUDA
// GLOBALI = risiedono nella memoria costante della GPU
// accessibili a tutti i thread efficentemente 
__constant__ short constNumCluster;
__constant__ int constNumPunti;
__constant__ short constDimPunti;

//Funzione di controllo degli errori CUDA
void ControllaErroreCuda(const char* file, unsigned linea, const char* istruzione, cudaError_t errore) {
    if (errore == cudaSuccess) {
        return;
    }
    std::cerr << istruzione << " ha restituito " << cudaGetErrorString(errore) << "(" << errore << ") in " << file << ":" << linea << std::endl;
    exit(1);
}
