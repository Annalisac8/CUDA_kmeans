#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK_RETURN(valore) ControllaErroreCudaAux(__FILE__, __LINE__, #valore, valore)
void ControllaErroreCudaAux(const char* file, unsigned line, const char* statement, cudaError_t err);

std::tuple<double*, short*> CUDA_kmeans(double* datasetDispositivo, double* centroidiDispositivo, const int numeroPunti, const short k, const short dimensionePunti);