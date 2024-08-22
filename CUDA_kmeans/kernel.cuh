#define Controllo_CUDA_Return(valore) ControllaErroreCuda(__FILE__, __LINE__, #valore, valore)
void ControllaErroreCuda(const char* file, unsigned linea, const char* istruzione, cudaError_t errore);

std::tuple<double*, short*> CUDA_kmeans(double* datasetDispositivo, double* centroidiDispositivo, const int numeroPunti, const short k, const short dimensionePunti);