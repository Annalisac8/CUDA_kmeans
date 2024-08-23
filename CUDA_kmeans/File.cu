#include <stdio.h>
#include <cuda_runtime.h>

__global__ void testArch() {
#ifdef __CUDA_ARCH__
    printf("CUDA Architecture: %d\n", __CUDA_ARCH__);
#else
    printf("__CUDA_ARCH__ is not defined!\n");
#endif
}

int main() {
    testArch << <1, 1 >> > ();
    cudaDeviceSynchronize();
    return 0;
}