#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern "C" {
void caract_GPU();

void caract_GPU(){
    int ngpus = 0;

    // check how many devices are available
    
    cudaGetDeviceCount(&ngpus);
    printf("%d cuda devices found on the node\n", ngpus);
    for (int n=0; n<ngpus; n++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, n);
      printf("Device nb %i, name %i \n", n, prop.name);
      printf("Total memory: %i \n", prop.totalGlobalMem);
      printf("Shared memory per block: %i \n", prop.sharedMemPerBlock);
    }
}


}
