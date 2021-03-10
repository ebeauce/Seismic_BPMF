#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

extern "C" {

#include "libcu.h"

#define BLOCKSIZE 512
#define WARPSIZE 32
#define NCHUNKS 20
#define SATURATION_SCALE 7.0f

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void __global__ moveouts_kernel(int* moveouts, int *moveouts_mm, 
                                int *test_sources, int *station_indexes,
                                int n_test, 
                                int n_stations_whole_array, 
                                int n_stations_restricted_array) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int s, ss, max_moveout, min_moveout;
   float moveout = 0.;

   if(idx >= n_test) return;

   max_moveout=0;
   min_moveout=1000000; // shoud be MAX_INT

   for (s = 0; s < n_stations_restricted_array; s++) {
       // map from the whole stations array to the restricted array
       ss = station_indexes[test_sources[idx] * n_stations_restricted_array + s];
       moveout = moveouts[test_sources[idx] * n_stations_whole_array + ss];
       if (moveout > max_moveout) {
            max_moveout = moveout;
       }
       if (moveout < min_moveout) {
            min_moveout = moveout;
       }
   }
   moveouts_mm[idx * 2 + 0] = min_moveout;
   moveouts_mm[idx * 2 + 1] = max_moveout;
}

void __global__ stack_S_kernel(float*  tracesN,
                               float*  tracesE,
                               int*    moveouts, 
			                   int*    moveouts_minmax,
                               int*    station_indexes,
			                   int*    test_sources,
			                   float*  nw_response,
                               int*    biggest_idx, 
			                   int n_samples, 
                               int n_test, 
                               int n_stations_whole_array,
                               int n_stations_restricted_array) {

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float network_response_max, sum_beamform;
   int i, s, ss, source_index_whole_array, source_index_restricted_array, network_resp_idx_max, shift, moveout;
   float *tracesN_, *tracesE_; // local pointers

   if(idx >= n_samples){
      return;
   }
   
   network_response_max = 0.0;
   
   for (i = 0; i < n_test; i++) {
        source_index_whole_array      = test_sources[i] * n_stations_whole_array; // position on the moveouts vector
        source_index_restricted_array = test_sources[i] * n_stations_restricted_array; // position on the station indexes vector
	    
        sum_beamform = 0.0;
	    shift = idx - moveouts_minmax[i * 2 + 0]; // position on the time axis

        if (shift < 0) continue; // don't do anything before time 0
        // ----------------------------------------------------------
        // define the local pointers
	    tracesN_ = tracesN + shift;
        tracesE_ = tracesE + shift;
        // ----------------------------------------------------------

        for (s = 0; s < n_stations_restricted_array; s++) {
            // map from the restricted array (closest stations) to the whole array of stations
		    ss = station_indexes[source_index_restricted_array + s];
            moveout   = moveouts[source_index_whole_array + ss];
            if(shift + moveout < n_samples){
                // rotate the traces to get the transverse component and stack them
                sum_beamform += __ldg(&(tracesN_[ss * n_samples + moveout])) + \
                                __ldg(&(tracesE_[ss * n_samples + moveout]));
            }
        }

            if (sum_beamform > network_response_max) {
                network_response_max = sum_beamform;
                network_resp_idx_max = test_sources[i];
            }
   }
   nw_response[idx] = network_response_max;
   biggest_idx[idx] = network_resp_idx_max;
}

void __global__ stack_SP_kernel_prestacked(float*  prestacked_trace,
                                           int*    moveouts_P, 
                                           int*    moveouts_S, 
			                               int*    moveouts_minmax,
                                           int*    station_indexes,
			                               int*    test_sources,
			                               float*  nw_response,
                                           int*    biggest_idx, 
			                               int n_samples, 
			                               int n_samples_thread, 
                                           int n_test, 
                                           int n_stations_whole_array,
                                           int n_stations_restricted_array) {

   /* This function handles single-channel detection traces. When the user
   uses positive-valued detection traces, stacking the three components
   ahead of computing the network response optimizes the computation, without
   any loss of precision. However, we loose the possibility of using the P moveouts
   only on the vertical components and the S moveouts on the horizontal components,
   as it could be useful for relocation purposes. */

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float network_response_max, sum_beamform_SP;
   int i, s, ss, source_index_whole_array, source_index_restricted_array, network_resp_idx_max, shift, moveout_P, moveout_S;
   float trace_P, trace_S; // samples to stack

   if(idx >= n_samples_thread){
      return;
   }
   
   network_response_max = -INFINITY;
   
   for (i = 0; i < n_test; i++) {
	    source_index_whole_array      = test_sources[i] * n_stations_whole_array; // position on the moveouts vector
        source_index_restricted_array = test_sources[i] * n_stations_restricted_array; // position on the station indexes vector
    
        sum_beamform_SP = 0.0;

	    shift = idx - moveouts_minmax[i * 2 + 0]; // position on the time axis

        if (shift < 0) continue; // don't do anything before time 0

        for (s = 0; s < n_stations_restricted_array; s++) { 
            // map from the closest stations to the whole array of stations
		    ss =   station_indexes[source_index_restricted_array + s];
            moveout_P = moveouts_P[source_index_whole_array + ss];
            moveout_S = moveouts_S[source_index_whole_array + ss];
            if(shift + moveout_S < n_samples){
                // !!! shift + moveout can still be > n_samples 
                trace_S = __ldg(&(prestacked_trace[ss * n_samples + shift + moveout_S]));
                trace_P = __ldg(&(prestacked_trace[ss * n_samples + shift + moveout_P]));
                sum_beamform_SP += (trace_S + trace_P);
            }
        }

        if (sum_beamform_SP > network_response_max) {
            network_response_max = sum_beamform_SP;
            network_resp_idx_max = test_sources[i];
        }
   }
   nw_response[idx] = network_response_max;
   biggest_idx[idx] = network_resp_idx_max;
}

void __global__ stack_SP_kernel(float*  tracesH,
                                float*  tracesZ,
                                int*    moveouts_P, 
                                int*    moveouts_S, 
			                    int*    moveouts_minmax,
                                int*    station_indexes,
			                    int*    test_sources,
			                    float*  nw_response,
                                int*    biggest_idx, 
			                    int n_samples, 
			                    int n_samples_thread, 
                                int n_test, 
                                int n_stations_whole_array,
                                int n_stations_restricted_array) {

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float network_response_max, sum_beamform_SP, sum_beamform_S, sum_beamform_P;
   int i, s, ss, source_index_whole_array, source_index_restricted_array, network_resp_idx_max, shift, moveout_P, moveout_S;
   float traceH_, traceZ_; // samples to stack

   if(idx >= n_samples_thread){
      return;
   }
   
   network_response_max = -INFINITY;
   
   for (i = 0; i < n_test; i++) {
	    source_index_whole_array      = test_sources[i] * n_stations_whole_array; // position on the moveouts vector
        source_index_restricted_array = test_sources[i] * n_stations_restricted_array; // position on the station indexes vector
    
        sum_beamform_SP = 0.0;
        sum_beamform_S = 0.0;
        sum_beamform_P = 0.0;

	    shift = idx - moveouts_minmax[i * 2 + 0]; // position on the time axis

        if (shift < 0) continue; // don't do anything before time 0

        for (s = 0; s < n_stations_restricted_array; s++) { 
            // map from the closest stations to the whole array of stations
		    ss =   station_indexes[source_index_restricted_array + s];
            moveout_P = moveouts_P[source_index_whole_array + ss];
            moveout_S = moveouts_S[source_index_whole_array + ss];
            if(shift + moveout_S < n_samples){
                // !!! shift + moveout can still be > n_samples 
                traceH_ = __ldg(&(tracesH[ss * n_samples + shift + moveout_S]));
                traceZ_ = __ldg(&(tracesZ[ss * n_samples + shift + moveout_P]));
                sum_beamform_S += traceH_;
                sum_beamform_P += traceZ_;
            }
        }

        sum_beamform_SP = sum_beamform_S + sum_beamform_P;

        if (sum_beamform_SP > network_response_max) {
            network_response_max = sum_beamform_SP;
            network_resp_idx_max = test_sources[i];
        }
   }
   nw_response[idx] = network_response_max;
   biggest_idx[idx] = network_resp_idx_max;
}

void network_response(int*  test_points, 
                      float*  tracesN, 
                      float*  tracesE,
                      int*  moveouts, 
                      int* st_idx,
                      int n_test, 
                      int n_samples,
                      float*  network_response, 
                      int*  biggest_idx,
                      int n_stations_whole_array, 
                      int n_stations_restricted_array, 
                      int n_sources) {
    // cuda device vars
    int   *t_sources_d;
    float *traces_N_d;
    float *traces_E_d;
    int   *moveouts_d;
    int   *st_idx_d;
    float *nw_response_d;
    int   *biggest_idx_d;
    int   *moveouts_minmax_d;
    int ngpus = 0;
    int GRID_SIZE = 1024;

    cudaError_t cuda_result;

    // check how many devices are available
    
    cudaGetDeviceCount(&ngpus);
    printf("%d cuda devices found on the node\n", ngpus);
    /*for (int n=0; n<ngpus; n++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, n);
      printf("Device nb %i, name %i \n", n, prop.name);
      printf("Total memory: %i \n", prop.totalGlobalMem);
      printf("Shared memory per block: %i \n", prop.sharedMemPerBlock);
    }*/
    
    size_t free, total;
    printf("\n");
    cudaMemGetInfo(&free,&total);
    printf("%d KB free of total %d KB\n",free/1024,total/1024); 

    // allocate memory on device
    cuda_result = cudaMalloc((void**)&t_sources_d, n_test * sizeof(int));
    if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

    cuda_result = cudaMalloc((void**)&traces_N_d, n_samples * n_stations_whole_array * sizeof(float));
    if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

    cuda_result = cudaMalloc((void**)&traces_E_d, n_samples * n_stations_whole_array * sizeof(float));
    if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

    cuda_result = cudaMalloc((void**)&moveouts_d, n_sources * n_stations_restricted_array * sizeof(int));
    if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

    cuda_result = cudaMalloc((void**)&st_idx_d, n_sources * n_stations_restricted_array * sizeof(int));
    if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

    cuda_result = cudaMalloc((void**)&nw_response_d, n_samples * sizeof(float));
    if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

    cuda_result = cudaMalloc((void**)&biggest_idx_d, n_samples * sizeof(int));
    if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

    cuda_result = cudaMalloc((void**)&moveouts_minmax_d, 2 * n_test * sizeof(int));
    if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

    // transfer from CPU to GPU
    cuda_result = cudaMemcpy(t_sources_d, test_points, n_test * sizeof(int),
               cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

    cuda_result = cudaMemcpy(traces_N_d, tracesN, n_samples * n_stations_whole_array * sizeof(float),
               cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

    cuda_result = cudaMemcpy(traces_E_d, tracesE, n_samples * n_stations_whole_array * sizeof(float),
               cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

    cuda_result = cudaMemcpy(moveouts_d, moveouts, n_sources * n_stations_restricted_array * sizeof(int),
               cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

    cuda_result = cudaMemcpy(st_idx_d, st_idx, n_sources * n_stations_restricted_array * sizeof(int),
               cudaMemcpyHostToDevice);
    if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");
    
    cudaMemGetInfo(&free,&total);
    printf("%d KB free of total %d KB\n",free/1024,total/1024); 

    //cudaFuncSetCacheConfig("stack_kernel", cudaFuncCachePreferL1);

    moveouts_kernel<<<ceil(n_test/(float)GRID_SIZE),GRID_SIZE>>>(moveouts_d,
                                                                 moveouts_minmax_d,
                                                                 t_sources_d,
                                                                 st_idx_d,
                                                                 n_test,
                                                                 n_stations_whole_array,
                                                                 n_stations_restricted_array);

    //printf("Number of calls: %.2f \n", ceil(n_samples/(float)GRID_SIZE));
    stack_S_kernel<<<ceil(n_samples/(float)GRID_SIZE),GRID_SIZE>>>(traces_N_d,
                                                                   traces_E_d,
                                                                   moveouts_d,
                                                                   moveouts_minmax_d,
                                                                   st_idx_d,
                                                                   t_sources_d,
                                                                   nw_response_d,
                                                                   biggest_idx_d,
                                                                   n_samples,
                                                                   n_test,
                                                                   n_stations_whole_array,
                                                                   n_stations_restricted_array);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cuda_result = cudaMemcpy(network_response, nw_response_d,n_samples * sizeof(float),
                cudaMemcpyDeviceToHost);
    if (cuda_result != cudaSuccess) printf("Problem with the transfer of memory !\n");
    cuda_result = cudaMemcpy(biggest_idx, biggest_idx_d, n_samples * sizeof(int),
                cudaMemcpyDeviceToHost);
    if (cuda_result != cudaSuccess) printf("Problem with the transfer of memory !\n");

    cudaFree(t_sources_d);
    cudaFree(traces_N_d);
    cudaFree(traces_E_d);
    cudaFree(moveouts_d);
    cudaFree(st_idx_d);
    cudaFree(nw_response_d);
    cudaFree(biggest_idx_d);
    cudaFree(moveouts_minmax_d);
}

void network_response_SP_prestacked(int*  test_points, 
                                    float*  prestacked_traces, 
                                    int*  moveouts_P, 
                                    int*  moveouts_S, 
                                    int* st_idx,
                                    int n_test, 
                                    int n_samples,
                                    float*  network_response, 
                                    int*  biggest_idx,
                                    int n_stations_whole_array, 
                                    int n_stations_restricted_array, 
                                    int n_sources) {
    int nGPUs = 0;

    // check how many devices are available
    
    cudaGetDeviceCount(&nGPUs);
    omp_set_num_threads(nGPUs);

    printf("%d cuda devices found on the node\n", nGPUs);
    
    size_t free, total;
    printf("\n");
    cudaMemGetInfo(&free,&total);
    printf("%d KB free of total %d KB\n",free/1024,total/1024); 

    #pragma omp parallel shared(test_points, prestacked_traces, moveouts_P, moveouts_S, st_idx, \
                                network_response, biggest_idx)
    {

        int id = omp_get_thread_num();
        cudaSetDevice(id);
        dim3 BS(256);
        int n_samples_thread = n_samples/nGPUs;
        int device_shift = id*n_samples_thread;
        if (id == (nGPUs-1)){
            n_samples_thread += n_samples - (n_samples/nGPUs)*nGPUs;
        }
        // cuda device vars
        int *t_sources_d;
        float *prestacked_traces_d;
        int *moveouts_P_d;
        int *moveouts_S_d;
        int *st_idx_d;
        float *nw_response_d;
        int *biggest_idx_d;
        int *moveouts_minmax_d;
        cudaError_t cuda_result;

        // allocate memory on device
        cuda_result = cudaMalloc((void**)&t_sources_d, n_test * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&prestacked_traces_d, n_samples * n_stations_whole_array * sizeof(float));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&moveouts_S_d, n_sources * n_stations_whole_array * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&moveouts_P_d, n_sources * n_stations_whole_array * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&st_idx_d, n_sources * n_stations_restricted_array * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&nw_response_d, n_samples_thread * sizeof(float));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&biggest_idx_d, n_samples_thread * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&moveouts_minmax_d, 2 * n_test * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        // transfer from CPU to GPU
        cuda_result = cudaMemcpy(t_sources_d, test_points, n_test * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(prestacked_traces_d, prestacked_traces, n_samples * n_stations_whole_array * sizeof(float),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(moveouts_S_d, moveouts_S, n_sources * n_stations_whole_array * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(moveouts_P_d, moveouts_P, n_sources * n_stations_whole_array * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(st_idx_d, st_idx, n_sources * n_stations_restricted_array * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");
        
        cudaMemGetInfo(&free,&total);
        printf("%d KB free of total %d KB\n",free/1024,total/1024); 

        //cudaFuncSetCacheConfig("stack_kernel", cudaFuncCachePreferL1);
        
        dim3 GS1(ceil((float)n_test/(float)BS.x) + 1);

        moveouts_kernel<<<GS1, BS>>>(moveouts_P_d,
                                     moveouts_minmax_d,
                                     t_sources_d,
                                     st_idx_d,
                                     n_test,
                                     n_stations_whole_array,
                                     n_stations_restricted_array);

        // return an error if something happened in the kernel (and crash the program)
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaDeviceSynchronize();

        dim3 GS2(ceil((float)n_samples_thread/(float)BS.x) + 1);

        //printf("Number of calls: %.2f \n", ceil(n_samples/(float)GRID_SIZE));
        stack_SP_kernel_prestacked<<<GS2, BS>>>(prestacked_traces_d + device_shift,
                                                moveouts_P_d,
                                                moveouts_S_d,
                                                moveouts_minmax_d,
                                                st_idx_d,
                                                t_sources_d,
                                                nw_response_d,
                                                biggest_idx_d,
                                                n_samples,
                                                n_samples_thread,
                                                n_test,
                                                n_stations_whole_array,
                                                n_stations_restricted_array);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaDeviceSynchronize();

        cuda_result = cudaMemcpy(network_response + device_shift, nw_response_d, n_samples_thread * sizeof(float),
                    cudaMemcpyDeviceToHost);
        if (cuda_result != cudaSuccess) printf("Problem with the transfer of memory !\n");
        cuda_result = cudaMemcpy(biggest_idx + device_shift, biggest_idx_d, n_samples_thread * sizeof(int),
                    cudaMemcpyDeviceToHost);
        if (cuda_result != cudaSuccess) printf("Problem with the transfer of memory !\n");
        cudaDeviceSynchronize();

        cudaFree(t_sources_d);
        cudaFree(prestacked_traces_d);
        cudaFree(moveouts_S_d);
        cudaFree(moveouts_P_d);
        cudaFree(st_idx_d);
        cudaFree(nw_response_d);
        cudaFree(biggest_idx_d);
        cudaFree(moveouts_minmax_d);
}
}

void network_response_SP(int*  test_points, 
                         float*  traces_H, 
                         float*  traces_Z,
                         int*  moveouts_P, 
                         int*  moveouts_S, 
                         int* st_idx,
                         int n_test, 
                         int n_samples,
                         float*  network_response, 
                         int*  biggest_idx,
                         int n_stations_whole_array, 
                         int n_stations_restricted_array, 
                         int n_sources) {
    int nGPUs = 0;

    // check how many devices are available
    
    cudaGetDeviceCount(&nGPUs);
    omp_set_num_threads(nGPUs);

    printf("%d cuda devices found on the node\n", nGPUs);
    
    size_t free, total;
    printf("\n");
    cudaMemGetInfo(&free,&total);
    printf("%d KB free of total %d KB\n",free/1024,total/1024); 

    #pragma omp parallel shared(test_points, traces_H, traces_Z, moveouts_P, moveouts_S, st_idx, \
                                network_response, biggest_idx)
    {

        int id = omp_get_thread_num();
        cudaSetDevice(id);
        dim3 BS(256);
        int n_samples_thread = n_samples/nGPUs;
        int device_shift = id*n_samples_thread;
        if (id == (nGPUs-1)){
            n_samples_thread += n_samples - (n_samples/nGPUs)*nGPUs;
        }
        // cuda device vars
        int *t_sources_d;
        float *traces_H_d;
        float *traces_Z_d;
        int *moveouts_P_d;
        int *moveouts_S_d;
        int *st_idx_d;
        float *nw_response_d;
        int *biggest_idx_d;
        int *moveouts_minmax_d;
        cudaError_t cuda_result;

        // allocate memory on device
        cuda_result = cudaMalloc((void**)&t_sources_d, n_test * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&traces_H_d, n_samples * n_stations_whole_array * sizeof(float));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&traces_Z_d, n_samples * n_stations_whole_array * sizeof(float));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&moveouts_S_d, n_sources * n_stations_whole_array * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&moveouts_P_d, n_sources * n_stations_whole_array * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&st_idx_d, n_sources * n_stations_restricted_array * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&nw_response_d, n_samples_thread * sizeof(float));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&biggest_idx_d, n_samples_thread * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        cuda_result = cudaMalloc((void**)&moveouts_minmax_d, 2 * n_test * sizeof(int));
        if (cuda_result != cudaSuccess) printf("Problem with the allocation of memory !\n");

        // transfer from CPU to GPU
        cuda_result = cudaMemcpy(t_sources_d, test_points, n_test * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(traces_H_d, traces_H, n_samples * n_stations_whole_array * sizeof(float),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(traces_Z_d, traces_Z, n_samples * n_stations_whole_array * sizeof(float),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(moveouts_S_d, moveouts_S, n_sources * n_stations_whole_array * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(moveouts_P_d, moveouts_P, n_sources * n_stations_whole_array * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");

        cuda_result = cudaMemcpy(st_idx_d, st_idx, n_sources * n_stations_restricted_array * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (cuda_result != cudaSuccess) printf("Problem when transfering memory from CPU to GPU \n");
        
        cudaMemGetInfo(&free,&total);
        printf("%d KB free of total %d KB\n",free/1024,total/1024); 

        //cudaFuncSetCacheConfig("stack_kernel", cudaFuncCachePreferL1);
        
        dim3 GS1(ceil((float)n_test/(float)BS.x) + 1);

        moveouts_kernel<<<GS1, BS>>>(moveouts_P_d,
                                     moveouts_minmax_d,
                                     t_sources_d,
                                     st_idx_d,
                                     n_test,
                                     n_stations_whole_array,
                                     n_stations_restricted_array);

        // return an error if something happened in the kernel (and crash the program)
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaDeviceSynchronize();

        dim3 GS2(ceil((float)n_samples_thread/(float)BS.x) + 1);

        //printf("Number of calls: %.2f \n", ceil(n_samples/(float)GRID_SIZE));
        stack_SP_kernel<<<GS2, BS>>>(traces_H_d + device_shift,
                                     traces_Z_d + device_shift,
                                     moveouts_P_d,
                                     moveouts_S_d,
                                     moveouts_minmax_d,
                                     st_idx_d,
                                     t_sources_d,
                                     nw_response_d,
                                     biggest_idx_d,
                                     n_samples,
                                     n_samples_thread,
                                     n_test,
                                     n_stations_whole_array,
                                     n_stations_restricted_array);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaDeviceSynchronize();

        cuda_result = cudaMemcpy(network_response + device_shift, nw_response_d, n_samples_thread * sizeof(float),
                    cudaMemcpyDeviceToHost);
        if (cuda_result != cudaSuccess) printf("Problem with the transfer of memory !\n");
        cuda_result = cudaMemcpy(biggest_idx + device_shift, biggest_idx_d, n_samples_thread * sizeof(int),
                    cudaMemcpyDeviceToHost);
        if (cuda_result != cudaSuccess) printf("Problem with the transfer of memory !\n");
        cudaDeviceSynchronize();

        cudaFree(t_sources_d);
        cudaFree(traces_H_d);
        cudaFree(traces_Z_d);
        cudaFree(moveouts_S_d);
        cudaFree(moveouts_P_d);
        cudaFree(st_idx_d);
        cudaFree(nw_response_d);
        cudaFree(biggest_idx_d);
        cudaFree(moveouts_minmax_d);
}
}

} // extern C
