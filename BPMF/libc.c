#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <limits.h>
#include <time.h>

#include "libc.h"

void find_moveouts(const int* moveouts, int *moveouts_minmax, 
                   const int *test_sources, const int *station_indexes,
                   const size_t n_sources, 
                   const size_t n_stations_whole_array,
                   const size_t n_stations_restricted_array) {
    /* Find the minimum and maximum moveouts for point of the grid.
     * Even indexes correspond to minimum moveouts,
     * odd indexes correspond to maximum moveouts. */

    int i; // source counter
    int s; // station counter

#pragma omp parallel for \
    private(i, s) \
    shared(moveouts, moveouts_minmax, \
           test_sources, station_indexes)
    for (i = 0; i < n_sources; i++) {
        int max_moveout = 0;
        int min_moveout = INT_MAX;
        int moveout, ss;

        for (s = 0; s < n_stations_restricted_array; s++) {
            // map from the whole stations array to the restricted array
            ss =  station_indexes[test_sources[i]*n_stations_restricted_array + s];
            moveout = moveouts[test_sources[i]*n_stations_whole_array + s];

            if (moveout > max_moveout) {
                 max_moveout = moveout;
            }
            if (moveout < min_moveout) {
                 min_moveout = moveout;
            }
        }
        moveouts_minmax[i * 2 + 0] = min_moveout;
        moveouts_minmax[i * 2 + 1] = max_moveout;
    }
}

void stack_one_phase(
        float* tracesN,  float* tracesE,     // inputs
        int* moveouts, int* moveouts_minmax, 
        int* station_indexes, int* test_sources,
        float* cosine_azimuths, float* sine_azimuths,
	    size_t n_samples, size_t n_test, 
        size_t n_stations_whole_array,
        size_t n_stations_restricted_array,
	    float* nw_response, int* biggest_idx // outputs
        ) {
    /* Shift and stack the horizontal and vertical characteristic
     * functions based on the moveouts of a single phase. Use the
     * source-receiver azimuths to rotate the traces. */

    int i; // time counter
    int j; // source counter
    int s; // station counter
    
    printf("Number of available threads : %d \n", omp_get_max_threads());
    
    #pragma omp parallel for \
    private(i, j, s) \
    shared(tracesN, tracesE, test_sources, cosine_azimuths, sine_azimuths, \
           moveouts, moveouts_minmax, station_indexes, \
           nw_response, biggest_idx)
    for (i = 0; i < n_samples; i++) {
        float network_response_max = -1000000.;
        int network_resp_idx_max = 0;
        float sum_beamform, transverse_component;
        float *tracesN_, *tracesE_, *cos_az_, *sin_az_;
        int source_idx_whole_array, source_idx_restricted_array, ss, moveout, idx_shift;
        
        for (j = 0; j < n_test; j++) {
            sum_beamform = 0.0;
            
            source_idx_whole_array = test_sources[j]*n_stations_whole_array; // position in the moveouts vector
            source_idx_restricted_array = test_sources[j]*n_stations_restricted_array; // position in the station indexes vector
            idx_shift = i - moveouts_minmax[j * 2 + 0]; // position on the time axis
            
            if (idx_shift < 0) continue; // don't do anything before time 0

            // ---------------------------------
            // define the local pointers
            tracesN_ = tracesN + idx_shift;
            tracesE_ = tracesE + idx_shift;
            cos_az_  = cosine_azimuths + source_idx_whole_array;
            sin_az_  = sine_azimuths   + source_idx_whole_array;
            // ---------------------------------
            
            for (s = 0; s < n_stations_restricted_array; s++) {
                // map from the closest stations to the whole array of stations
                ss = station_indexes[source_idx_restricted_array + s];
                moveout = moveouts[source_idx_whole_array + ss];
                if ((idx_shift + moveout) < n_samples) {
                    // Rotate the seismograms to get the transverse component.
                    transverse_component = pow(tracesN_[ss*n_samples + moveout] * cos_az_[ss] +\
                                               tracesE_[ss*n_samples + moveout] * sin_az_[ss], 2);
                    sum_beamform += transverse_component;
                }
            }
            
            if (sum_beamform > network_response_max) {
                network_response_max = sum_beamform; 
                network_resp_idx_max = test_sources[j];
            }
        }
        nw_response[i] = network_response_max;
        biggest_idx[i] = network_resp_idx_max;
    }
}


void stack_SP(float* traces_H, float* traces_Z, 
              int* moveouts_P, int* moveouts_S, int* station_indexes,
	          int* moveouts_minmax, int* test_sources,
	          float* nw_response, int* biggest_idx, 
	          size_t n_samples, size_t n_test, 
              size_t n_stations_whole_array,
              size_t n_stations_restricted_array) {
  /* Shift and stack the horizontal and vertical characteristic
   * functions based on the moveouts of both the first S- and P-wave
   * arrivals. */

  int i; // time counter
  int j; // source counter
  int s; // station counter
  
  printf("Number of available threads : %d \n", omp_get_max_threads());

  #pragma omp parallel for \
  private(i, j, s) \
  shared(traces_H, traces_Z, test_sources, \
         moveouts_P, moveouts_S, station_indexes, moveouts_minmax, \
         nw_response, biggest_idx)
  for (i = 0; i < n_samples; i++) {
    float network_response_max = -1000000.;
    int network_resp_idx_max = 0;
    float sum_beamform_SP, sum_beamform_S, sum_beamform_P;
    int source_idx_whole_array, source_idx_restricted_array, ss, moveout_P, moveout_S, idx_shift;

    for (j = 0; j < n_test; j++) {
      sum_beamform_SP = 0.0;
      sum_beamform_S  = 0.0;
      sum_beamform_P  = 0.0;

      source_idx_whole_array      = test_sources[j] * n_stations_whole_array; // position in the moveouts vector
      source_idx_restricted_array = test_sources[j] * n_stations_restricted_array; // position in the station indexes vector
      idx_shift = i - moveouts_minmax[j * 2 + 0];

      if (idx_shift < 0) continue; // don't do anything before 0 time
      
      for (s = 0; s < n_stations_restricted_array; s++) {
        // map from the closest stations to the whole array of stations
        ss        = station_indexes[source_idx_restricted_array + s]; 
        moveout_P = moveouts_P[source_idx_whole_array + ss];
        moveout_S = moveouts_S[source_idx_whole_array + ss];
        if ((idx_shift + moveout_S) < n_samples && (idx_shift + moveout_P) < n_samples) {
          // give equal weight to horizontal and vertical components.
          sum_beamform_S += traces_H[ss * n_samples + idx_shift + moveout_S];
          sum_beamform_P += traces_Z[ss * n_samples + idx_shift + moveout_P];
          // give more weight to horizontal components
          //sum_beamform_S += 2.*traces_H[ss * n_samples + idx_shift + moveout_S] \
          //                  + traces_Z[ss * n_samples + idx_shift + moveout_S];
          //sum_beamform_P += 2.*traces_H[ss * n_samples + idx_shift + moveout_P] \
          //                  + traces_Z[ss * n_samples + idx_shift + moveout_P];

        }
      }

      sum_beamform_SP = sum_beamform_S + sum_beamform_P;

      if (sum_beamform_SP > network_response_max) {
        network_response_max = sum_beamform_SP;
        network_resp_idx_max = test_sources[j];
      }
    }
    nw_response[i] = network_response_max;
    biggest_idx[i] = network_resp_idx_max;
    //printf("NR=%.2f\n", nw_response[i]);
  }
}

void stack_SP_prestacked(float* traces, 
                         int* moveouts_P,
                         int* moveouts_S,
                         int* station_indexes,
	                     int* moveouts_minmax,
                         int* test_sources,
	                     float* nw_response,                      // output pointer
                         int* biggest_idx,                        // output pointer
	                     size_t n_samples,
                         size_t n_test, 
                         size_t n_stations_whole_array,
                         size_t n_stations_restricted_array) {
  /* Shift and stack the composite characteristic functions based
   * on the moveouts of both the first S- and P-wave arrivals. The
   * horizontal and vertical characteristic functions were stacked
   * before calling the C routine to optimize the computation time. */

  int i; // time counter
  int j; // source counter
  int s; // station counter
  
  printf("Number of available threads : %d \n", omp_get_max_threads());

  #pragma omp parallel for \
  private(i, j, s) \
  shared(traces, test_sources, \
         moveouts_P, moveouts_S, station_indexes, \
         moveouts_minmax, nw_response, biggest_idx)
  for (i = 0; i < n_samples; i++) {
    float network_response_max = -10000.;
    int network_resp_idx_max = 0;
    float sum_beamform_SP;
    int source_idx_whole_array, source_idx_restricted_array, ss, moveout_P, moveout_S, idx_shift;

    for (j = 0; j < n_test; j++) {
      sum_beamform_SP = 0.0;

      source_idx_whole_array      = test_sources[j] * n_stations_whole_array; // position in the moveouts vector
      source_idx_restricted_array = test_sources[j] * n_stations_restricted_array; // position in the station indexes vector
      idx_shift = i - moveouts_minmax[j * 2 + 0];

      if (idx_shift < 0) continue; // don't do anything before 0 time
      
      for (s = 0; s < n_stations_restricted_array; s++) {
        // map from the closest stations to the whole array of stations
        ss        = station_indexes[source_idx_restricted_array + s]; 
        moveout_P = moveouts_P[source_idx_whole_array + ss];
        moveout_S = moveouts_S[source_idx_whole_array + ss];
        if ((idx_shift + moveout_S) < n_samples && (idx_shift + moveout_P) < n_samples) {
          sum_beamform_SP +=    traces[ss * n_samples + idx_shift + moveout_S]\
                              + traces[ss * n_samples + idx_shift + moveout_P];
        }
      }

      if (sum_beamform_SP > network_response_max) {
        network_response_max = sum_beamform_SP;
        network_resp_idx_max = test_sources[j];
      }
    }
    nw_response[i] = network_response_max;
    biggest_idx[i] = network_resp_idx_max;
  }
}


void network_response(int*  test_points, float*  tracesN, float* tracesE, 
                      float* cosine_azimuths, float* sine_azimuths,
                      int*  moveouts, int* station_indexes,
                      int n_test, int n_samples,
                      float*  network_response, int*  biggest_idx,
                      int n_stations_whole_array, 
                      int n_stations_restricted_array, 
                      int n_sources) {

    /*
     * This function computes the beamformed network response by rotating the horizontal traces 
     * to get the transverse component. For a well-defined source-receiver path, the transverse
     * component is free of any P phases.
     */

    int *moveouts_minmax;
    moveouts_minmax = (int *)malloc(2 * n_test * sizeof(int));
    
    find_moveouts(moveouts,
                  moveouts_minmax,
                  test_points,
                  station_indexes,
                  n_test,
                  n_stations_whole_array,
                  n_stations_restricted_array);
    
    stack_one_phase(tracesN,
            tracesE,
            moveouts,
            moveouts_minmax,
            station_indexes,
            test_points,
            cosine_azimuths,
            sine_azimuths,
            n_samples,
            n_test,
            n_stations_whole_array,
            n_stations_restricted_array,
            network_response,
            biggest_idx);
    
    free(moveouts_minmax);
}

void network_response_SP(int*  test_points, float*  traces_H, float* traces_Z, 
                         int* moveouts_P, int* moveouts_S, int* station_indexes,
                         int n_test, int n_samples,
                         float*  network_response, int*  biggest_idx,
                         int n_stations_whole_array,
                         int n_stations_restricted_array,
                         int n_sources) {
    /*
     * Stack the P- and S-wave beams for all sources in test_points
     * and all time steps.
     */

    int *moveouts_minmax;
    moveouts_minmax = (int *)malloc(2 * n_test * sizeof(int));

    find_moveouts(moveouts_P,
                  moveouts_minmax,
                  test_points,
                  station_indexes,
                  n_test,
                  n_stations_whole_array,
                  n_stations_restricted_array);

    stack_SP(traces_H,
             traces_Z,
             moveouts_P,
             moveouts_S,
             station_indexes,
             moveouts_minmax,
             test_points,
             network_response,
             biggest_idx,
             n_samples,
             n_test,
             n_stations_whole_array,
             n_stations_restricted_array);


    free(moveouts_minmax);
}

void network_response_SP_prestacked(int* test_points,
                                    float* traces, 
                                    int* moveouts_P,
                                    int* moveouts_S,
                                    int* station_indexes,
                                    int n_test,
                                    int n_samples,
                                    float* network_response,
                                    int* biggest_idx,
                                    int n_stations_whole_array,
                                    int n_stations_restricted_array,
                                    int n_sources) {
    /*
     * Stack the P- and S-wave beams for all sources in test_points
     * and all time steps. The horizontal and vertical detection traces
     * are already stacked before calling this function.
     */

    int *moveouts_minmax;
    moveouts_minmax = (int *)malloc(2 * n_test * sizeof(int));

    find_moveouts(moveouts_P,
                  moveouts_minmax,
                  test_points,
                  station_indexes,
                  n_test,
                  n_stations_whole_array,
                  n_stations_restricted_array);

    stack_SP_prestacked(traces,
                        moveouts_P,
                        moveouts_S,
                        station_indexes,
                        moveouts_minmax,
                        test_points,
                        network_response,
                        biggest_idx,
                        n_samples,
                        n_test,
                        n_stations_whole_array,
                        n_stations_restricted_array);


    free(moveouts_minmax);
}


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

void kurtosis(float *signal,
              int W,
              int n_stations,
              int n_components,
              int length,
              float *kurto){
    /* Naive running kurtosis. This might benefit from implementing
     * an enhanced running kurtosis algorithm. */

    int s,c,n,i,nn; // counter

#pragma omp parallel for \
    private(s, c, n, i, nn) \
    shared(signal, kurto, n_stations, n_components, length, W)
    for (s=0; s<n_stations; s++){
        int station_offset = s * n_components * length;
        float Wf = (float)W;
        float *signal_ch=NULL, *kurto_ch=NULL;
        float mean = 0., m2 = 0., m4 = 0., ds = 0.;
        int component_offset;
        for (c=0; c<n_components; c++){
            component_offset = c * length;
            signal_ch = signal + (station_offset + component_offset);
            kurto_ch = kurto + (station_offset + component_offset);
            for (n=W; n<length; n++){
                nn = n - W;
                mean = 0.;
                m2 = 0.;
                m4 = 0.;
                for (i=0; i<W; i++) mean += signal_ch[nn+i];
                mean /= Wf;
                for (i=0; i<W; i++){
                    ds = signal_ch[nn+i] - mean;
                    m2 += pow(ds, 2.);
                    m4 += pow(ds, 4.);
                }
                m2 /= Wf;
                m4 /= Wf;
                if (m2 > 0.000001) kurto_ch[n] = 1./( (Wf-2)*(Wf-3) ) * ((pow(Wf, 2)-1) * m4/pow(m2, 2) - 3*pow(Wf-1., 2));
            }
        }
    }
}

void find_similar_moveouts(float *moveouts,
                           float threshold,
                           int n_sources,
                           int n_stations,
                           int *redundant_sources // output pointer
                           ){

    /* redundant_sources should be initialized with zeros
     * threshold is the average absolute time difference
     * in seconds
     * */

    int i1, i2, s; // counters
    int source1_offset, source2_offset; // pointer offsets
    float dt2; // cumulative square time difference between two sources

    // first: get min travel times
    int i; // source counter
    int *moveouts_min;
    moveouts_min = (int *)malloc(n_sources * sizeof(int));
   
    #pragma omp parallel for \
    private(i, s) \
    shared(moveouts, moveouts_min)
    for (i = 0; i < n_sources; i++) {
      int min_moveout = INT_MAX;
      int moveout;
    
      for (s = 0; s < n_stations; s++) {
          moveout = moveouts[i * n_stations + s];
    
          if (moveout < min_moveout) {
               min_moveout = moveout;
          }
      }
      moveouts_min[i] = min_moveout;
    }

    for (i1=0; i1<n_sources-1; i1++){
        if (redundant_sources[i1] == 1){
            // pass to the next source if this source
            // has already been identified as redundant
            continue;
        }
        printf("Source %d / %d\n", i1+1, n_sources);
        source1_offset = i1 * n_stations;
        for (i2=i1+1; i2<n_sources; i2++){
            //if (i2 == i1){
            //    // avoid the trivial match between
            //    // the source and itself
            //    continue;
            //}
            if (redundant_sources[i2] == 1){
                // this source has already been identified
                // as redundant with another source, pass
                continue;
            }
            source2_offset = i2 * n_stations;
            // initialize the cumulative time difference
            dt2 = 0.;
            for (s=0; s<n_stations; s++){
                dt2 += pow((moveouts[source1_offset + s] - moveouts_min[i1])
                         - (moveouts[source2_offset + s] - moveouts_min[i2]), 2);
            }
            if (dt2 < pow((float)n_stations*threshold, 2)){
                // if the difference is small enough,
                // we consider the two sources to be redundant
                redundant_sources[i2] = 1;
                //printf("Source %d and source %d are redundant: dt2=%.2f\n", i1, i2, dt2);
            }
        }
    }
}

