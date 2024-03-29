#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <limits.h>
#include <time.h>

#include "libc.h"

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
                           float *source_longitude,
                           float *source_latitude,
                           float *cell_longitude,
                           float *cell_latitude,
                           float threshold,
                           size_t n_sources,
                           size_t n_stations,
                           size_t n_cells_longitude,
                           size_t n_cells_latitude,
                           size_t n_stations_for_diff,
                           int num_threads,
                           int *redundant_sources // output pointer
                           ){

    /* redundant_sources should be initialized with zeros
     * threshold is the root mean square time difference in seconds
     * */

    size_t source1_offset, source2_offset; // pointer offsets
    float dt2; // cumulative square time difference between two sources
    float progress_percent = 0.;
    size_t progress_step = 0;
    float squared_diff[n_stations];
    float threshold2 = (float)n_stations * pow(threshold, 2);
    printf(
            "RMS threshold: %.4f, summed square threshold: %.4f\n",
            threshold,
            threshold2
            );

    // --------------------------
    //       test sorting (uncomment to double check that the code works)
    // --------------------------
    //size_t n1=0;
    //size_t n2=1;
    //float squared_diff[n_stations];

    //printf("Unsorted array:\n");
    //for (size_t s=0; s < n_stations; s++){
    //    squared_diff[s] = pow(
    //            moveouts[n1 * n_stations + s]
    //            -
    //            moveouts[n2 * n_stations + s],
    //            2
    //            );
    //    printf("Element %zu: %.4f ,", s, squared_diff[s]);
    //}
    //printf("\n");

    //selectionSort(squared_diff, n_stations);
    //printf("Sorted array:\n");
    //for (size_t s=0; s < n_stations; s++){
    //    printf("Element %zu: %.4f ,", s, squared_diff[s]);
    //}
    //printf("\n");


    if (num_threads != -1){
        omp_set_num_threads(num_threads);
    }

    for(size_t i=0; i < n_cells_longitude; i++){
        //printf("%d\n", i);
        printf("---------- %d / %d ----------\n", i+1, n_cells_longitude);
        for (size_t j=0; j < n_cells_latitude; j++){
            for (size_t n1=0; n1 < n_sources - 1; n1++){
                // test whether n1 is in cell
                if ( (source_longitude[n1] < cell_longitude[i]) ||
                     (source_longitude[n1] >= cell_longitude[i+1]) ||
                     (source_latitude[n1] < cell_latitude[j]) ||
                     (source_latitude[n1] >= cell_latitude[j+1]) ){
                    continue;
                }
                // test whether n1 is already classified as a redundant source
                if (redundant_sources[n1] == 1) continue;
                source1_offset = n1 * n_stations;
                #pragma omp parallel for\
                private(dt2, threshold, squared_diff)
                for (size_t n2=n1+1; n2 < n_sources; n2++){
                    // test whether n2 is in cell
                    if ( (source_longitude[n2] < cell_longitude[i]) ||
                         (source_longitude[n2] >= cell_longitude[i+1]) ||
                         (source_latitude[n2] < cell_latitude[j]) ||
                         (source_latitude[n2] >= cell_latitude[j+1]) ){
                        continue;
                    }
                    // test whether n2 is already classified as a redundant source
                    if (redundant_sources[n2] == 1) continue;
                    source2_offset = n2 * n_stations;
                    // initialize sum
                    dt2 = 0.;
                    for (size_t s=0; s < n_stations; s++){
                        squared_diff[s] = pow(
                                moveouts[source1_offset + s]
                                -
                                moveouts[source2_offset + s],
                                2
                                );
                    }
                    // sort the squared difference array 
                    selectionSort(squared_diff, n_stations);
                    // now, compute dt2 using the n_stations_for_diff
                    // stations with the smallest time differences
                    for (size_t s=0; s < n_stations_for_diff; s++){
                        dt2 += squared_diff[s];
                    }
                    // test whether the summed squared difference is below threshold
                    if (dt2 < threshold2) redundant_sources[n2] = 1;

                }
            }
        }
    }

    printf("Second loop\n");
    for (size_t n1=0; n1 < n_sources-1; n1++){
        if (redundant_sources[n1] == 1){
            // pass to the next source if this source
            // has already been identified as redundant
            continue;
        }

        // some kind of progress bar
        if (n1+1 >= progress_step){
            printf("Progress: %.0f%%\n", 100.*progress_percent);
            progress_percent += 0.1;
            progress_step = (int)(progress_percent*n_sources);
        }

        source1_offset = n1 * n_stations;
        #pragma omp parallel for\
        private(dt2, threshold)
        for (size_t n2=n1+1; n2 < n_sources; n2++){
            if (redundant_sources[n2] == 1){
                // this source has already been identified
                // as redundant with another source, pass
                continue;
            }
            source2_offset = n2 * n_stations;
            // initialize the cumulative time difference
            dt2 = 0.;
            for (size_t s=0; s < n_stations; s++){
                dt2 += pow(moveouts[source1_offset + s]
                        - moveouts[source2_offset + s], 2);
                if (dt2 > threshold2) break;
            }
            if (dt2 < threshold2) redundant_sources[n2] = 1;
        }
    }
}


void swap(float* xp, float* yp) 
{ 
    int temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 
  
// Function to perform Selection Sort 
void selectionSort(float arr[], int n) 
{ 
    int i, j, min_idx; 
  
    // One by one move boundary of 
    // unsorted subarray 
    for (i = 0; i < n - 1; i++) { 
        // Find the minimum element in 
        // unsorted array 
        min_idx = i; 
        for (j = i + 1; j < n; j++) 
            if (arr[j] < arr[min_idx]) 
                min_idx = j; 
  
        // Swap the found minimum element 
        // with the first element 
        swap(&arr[min_idx], &arr[i]); 
    } 
} 

void select_cc_indexes(float *cc,
                       float *threshold,
                       size_t search_win,
                       size_t n_corr,
                       int *selection){

    /* Select indexes of correlation coefficients (CC) that are
     * above `threshold` and spaced by at least `search_win`.
     */
    size_t i_start;

//#pragma omp parallel \
//    private(i_start) \
//    firstprivate(search_win, n_corr) \
//    shared(cc, threshold, selection)
    for (size_t i=0; i<n_corr; i++){
        // first test: is it above detection threshold?
        if (cc[i] > threshold[i]){
            selection[i] = 1;
        }
        else{
            selection[i] = 0;
        }
        // second test: was there another detection within the preceding
        // `search_win`-correlation window?
        // if yes, keep highest cc's index
        if (i <= search_win){
            i_start = 0;
        }
        else{
            i_start = i-search_win;
        }
        for (size_t j=i_start; j<i; j++){
            if (cc[j] > cc[i]){
                // i-th correlation is not a new event detection
                selection[i] = 0;
                break;
            }
            else{
                // the i-th correlation is a better detection
                // than the j-th
                selection[j] = 0;
            }
        }
    }
}
