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
    float threshold2 = (float)n_stations_for_diff * pow(threshold, 2);
    printf(
            "RMS threshold: %.4f, summed squares threshold: %.4f\n",
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
                        //printf(
                        //        "mv 1 is %f, mv 2 if %f, so the diff is %f\n",
                        //        moveouts[source1_offset + s],
                        //        moveouts[source2_offset + s],
                        //        squared_diff[s]
                        //        );
                    }
                    // sort the squared difference array 
                    selectionSort(squared_diff, n_stations);
                    //for (size_t s=0; s<n_stations; s++){
                    //    printf("Squared diff: %f\n", squared_diff[s]);
                    //}
                    // now, compute dt2 using the n_stations_for_diff
                    // stations with the smallest time differences
                    for (size_t s=0; s < n_stations_for_diff; s++){
                        //printf("ss is %f\n", squared_diff[s]);
                        dt2 += squared_diff[s];
                    }
                    // test whether the summed squared difference is below threshold
                    //if (dt2 < threshold2) redundant_sources[n2] = 1;
                    if (dt2 < threshold2){
                        //printf("Sum %f is smaller than %f\n", dt2, threshold2);
                        redundant_sources[n2] = 1;
                    }

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
        private(dt2, threshold, squared_diff)
        for (size_t n2=n1+1; n2 < n_sources; n2++){
            if (redundant_sources[n2] == 1){
                // this source has already been identified
                // as redundant with another source, pass
                continue;
            }
            source2_offset = n2 * n_stations;
            //// initialize the cumulative time difference
            //dt2 = 0.;
            //for (size_t s=0; s < n_stations; s++){
            //    dt2 += pow(moveouts[source1_offset + s]
            //            - moveouts[source2_offset + s], 2);
            //    if (dt2 > threshold2) break;
            //}
            //if (dt2 < threshold2) redundant_sources[n2] = 1;

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


void swap(float* xp, float* yp) 
{ 
    float temp = *xp; 
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


float _compute_average(float *x,
                       size_t len_x
        ){

    /* Compute the average of x */
    float average = 0.;

    for (size_t i=0; i<len_x; i++){
        average += x[i];
    }

    return average / (float)len_x;
}

float _compute_std(float *x,
                   float x_mean,
                   size_t len_x
        ){
    /* Compute standard deviation of x */
    float sum_squares = 0.;

    for (size_t i=0; i<len_x; i++){
        sum_squares += pow(x[i] - x_mean, 2);
    }

    return sqrtf(sum_squares / (float)len_x);
}

void time_dependent_threshold(float *time_series,
                              float *gaussian_sample,
                              float num_dev,
                              size_t num_samples,
                              size_t half_window_samp,
                              size_t shift_samp,
                              int num_threads,
                              float *threshold
        ){

    const size_t GAUSSIAN_SAMPLE_LEN = 500;
    size_t window_samp = 2 * half_window_samp;
    size_t num_sliding_windows = (num_samples - (window_samp - shift_samp)) / shift_samp;
    float center = 0.;
    float dev = 0.;
    float center_i;
    float dev_i;
    size_t nonzero_i;
    size_t window_offset;
    size_t i, j;
    size_t nonzero_counter = 0;
    size_t num_windows_for_global_stat;

    float* threshold_win;
    float* threshold_diff;
    float* time_series_i;

    threshold_win = malloc(num_sliding_windows * sizeof(float));
    threshold_diff = malloc((num_sliding_windows-1) * sizeof(float));

    if (num_threads != -1){
        omp_set_num_threads(num_threads);
    }

    // first, handle zeros: compute global mean and std without zeros
    // calculate global mean and std using many small windows to
    // leverage parallelization
    num_windows_for_global_stat = num_samples / window_samp;
    #pragma omp parallel for shared(time_series, center, nonzero_counter)
    for (i=0; i<num_windows_for_global_stat; i++){
        center_i = 0.;
        nonzero_i = 0;
        window_offset = i * window_samp;
        time_series_i = time_series + window_offset;
        for (j=0; j<window_samp; j++){
            if (time_series_i[j] != 0.){
                center_i += time_series_i[j];
                nonzero_i++;
            }
        }

        #pragma omp atomic
        center += center_i;
        #pragma omp atomic
        nonzero_counter += nonzero_i;
    }

    center /= (float)nonzero_counter;
    #pragma omp parallel for shared(time_series, center, nonzero_counter)
    for (i=0; i<num_windows_for_global_stat; i++){
        dev_i = 0.;
        window_offset = i * window_samp;
        time_series_i = time_series + window_offset;
        for (j=0; j<window_samp; j++){
            if (time_series_i[j] != 0.){
                dev_i += pow(time_series_i[j] - center, 2);
            }
        }
        #pragma omp atomic
        dev += dev_i;
    }
    dev = sqrtf(dev / (float)nonzero_counter);

    //for (i=0; i<num_samples; i++){
    //    if (time_series[i] != 0.){
    //        center += time_series[i];
    //        nonzero_counter++;
    //    }
    //}
    //if (nonzero_counter > 0){
    //    center /= (float)nonzero_counter;
    //    for (i=0; i<num_samples; i++){
    //        if (time_series[i] != 0.){
    //            dev += pow(time_series[i] - center, 2);
    //        }
    //    }
    //    dev = sqrtf(dev / (float)nonzero_counter);
    //}

    // replace zeros by random samples drawn from a properly scaled gaussian
    #pragma omp parallel for\
    shared(time_series, gaussian_sample)
    for (i=0; i<num_samples; i++){
        if (time_series[i] == 0.){
            time_series[i] = center + gaussian_sample[i%GAUSSIAN_SAMPLE_LEN] * dev;
        }
    }

    // first, compute the central tendency and deviation in the sliding windows
    #pragma omp parallel for\
    private(window_offset, center, dev)\
    shared(time_series, threshold_win)
    for (i=0; i<num_sliding_windows; i++){
        window_offset = i * shift_samp;
        center = _compute_average(
                time_series + window_offset, window_samp
                );
        dev = _compute_std(
                time_series + window_offset, center, window_samp
                );
        threshold_win[i] = center + num_dev * dev;
    }

    // every time the threshold varies, "delay the jump" by one sample
    // in a conservative fashion, keeping the highest threshold value
    for (i=0; i<num_sliding_windows-1; i++){
        threshold_diff[i] = threshold_win[i+1] - threshold_win[i];
    }

    for (i=1; i<num_sliding_windows; i++){
        // i is threshold_win index
        // j is threshold_diff index
        j = i-1;

        // look at backward step at i
        if (threshold_diff[j] < 0.) threshold_win[i] -= threshold_diff[j];

        // update threshold_diff
        threshold_diff[j] = threshold_win[i] - threshold_win[i-1];
    }
    for (i=0; i<num_sliding_windows-1; i++){
        // i is threshold_win index
        // j is threshold_diff index
        j = i;
        if (threshold_diff[j] > 0.) threshold_win[i] += threshold_diff[j];
    }

    // then, build the num_samples-long, step-wise threshold
    #pragma omp parallel for\
    shared(time_series, threshold_win)
    for (i=0; i<num_samples; i++){
        if (i < half_window_samp){
            threshold[i] = threshold_win[0];
        }
        if (i >= num_sliding_windows * shift_samp){
            threshold[i] = threshold_win[num_sliding_windows - 1];
        }
        else{
            threshold[i] = threshold_win[i / shift_samp];
        }
    }

    free(threshold_win);
    free(threshold_diff);
}
