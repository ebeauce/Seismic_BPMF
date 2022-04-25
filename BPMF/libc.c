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
                           float threshold,
                           size_t n_sources,
                           size_t n_stations,
                           size_t n_nn,
                           int *redundant_sources // output pointer
                           ){

    /* redundant_sources should be initialized with zeros
     * threshold is the average absolute time difference
     * in seconds
     * */

    size_t source1_offset, source2_offset; // pointer offsets
    float dt2; // cumulative square time difference between two sources
    float progress_percent = 0.;
    size_t progress_step = 0;

    // make a first iteration just comparing to nearest neighbors
    printf("First (parallel) loop\n");
#pragma omp parallel \
    private(source1_offset, source2_offset)\
    firstprivate(n_nn, n_sources, n_stations, threshold)\
    shared(redundant_sources, moveouts)
    {
        for (size_t i1=0; i1<n_sources-n_nn; i1+=n_nn){
            source1_offset = i1 * n_stations;
            for (size_t i2=i1+1; i2<i1+n_nn; i2++){
                source2_offset = i2 * n_stations;
                // initialize the cumulative time difference
                dt2 = 0.;
                for (size_t s=0; s<n_stations; s++){
                    dt2 += pow(moveouts[source1_offset + s]
                             - moveouts[source2_offset + s], 2);
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

    printf("Second (serial) loop\n");
    for (size_t i1=0; i1<n_sources-1; i1++){
        if (redundant_sources[i1] == 1){
            // pass to the next source if this source
            // has already been identified as redundant
            continue;
        }
        printf("Source %zu\n", i1);

        // some kind of progress bar
        if (i1+1 >= progress_step){
            printf("Progress: %.0f%%\n", 100.*progress_percent);
            progress_percent += 0.1;
            progress_step = (int)(progress_percent*n_sources);
        }

        source1_offset = i1 * n_stations;
        for (size_t i2=i1+1; i2<n_sources; i2++){
            if (redundant_sources[i2] == 1){
                // this source has already been identified
                // as redundant with another source, pass
                continue;
            }
            source2_offset = i2 * n_stations;
            // initialize the cumulative time difference
            dt2 = 0.;
            for (size_t s=0; s<n_stations; s++){
                dt2 += pow(moveouts[source1_offset + s]
                        - moveouts[source2_offset + s], 2);
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

