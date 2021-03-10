Things to implement:



- When doing the matched filter search, we should keep in memory the normalization
 factors that were used for each channel so that it is easy to switch back to the
 recorded amplitudes. The waveforms that are stored in the hdf5 files should also
 be expressed in count/m/s so that we can use them to retrieve the waveforms in
 m/s, which would be useful for the computation of magnitudes.
