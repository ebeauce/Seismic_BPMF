import os

import ctypes as C
import numpy as np
import datetime as dt
from .config import package

cpu_loaded = False
gpu_loaded = False

libpath = os.path.join(package, 'lib')

try:
    _libc = C.cdll.LoadLibrary(os.path.join(libpath, 'libc.so'))
    cpu_loaded = True
except:
    print('Missing libc.so! You won''t be able to run multiplet/template searches on the CPU')
    print('Should be at {}libc.so'.format(os.path.join(libpath, '')))


try:
    _libcu = C.cdll.LoadLibrary(os.path.join(libpath, 'libcu.so'))
    gpu_loaded = True
except:
    print('Missing libcu.so! You won''t be able to run multiplet/template searches on the GPU')
    print('Should be at {}libcu.so'.format(os.path.join(libpath, '')))


if cpu_loaded:
    _libc.network_response.argtypes = [C.POINTER(C.c_int),    # test points
                                       C.POINTER(C.c_float),  # traces N
                                       C.POINTER(C.c_float),  # traces E
                                       C.POINTER(C.c_float),  # cosine azimuths
                                       C.POINTER(C.c_float),  # sine azimuths
                                       C.POINTER(C.c_int),    # moveouts
                                       C.POINTER(C.c_int),    # stations index (ligne q commenter pour revenir a une version classique)
                                       C.c_int,               # n_test
                                       C.c_int,               # n_samples
                                       C.POINTER(C.c_float),  # ntwkrsp
                                       C.POINTER(C.c_int),    # biggest_idx
                                       C.c_int,               # n_stations_whole_array
                                       C.c_int,               # n_stations_restricted_array
                                       C.c_int]               # n_sources

    _libc.network_response_SP.argtypes = [C.POINTER(C.c_int),    # test points
                                          C.POINTER(C.c_float),  # average horizontal traces
                                          C.POINTER(C.c_float),  # vertical traces
                                          C.POINTER(C.c_int),    # moveouts P
                                          C.POINTER(C.c_int),    # moveouts S
                                          C.POINTER(C.c_int),    # station indexes
                                          C.c_int,               # n_test
                                          C.c_int,               # n_samples
                                          C.POINTER(C.c_float),  # ntwkrsp
                                          C.POINTER(C.c_int),    # biggest_idx
                                          C.c_int,               # n_stations
                                          C.c_int,               # n_stations_used
                                          C.c_int]               # n_sources

    _libc.network_response_SP_prestacked.argtypes = [C.POINTER(C.c_int),    # test points
                                                     C.POINTER(C.c_float),  # stacked detection traces
                                                     C.POINTER(C.c_int),    # moveouts P
                                                     C.POINTER(C.c_int),    # moveouts S
                                                     C.POINTER(C.c_int),    # station indexes
                                                     C.c_int,               # n_test
                                                     C.c_int,               # n_samples
                                                     C.POINTER(C.c_float),  # ntwkrsp
                                                     C.POINTER(C.c_int),    # biggest_idx
                                                     C.c_int,               # n_stations
                                                     C.c_int,               # n_stations_used
                                                     C.c_int]               # n_sources

    _libc.kurtosis.argtypes = [C.POINTER(C.c_float),
                               C.c_int,
                               C.c_int,
                               C.c_int,
                               C.c_int,
                               C.POINTER(C.c_float)]

    _libc.find_similar_moveouts.argtypes = [C.POINTER(C.c_float), # moveouts
                                            C.c_float,            # average absolute time difference threshold
                                            C.c_int,              # number of grid points 
                                            C.c_int,              # number of stations
                                            C.POINTER(C.c_int)    # output pointer: redundant sources
                                            ]

if gpu_loaded:
    _libcu.network_response.argtypes = [C.POINTER(C.c_int),    # test points
                                        C.POINTER(C.c_float),  # traces N
                                        C.POINTER(C.c_float),  # traces E
                                        C.POINTER(C.c_float),  # cosine azimuths
                                        C.POINTER(C.c_float),  # sine azimuths
                                        C.POINTER(C.c_int),    # moveouts
                                        C.POINTER(C.c_int),    # stations index (ligne q commenter pour revenir a une version classique)
                                        C.c_int,               # n_test
                                        C.c_int,               # n_samples
                                        C.POINTER(C.c_float),  # ntwkrsp
                                        C.POINTER(C.c_int),    # biggest_idx
                                        C.c_int,               # n_stations
                                        C.c_int,               # n_stations_used
                                        C.c_int]               # n_sources

    _libcu.network_response_SP.argtypes = [C.POINTER(C.c_int),    # test points
                                          C.POINTER(C.c_float),  # average horizontal traces
                                          C.POINTER(C.c_float),  # vertical traces
                                          C.POINTER(C.c_int),    # moveouts P
                                          C.POINTER(C.c_int),    # moveouts S
                                          C.POINTER(C.c_int),    # station indexes
                                          C.c_int,               # n_test
                                          C.c_int,               # n_samples
                                          C.POINTER(C.c_float),  # ntwkrsp
                                          C.POINTER(C.c_int),    # biggest_idx
                                          C.c_int,               # n_stations
                                          C.c_int,               # n_stations_used
                                          C.c_int]               # n_sources

    _libcu.network_response_SP_prestacked.argtypes = [C.POINTER(C.c_int),    # test points
                                                      C.POINTER(C.c_float),  # stacked traces
                                                      C.POINTER(C.c_int),    # moveouts P
                                                      C.POINTER(C.c_int),    # moveouts S
                                                      C.POINTER(C.c_int),    # station indexes
                                                      C.c_int,               # n_test
                                                      C.c_int,               # n_samples
                                                      C.POINTER(C.c_float),  # ntwkrsp
                                                      C.POINTER(C.c_int),    # biggest_idx
                                                      C.c_int,               # n_stations
                                                      C.c_int,               # n_stations_used
                                                      C.c_int]               # n_sources

def network_response(data_N, data_E, cosine_azimuths, sine_azimuths,
                     moveouts, smooth_win, device='gpu',
                     closest_stations=None, test_points=None):
    """Test function.  

    """
    n_stations = np.int32(data_N.shape[0])
    n_sources = np.int32(moveouts.shape[0])
    if closest_stations is not None:
        n_stations_used = np.int32(closest_stations.shape[-1])
        print(f'Number of stations used: {n_stations_used}')
    else:
        n_stations_used = n_stations
        closest_stations = np.arange(n_sources*n_stations, dtype=np.int32)
    moveouts = moveouts.reshape(
            moveouts.size//n_stations, n_stations)
    if test_points is None:
        test_points = np.arange(n_sources, dtype=np.int32)
    else:
        test_points = np.int32(test_points)
    n_test = np.int32(test_points.size)
    n_samples = np.int32(data_N.shape[-1])
    #-----------flatten the arrays---------
    moveouts = np.int32(moveouts.flatten())
    data_N = np.float32(data_N.flatten())
    data_E = np.float32(data_E.flatten())
    cosine_azimuths = np.float32(cosine_azimuths.flatten())
    sine_azimuths = np.float32(sine_azimuths.flatten())
    #--------------------------------------
    ntwkrsp = np.zeros(n_samples, np.float32)
    biggest_idx = np.zeros(n_samples, np.int32)
    smoothed = np.zeros(n_samples, np.float32)
    half_smooth = np.int32(smooth_win / 2)
    print(f'nb of samples per trace = {n_samples}, '
          f'nb of stations = {n_stations}, '
          f'nb of stations used with each grid point = {n_stations_used}')

    if device == 'gpu':
        print('Memory required on the GPU: {:d}Mb'.\
                format((test_points.nbytes + moveouts.nbytes + data_N.nbytes\
                     + data_E.nbytes + stations_idx.nbytes + ntwkrsp.nbytes\
                     + biggest_idx.nbytes + smoothed.nbytes)/1024.e3))
        _libcu.network_response(
            test_points.ctypes.data_as(C.POINTER(C.c_int)),
            data_N.ctypes.data_as(C.POINTER(C.c_float)),
            data_E.ctypes.data_as(C.POINTER(C.c_float)),
            cosine_azimuths.ctypes.data_as(C.POINTER(C.c_float)),
            sine_azimuths.ctypes.data_as(C.POINTER(C.c_float)),
            moveouts.ctypes.data_as(C.POINTER(C.c_int)),
            closest_stations.ctypes.data_as(C.POINTER(C.c_int)),
            n_test,
            n_samples,
            ntwkrsp.ctypes.data_as(C.POINTER(C.c_float)),
            biggest_idx.ctypes.data_as(C.POINTER(C.c_int)),
            smoothed.ctypes.data_as(C.POINTER(C.c_float)),
            n_stations,
            n_stations_used,
            n_sources,
            half_smooth)
    elif device == 'cpu':
        _libc.network_response(
            test_points.ctypes.data_as(C.POINTER(C.c_int)),
            data_N.ctypes.data_as(C.POINTER(C.c_float)),
            data_E.ctypes.data_as(C.POINTER(C.c_float)),
            cosine_azimuths.ctypes.data_as(C.POINTER(C.c_float)),
            sine_azimuths.ctypes.data_as(C.POINTER(C.c_float)),
            moveouts.ctypes.data_as(C.POINTER(C.c_int)),
            closest_stations.ctypes.data_as(C.POINTER(C.c_int)),
            n_test,
            n_samples,
            ntwkrsp.ctypes.data_as(C.POINTER(C.c_float)),
            biggest_idx.ctypes.data_as(C.POINTER(C.c_int)),
            smoothed.ctypes.data_as(C.POINTER(C.c_float)),
            n_stations,
            n_stations_used,
            n_sources,
            half_smooth)

    return ntwkrsp, biggest_idx, smoothed

def network_response_SP_prestacked(
        prestacked_traces, moveouts_P, moveouts_S, smooth_win,
        test_points=None, closest_stations=None, device='cpu'):
    """Backproject the seismic wavefield onto a grid of test seismic sources.  

    The 3-component detection traces are already stacked to speed-up the
    computation.

    Parameters
    -------------
    prestacked_traces: (n_stations, n_samples) array
        Numpy array with the characteristic functions already stacked
        along the component axis.
    moveouts_P: (n_sources, n_stations) int array
        P-wave moveouts, in samples, from each of the `n_sources` grid points
        to each of the `n_stations` seismometers.
    moveouts_S: (n_sources, n_stations) int array
        S-wave moveouts, in samples, from each of the `n_sources` grid points
        to each of the `n_stations` seismometers.
    smooth_win: remove?
    test_points: (n_sources,) int array, default to None
        Indexes of the test sources to use in the backprojection.
        If None, all sources are used.
    closest_stations: (n_sources, n_cl_stations) int array, default to None
        If not None, then only the closest `n_cl_stations` seismometers to a
        given grid point contribute to the stacking. This is particularly
        useful for large networks where remote stations cannot record
        earthquakes with good SNR.
    device: string, default to 'cpu'
        If `device` is 'cpu', use the C code. If `device` is 'gpu',
        use the CUDA-C code. 
    """
    n_stations = np.int32(prestacked_traces.shape[0])
    n_sources = np.int32(moveouts_S.shape[0]) # number of sources in the moveouts array
    if closest_stations is not None:
        n_stations_used = np.int32(closest_stations.shape[-1])
        closest_stations = np.int32(closest_stations.flatten())
        print(f'Number of stations used: {n_stations_used}')
    else:
        n_stations_used = n_stations
        closest_stations = np.repeat(
                np.arange(n_stations, dtype=np.int32), n_sources).\
                        reshape(n_sources, n_stations, order='F').flatten()
    #--------------------------------------------------------------------------
    #-------- reshape the S moveouts array to get the number of sources -------
    if test_points is None:
        test_points = np.arange(n_sources, dtype=np.int32)
    else:
        # typically restrict the grid search to a subset of points
        print('Use the restricted grid provided by test_points.')
        test_points = np.int32(test_points)
    n_test = np.int32(test_points.size) # number of sources that are going to be visited
    n_samples = np.int32(prestacked_traces.shape[-1])
    #---------- flatten the traces -------------
    prestacked_traces = np.float32(prestacked_traces.flatten())
    # ---------------------------------------------------------------------
    # FLATTEN MOVEOUTS
    moveouts_P  = np.int32(moveouts_P.flatten())
    moveouts_S  = np.int32(moveouts_S.flatten())
    ntwkrsp     = np.zeros(n_samples, np.float32)
    biggest_idx = np.zeros(n_samples, np.int32)
    print(f'{n_stations_used} stations are used per test source, '
          f'{n_samples} samples on each')
    print(f'{n_sources} sources in the grid, {n_test} test sources')
    if device == 'gpu':
        _libcu.network_response_SP_prestacked(
            test_points.ctypes.data_as(C.POINTER(C.c_int)),
            prestacked_traces.ctypes.data_as(C.POINTER(C.c_float)),
            moveouts_P.ctypes.data_as(C.POINTER(C.c_int)),
            moveouts_S.ctypes.data_as(C.POINTER(C.c_int)),
            closest_stations.ctypes.data_as(C.POINTER(C.c_int)),
            n_test,
            n_samples,
            ntwkrsp.ctypes.data_as(C.POINTER(C.c_float)),
            biggest_idx.ctypes.data_as(C.POINTER(C.c_int)),
            n_stations,
            n_stations_used,
            n_sources)

    elif device == 'cpu':
        _libc.network_response_SP_prestacked(
            test_points.ctypes.data_as(C.POINTER(C.c_int)),
            prestacked_traces.ctypes.data_as(C.POINTER(C.c_float)),
            moveouts_P.ctypes.data_as(C.POINTER(C.c_int)),
            moveouts_S.ctypes.data_as(C.POINTER(C.c_int)),
            closest_stations.ctypes.data_as(C.POINTER(C.c_int)),
            n_test,
            n_samples,
            ntwkrsp.ctypes.data_as(C.POINTER(C.c_float)),
            biggest_idx.ctypes.data_as(C.POINTER(C.c_int)),
            n_stations,
            n_stations_used,
            n_sources)

    return ntwkrsp, biggest_idx

def network_response_SP(traces_H, traces_Z, moveouts_P, moveouts_S,
                        smooth_win, test_points=None, closest_stations=None,
                        device='gpu'):
    """Backproject the seismic wavefield onto a grid of test seismic sources.  

    The 3-component characteristic functions are already stacked to speed-up the
    computation.

    Parameters
    -------------
    traces_H: (n_stations, n_samples) array
        Numpy array with the horizontal component characteristic function.
    traces_Z: (n_stations, n_samples) array
        Numpy array with the vertical component characteristic function.
    moveouts_P: (n_sources, n_stations) int array
        P-wave moveouts, in samples, from each of the `n_sources` grid points
        to each of the `n_stations` seismometers.
    moveouts_S: (n_sources, n_stations) int array
        S-wave moveouts, in samples, from each of the `n_sources` grid points
        to each of the `n_stations` seismometers.
    smooth_win: remove?
    test_points: (n_sources,) int array, default to None
        Indexes of the test sources to use in the backprojection.
        If None, all sources are used.
    closest_stations: (n_sources, n_cl_stations) int array, default to None
        If not None, then only the closest `n_cl_stations` seismometers to a
        given grid point contribute to the stacking. This is particularly
        useful for large networks where remote stations cannot record
        earthquakes with good SNR.
    device: string, default to 'cpu'
        If `device` is 'cpu', use the C code. If `device` is 'gpu',
        use the CUDA-C code. 
    """

    n_stations = np.int32(traces_H.shape[0])
    n_sources = np.int32(moveouts_S.shape[0]) # number of sources in the moveouts array
    if closest_stations is not None:
        n_stations_used = np.int32(closest_stations.shape[-1])
        closest_stations = np.int32(closest_stations.flatten())
        print('Number of stations used: {:d}'.format(n_stations_used))
    else:
        n_stations_used = n_stations
        #closest_stations = np.arange(n_sources * n_stations, dtype=np.int32)
        closest_stations = np.repeat(
                np.arange(n_stations, dtype=np.int32), n_sources).\
                        reshape(n_sources, n_stations, order='F').flatten()
    #--------------------------------------------------------------------------
    #-------- reshape the S moveouts array to get the number of sources -------
    #moveouts_P = moveouts_P.reshape([moveouts_P.size // n_stations,
    #                                 n_stations])
    #moveouts_S = moveouts_S.reshape([moveouts_S.size // n_stations,
    #                                 n_stations])
    if test_points is None:
        test_points = np.arange(n_sources, dtype=np.int32)
    else:
        # typically restrict the grid search to a subset of points
        print('Use the restricted grid provided by test_points.')
        test_points = np.int32(test_points)
    n_test = np.int32(test_points.size) # number of sources that are going to be visited
    n_samples = np.int32(traces_H.shape[-1])
    #---------- flatten the traces -------------
    traces_H = np.float32(traces_H.flatten())
    # ---------------------------------------------------------------------
    # uncomment the following if you want to check how the detection traces look like
    #import matplotlib.pyplot as plt
    #plt.figure('test')
    #for s in range(n_stations):
    #    plt.subplot(n_stations, 1, 1+s)
    #    plt.plot(traces_H[s*n_samples:(s+1)*n_samples])
    #plt.subplots_adjust(top=0.98, bottom=0.01)
    #plt.show()
    # ---------------------------------------------------------------------
    traces_Z = np.float32(traces_Z.flatten())
    # FLATTEN MOVEOUTS
    moveouts_P  = np.int32(moveouts_P.flatten())
    moveouts_S  = np.int32(moveouts_S.flatten())
    ntwkrsp     = np.zeros(n_samples, np.float32)
    biggest_idx = np.zeros(n_samples, np.int32)
    print('{:d} stations are used per test source, {:d} samples on each'.format(n_stations_used, n_samples))
    print('{:d} sources in the grid, {:d} test sources'.format(n_sources, n_test))
    if device == 'gpu':
        _libcu.network_response_SP(
            test_points.ctypes.data_as(C.POINTER(C.c_int)),
            traces_H.ctypes.data_as(C.POINTER(C.c_float)),
            traces_Z.ctypes.data_as(C.POINTER(C.c_float)),
            moveouts_P.ctypes.data_as(C.POINTER(C.c_int)),
            moveouts_S.ctypes.data_as(C.POINTER(C.c_int)),
            closest_stations.ctypes.data_as(C.POINTER(C.c_int)),
            n_test,
            n_samples,
            ntwkrsp.ctypes.data_as(C.POINTER(C.c_float)),
            biggest_idx.ctypes.data_as(C.POINTER(C.c_int)),
            n_stations,
            n_stations_used,
            n_sources)
    elif device == 'cpu':
        _libc.network_response_SP(
            test_points.ctypes.data_as(C.POINTER(C.c_int)),
            traces_H.ctypes.data_as(C.POINTER(C.c_float)),
            traces_Z.ctypes.data_as(C.POINTER(C.c_float)),
            moveouts_P.ctypes.data_as(C.POINTER(C.c_int)),
            moveouts_S.ctypes.data_as(C.POINTER(C.c_int)),
            closest_stations.ctypes.data_as(C.POINTER(C.c_int)),
            n_test,
            n_samples,
            ntwkrsp.ctypes.data_as(C.POINTER(C.c_float)),
            biggest_idx.ctypes.data_as(C.POINTER(C.c_int)),
            n_stations,
            n_stations_used,
            n_sources)

    return ntwkrsp, biggest_idx

def kurtosis(signal, W):
    n_stations = signal.shape[0]
    n_components = signal.shape[1]
    length = signal.shape[-1]
    Kurto = np.zeros(n_stations * n_components * length, dtype=np.float32)
    signal = np.float32(signal.flatten())
    _libc.kurtosis(signal.ctypes.data_as(C.POINTER(C.c_float)),
                   np.int32(W),
                   np.int32(n_stations),
                   np.int32(n_components),
                   np.int32(length),
                   Kurto.ctypes.data_as(C.POINTER(C.c_float)))
    return Kurto.reshape(n_stations, n_components, length)

def find_similar_sources(moveouts, threshold):
    """
    Find sources with similar moveouts so that users can discard
    some of them during the computation of the network response,
    and thus speedup the process.

    Parameters
    -------------
    moveouts : (n_sources, n_stations) array_like
    threshold : float
        The average absolute time difference tolerance to consider
        two sources as being redundant.

    Returns
    -------------
    redundant_sources : (n_sources) array_like
        Boolean numpy array with True elements for sources that
        share similar moveouts with other sources.
    """
    n_sources = moveouts.shape[0]
    n_stations = moveouts.shape[1]
    # format moveouts
    moveouts = np.float32(moveouts.flatten())
    # initialize the output pointer
    redundant_sources = np.zeros(n_sources, dtype=np.int32)
    # call the C function:
    _libc.find_similar_moveouts(moveouts.ctypes.data_as(C.POINTER(C.c_float)),
                                np.float32(threshold),
                                np.int32(n_sources),
                                np.int32(n_stations),
                                redundant_sources.ctypes.data_as(C.POINTER(C.c_int)))
    return redundant_sources.astype(np.bool)
