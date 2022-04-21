import os

import ctypes as C
import numpy as np
import datetime as dt
from .config import package

cpu_loaded = False

libpath = os.path.join(package, 'lib')

try:
    _libc = C.cdll.LoadLibrary(os.path.join(libpath, 'libc.so'))
    cpu_loaded = True
except:
    print('Missing libc.so! You won''t be able to run multiplet/template searches on the CPU')
    print('Should be at {}libc.so'.format(os.path.join(libpath, '')))


if cpu_loaded:
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

def find_similar_sources(moveouts, threshold, n_nearest_neighbors=200):
    """
    Find sources with similar moveouts so that users can discard
    some of them during the computation of the network response,
    and thus speedup the process.

    Parameters
    -------------
    moveouts: (n_sources, n_stations) float numpy.ndarray
        The moveouts in seconds. Note: It makes more sense to input the moveouts
        rather than the absolute travel times here.
    threshold: scalar float
        The station average time difference tolerance to consider
        two sources as being redundant.
    n_nearest_neighbors: scalar int, default to 200
        Check redundancy in chunks of `n_nearest_neighbors` contiguous sources
        before starting the all pair-wise computation. If the moveouts are
        ordered such that contiguous elements correspond to proximal grid
        sources, then this can considerably speed up the computation.

    Returns
    -------------
    redundant_sources: (n_sources,) boolean numpy.ndarray 
        Boolean numpy array with True elements for sources that
        share similar moveouts with other sources.
    """
    n_sources = moveouts.shape[0]
    n_stations = moveouts.shape[1]
    if moveouts.dtype in (np.int32, np.int64):
        print('Integer typed moveouts detected. Are you sure these are in'
              ' seconds?')
    # format moveouts
    moveouts = np.float32(moveouts.flatten())
    # initialize the output pointer
    redundant_sources = np.zeros(n_sources, dtype=np.int32)
    n_nearest_neighbors = min(n_sources, n_nearest_neigbors)
    # call the C function:
    _libc.find_similar_moveouts(moveouts.ctypes.data_as(C.POINTER(C.c_float)),
                                np.float32(threshold),
                                int(n_sources),
                                int(n_stations),
                                int(n_nearest_neighbors),
                                redundant_sources.ctypes.data_as(C.POINTER(C.c_int)))
    return redundant_sources.astype(np.bool)
