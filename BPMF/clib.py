import os

import ctypes as C
import numpy as np
import datetime as dt
from .config import cfg

cpu_loaded = False

libpath = os.path.join(cfg.PACKAGE, "lib")

try:
    _libc = C.cdll.LoadLibrary(os.path.join(libpath, "libc.so"))
    cpu_loaded = True
except:
    print(
        "Missing libc.so! You won"
        "t be able to run multiplet/template searches on the CPU"
    )
    print("Should be at {}libc.so".format(os.path.join(libpath, "")))


if cpu_loaded:
    _libc.kurtosis.argtypes = [
        C.POINTER(C.c_float),
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_int,
        C.POINTER(C.c_float),
    ]

    _libc.find_similar_moveouts.argtypes = [
        C.POINTER(C.c_float),  # moveouts
        C.POINTER(C.c_float),  # source_longitude
        C.POINTER(C.c_float),  # source_latitude
        C.POINTER(C.c_float),  # cell_longitude
        C.POINTER(C.c_float),  # cell_latitude
        C.c_float,  # rms time difference threshold
        C.c_size_t,  # number of grid points
        C.c_size_t,  # number of stations
        C.c_size_t,  # number of cells in longitude
        C.c_size_t,  # number of cells in latitude
        C.c_size_t,  # number of stations for diff
        C.c_int, # num threads
        C.POINTER(C.c_int),  # output pointer: redundant sources
    ]


    _libc.select_cc_indexes.argtypes = [
        C.POINTER(C.c_float),  # CCs
        C.POINTER(C.c_float),  # threshold
        C.c_size_t,  # search window
        C.c_size_t,  # length of CCs
        C.POINTER(C.c_int),  # selected corr
    ]


def kurtosis(signal, W):
    n_stations = signal.shape[0]
    n_components = signal.shape[1]
    length = signal.shape[-1]
    Kurto = np.zeros(n_stations * n_components * length, dtype=np.float32)
    signal = np.float32(signal.flatten())
    _libc.kurtosis(
        signal.ctypes.data_as(C.POINTER(C.c_float)),
        np.int32(W),
        np.int32(n_stations),
        np.int32(n_components),
        np.int32(length),
        Kurto.ctypes.data_as(C.POINTER(C.c_float)),
    )
    return Kurto.reshape(n_stations, n_components, length)


def find_similar_sources(
        moveouts,
        source_longitude,
        source_latitude,
        cell_longitude,
        cell_latitude,
        threshold,
        num_threads=None,
        num_stations_for_diff=None,
        ):
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

    Returns
    -------------
    redundant_sources: (n_sources,) boolean numpy.ndarray
        Boolean numpy array with True elements for sources that
        share similar moveouts with other sources.
    """
    n_sources = moveouts.shape[0]
    n_stations = moveouts.shape[1]
    n_cells_longitude = len(cell_longitude) - 1
    n_cells_latitude = len(cell_latitude) - 1

    if num_stations_for_diff is None:
        num_stations_for_diff = n_stations

    if moveouts.dtype in (np.int32, np.int64):
        print("Integer typed moveouts detected. Are you sure these are in" " seconds?")

    if num_threads is None:
        # set num_threads to -1 so that the C routine
        # understands to use all CPUs
        #num_threads = os.cpu_count()
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))

    # format input arrays
    moveouts = np.float32(moveouts.flatten())
    source_longitude = np.float32(source_longitude)
    source_latitude = np.float32(source_latitude)
    cell_longitude = np.float32(cell_longitude)
    cell_latitude = np.float32(cell_latitude)

    # initialize the output pointer
    redundant_sources = np.zeros(n_sources, dtype=np.int32)

    # call the C function:
    _libc.find_similar_moveouts(
        moveouts.ctypes.data_as(C.POINTER(C.c_float)),
        source_longitude.ctypes.data_as(C.POINTER(C.c_float)),
        source_latitude.ctypes.data_as(C.POINTER(C.c_float)),
        cell_longitude.ctypes.data_as(C.POINTER(C.c_float)),
        cell_latitude.ctypes.data_as(C.POINTER(C.c_float)),
        np.float32(threshold),
        int(n_sources),
        int(n_stations),
        int(n_cells_longitude),
        int(n_cells_latitude),
        int(num_stations_for_diff),
        int(num_threads),
        redundant_sources.ctypes.data_as(C.POINTER(C.c_int)),
    )
    return redundant_sources.astype(bool)


def select_cc_indexes(ccs, threshold, search_win):
    """Select new event detection's correlation indexes.

    Parameters
    -----------
    ccs: (n_corr,) `numpy.ndarray`
        Time series of correlation coefficients.
    threshold (n_corr,) `numpy.ndarray` or `float` scalar
        Time series or scalar detection threshold.
    search_win: `int` scalar
        Size of the time window, in number of consecutive correlations, defining
        grouped detections.

    Returns
    --------
    selection: (n_corr,) bool `numpy.ndarray`
        Vector of `n_corr` booleans that are true if the corresponding CC index
        is a new event detection.
    """
    n_corr = len(ccs)
    if isinstance(threshold, float) or isinstance(threshold, int):
        threshold = threshold * np.ones(n_corr, dtype=np.float32)
    threshold = np.float32(threshold)
    selection = np.zeros(n_corr, dtype=np.int32)
    _libc.select_cc_indexes(
        ccs.ctypes.data_as(C.POINTER(C.c_float)),
        threshold.ctypes.data_as(C.POINTER(C.c_float)),
        int(search_win),
        int(n_corr),
        selection.ctypes.data_as(C.POINTER(C.c_int)),
    )
    return selection.astype(bool)
