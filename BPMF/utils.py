import os
import sys
import glob
import numpy as np
import h5py as h5

import obspy as obs
from obspy.core import UTCDateTime as udt
from time import time as give_time

from .config import cfg
from . import dataset


# -------------------------------------------------
#              Filtering routines
# -------------------------------------------------

def bandpass_filter(X,
                    filter_order=4,
                    freqmin=cfg.min_freq,
                    freqmax=cfg.max_freq,
                    f_Nyq=cfg.sampling_rate/2.,
                    taper_alpha=0.01,
                    zerophase=True):
    """
    Parameters
    -----------
    X: (n x m) numpy array,
        Numpy array of n observations of m samples each.
        Use X.reshape(1, -1) if you want to process a
        single observation.
    filter_order: integer scalar, default to 4,
        Order/number of corners of the bandpass filter.
    freqmin: scalar float, default to cfg.min_freq,
        Low frequency cutoff.
    freqmax: scalar float, default to cfg.max_freq,
        High frequency cutoff.
    f_Nyq: scalar float, default to cfg.sampling_rate/2,
        Nyquist frequency of the data. By definition,
        the Nyquist frequency is half the sampling rate.
    taper_alpha: scalar float, default to 0.01,
        Defines the sharpness of the edges of the taper
        function. We use the tukey window function. The
        default value of 0.01 produces sharp windows.

    Returns
    --------
    filtered_X: (n x m) numpy array,
        The input array X after filtering.
    """

    from scipy.signal import iirfilter, tukey
    #from scipy.signal import lfilter
    from scipy.signal import zpk2sos, sosfilt
    from scipy.signal import detrend

    # detrend the data
    X = detrend(X, type='constant', axis=-1)
    X = detrend(X, type='linear', axis=-1)

    # design the taper function
    taper = np.repeat(tukey(X.shape[1], alpha=taper_alpha), X.shape[0])
    taper = taper.reshape(X.shape[0], X.shape[1], order='F')

    # design the filter
    #filter_num, filter_den = iirfilter(filter_order,
    #                                   [freqmin/f_Nyq, freqmax/f_Nyq],
    #                                   btype='bandpass',
    #                                   ftype='butter',
    #                                   output='ba')
    #filtered_X = lfilter(filter_num,
    #                     filter_den,
    #                     X*taper)
    ## apply the filter a second time to have a 
    ## zero phase filter
    #filtered_X = lfilter(filter_num,
    #                     filter_den,
    #                     X[::-1])[::-1]
    z, p, k = iirfilter(filter_order,
                        [freqmin/f_Nyq, freqmax/f_Nyq],
                        btype='bandpass',
                        ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    filtered_X = sosfilt(sos, X*taper)
    if zerophase:
        filtered_X = sosfilt(sos, filtered_X[:, ::-1])[:, ::-1]
    return filtered_X

def lowpass_chebyshev_I(X, freqmax, sampling_rate,
                        order=8, max_ripple=5, zerophase=False):
    from scipy.signal import cheby1, sosfilt

    nyquist = sampling_rate/2.

    sos = cheby1(order,
                 max_ripple,
                 freqmax/nyquist,
                 analog=False,
                 btype='lowpass',
                 output='sos')

    X = sosfilt(sos, X)
    if zerophase:
        X = sosfilt(sos, X[::-1])[::-1]
    return X

def lowpass_chebyshev_II(X, freqmax, sampling_rate,
                         order=10, min_attenuation_dB=40., zerophase=False):
    from scipy.signal import cheby2, sosfilt

    nyquist = sampling_rate/2.

    sos = cheby2(order,
                 min_attenuation_dB,
                 #freqmax/nyquist,
                 freqmax,
                 analog=False,
                 fs=sampling_rate,
                 btype='lowpass',
                 output='sos')

    X = sosfilt(sos, X)
    if zerophase:
        X = sosfilt(sos, X[::-1])[::-1]
    return X

# -------------------------------------------------
#       Loading travel-time data
# -------------------------------------------------

def get_moveout_array(tts, stations, phases):
    """Format the travel times into a numpy.ndarray.  

    Parameters
    -----------
    tts: dictionary
        Output of `load_travel_times`.
    stations: list of strings
        List of station names. Determine the order in which travel times are
        written in the output array.
    phases: list of strings
        List of seismic phases. Determine the order in which travel times are
        writtem in the output array.

    Returns
    ---------
    moveout_arr: (n_sources, n_stations, n_phases) numpy.ndarray
        Numpy array of travel times, in the order specified by `stations` and
        `phases`. At this stage, moveout_arr should still be in units of
        seconds.
    """
    n_stations = len(stations)
    n_phases = len(phases)
    moveout_arr = np.array([tts[ph][sta] for sta in stations for ph in phases]).T
    return moveout_arr.reshape(-1, n_stations, n_phases)

def load_travel_times(path, phases, source_indexes=None, return_coords=False):
    """Load the travel times from `path`.  

    Parameters
    ------------
    path: string
        Path to the file with the travel times.
    phases: list of strings
        List of the seismic phases to load.
    source_indexes: (n_sources,) int numpy.ndarray, default to None
        If not None, this is used to select a subset of sources from the grid.
    return_coords: boolean, default to False
        If True, also return the source coordinates.

    Returns
    ---------
    tts: dictionary
        Dictionary with one field per phase. `tts[phase_n]` is itself
        a dictionary with one field per station.
    source_coords: dictionary, optional
        Returned only if `return_coords` is True. This is a dictionary with
        three (n_sources,) numpy.ndarray: `source_coords['latitude']`,
        `source_coords['longitude']`, `source_coords['depth']`.
    """
    tts = {}
    if return_coords:
        source_coords = {}
    with h5.File(path, mode='r') as f:
        for ph in phases:
            tts[ph] = {}
            for sta in f[f'tt_{ph}'].keys():
                # flatten the lon/lat/dep grid as we work with 
                # flat source indexes
                if source_indexes is not None:
                    # select a subset of the source grid
                    tts[ph][sta] = \
                            f[f'tt_{ph}'][sta][()].flatten()[source_indexes]
                else:
                    tts[ph][sta] = f[f'tt_{ph}'][sta][()].flatten()
        if return_coords:
            for coord in f['source_coordinates'].keys():
                if source_indexes is not None:
                    source_coords[coord] = \
                            f['source_coordinates'][coord][()].flatten()[source_indexes]
                else:
                    source_coords[coord] = \
                            f['source_coordinates'][coord][()].flatten()
    if return_coords:
        return tts, source_coords
    else:
        return tts

# -------------------------------------------------
#             Stacking routines
# -------------------------------------------------

def SVDWF(matrix,
          expl_var=0.4,
          max_singular_values=5,
          freqmin=cfg.min_freq,
          freqmax=cfg.max_freq,
          sampling_rate=cfg.sampling_rate,
          wiener_filter_colsize=None):
    """
    Implementation of the Singular Value Decomposition Wiener Filter (SVDWF)
    described in Moreau et al 2017.

    Parameters
    ----------
    matrix: (n x m) numpy array
        n is the number of events, m is the number of time samples
        per event.
    n_singular_values: scalar float
        Number of singular values to retain in the
        SVD decomposition of matrix.
    max_freq: scalar float, default to cfg.max_freq
        The maximum frequency of the data, or maximum target
        frequency, is used to determined the size in the 
        time axis of the Wiener filter.

    Returns
    --------
    filtered_data: (n x m) numpy array
        The matrix filtered through the SVD procedure.
    """
    from scipy.linalg import svd
    from scipy.signal import wiener
    import matplotlib.pyplot as plt
    try:
        U, S, Vt = svd(matrix, full_matrices=False)
    except Exception as e:
        print(e)
        print('Problem while computing the svd...!')
        return np.random.normal(loc=0., scale=1., size=matrix.shape)
    if wiener_filter_colsize is None:
        wiener_filter_colsize = U.shape[0]
    #wiener_filter = [wiener_filter_colsize, int(cfg.sampling_rate/max_freq)]
    wiener_filter = [wiener_filter_colsize, 1]
    filtered_data = np.zeros((U.shape[0], Vt.shape[1]), dtype=np.float32)
    # select the number of singular values
    # in order to explain 100xn_singular_values%
    # of the variance of the matrix
    var = np.cumsum(S**2)
    if var[-1] == 0.:
        # only zeros in matrix
        return filtered_data
    var /= var[-1]
    n_singular_values = np.min(np.where(var >= expl_var)[0])+1
    n_singular_values = min(max_singular_values, n_singular_values)
    for n in range(min(U.shape[0], n_singular_values)):
        s_n = np.zeros(S.size, dtype=np.float32)
        s_n[n] = S[n]
        projection_n = np.dot(U, np.dot(np.diag(s_n), Vt))
        if wiener_filter[0] == 1 and wiener_filter[1] == 1:
            # no wiener filtering
            filtered_projection = projection_n
        else:
            # the following application of Wiener filtering is questionable: because each projection in this loop is a projection
            # onto a vector space with one dimension, all the waveforms are colinear: they just differ by an amplitude factor (but same shape).
            filtered_projection = wiener(projection_n,
                                         #mysize=[max(2, int(U.shape[0]/10)), int(cfg.sampling_rate/freqmax)]
                                         mysize=wiener_filter
                                         )
        #filtered_projection = projection_n
        if np.isnan(filtered_projection.max()):
            continue
        filtered_data += filtered_projection
    if wiener_filter[0] == 1 and wiener_filter[1] == 1:
        # no wiener filtering
        pass
    else:
        filtered_data = wiener(filtered_data,
                               mysize=wiener_filter)
    # remove nans or infs
    filtered_data[np.isnan(filtered_data)] = 0.
    filtered_data[np.isinf(filtered_data)] = 0.
    # SVD adds noise in the low and the high frequencies
    # refiltering the SVD-filtered data seems necessary
    filtered_data = bandpass_filter(filtered_data,
                                    filter_order=4,
                                    freqmin=freqmin,
                                    freqmax=freqmax,
                                    f_Nyq=sampling_rate/2.)
    return filtered_data

def fetch_detection_waveforms(tid, db_path_T, db_path_M,
                              db_path=cfg.dbpath, best_CC=False,
                              max_n_events=0, norm_rms=True,
                              ordering='correlation_coefficients',
                              flip_order=True, selection=None,
                              return_event_ids=False, unique_events=False,
                              catalog=None):

    from itertools import groupby
    from operator import itemgetter

    if catalog is None:
        cat = dataset.Catalog(f'multiplets{tid}catalog.h5', db_path_M)
        cat.read_data()
    else:
        cat = catalog
    CC = np.sort(cat.correlation_coefficients.copy())
    if max_n_events > 0:
        max_n_events = min(max_n_events, len(CC))
        CC_thres = CC[-max_n_events]
    elif best_CC:
        if len(CC) > 300:
            CC_thres = CC[-100] 
        elif len(CC) > 70:
            CC_thres = CC[int(7./10.*len(CC))] # the best 30%
        elif len(CC) > 30:
            CC_thres = np.median(CC) # the best 50%
        elif len(CC) > 10:
            CC_thres = np.percentile(CC, 33.) # the best 66% detections 
        else:
            CC_thres = 0.
    else:
        CC_thres = -1.
    if selection is None:
        selection = cat.correlation_coefficients >= CC_thres
        if unique_events:
            selection = selection & cat.unique_events
    if (np.sum(selection) == 0) and return_event_ids:
        return np.empty(0), np.empty(0), np.empty(0)
    elif (np.sum(selection) == 0):
        return np.empty(0), np.empty(0)
    else:
        pass
    filenames = cat.filenames[selection].astype('U')
    indices = cat.indices[selection]
    CCs = cat.correlation_coefficients[selection]
    event_ids = np.arange(len(cat.origin_times), dtype=np.int32)[selection]
    detection_waveforms  = []
    t1 = give_time()
    for filename, rows in groupby(zip(filenames, indices), itemgetter(0)):
        full_filename = os.path.join(db_path, db_path_M, filename+'wav.h5')
        with h5.File(full_filename, mode='r') as f:
            for row in rows:
                idx = row[1]
                detection_waveforms.append(
                        f[str(tid)]['waveforms'][idx, ...])
    detection_waveforms = np.stack(detection_waveforms, axis=0)
    if norm_rms:
        norm = np.std(detection_waveforms, axis=(2, 3))[..., np.newaxis, np.newaxis]
        norm[norm == 0.] = 1.
        detection_waveforms /= norm
    n_detections = detection_waveforms.shape[0]
    t2 = give_time()
    print('{:.2f} s to retrieve the waveforms.'.format(t2-t1))
    if ordering is not None:
        # use the requested attribute to order the detections
        if not hasattr(cat, ordering):
            print(f'The catalog does not have the {ordering} attribute, '
                  'return by chronological order.')
        else:
            order = np.argsort(getattr(cat, ordering)[selection])
            if flip_order:
                order = order[::-1]
            detection_waveforms = detection_waveforms[order, ...]
            CCs = CCs[order]
            event_ids = event_ids[order]
    if return_event_ids:
        return detection_waveforms, CCs, event_ids
    else:
        return detection_waveforms, CCs

def fetch_detection_waveforms_refilter(
        tid, db_path_T, db_path_M, net, db_path=cfg.dbpath, best_CC=False,
        max_n_events=0, norm_rms=True, freqmin=0.5, freqmax=12.0, target_SR=50.,
        integrate=False, t0='detection_time',
        ordering='correlation_coefficients', flip_order=True,
        **preprocess_kwargs):

    #sys.path.append(os.path.join(cfg.base, 'earthquake_location_eb'))
    #import relocation_utils
    from . import event_extraction

    cat = dataset.Catalog(f'multiplets{tid}catalog.h5', db_path_M)
    cat.read_data()
    CC = np.sort(cat.correlation_coefficients.copy())

    T = dataset.Template(f'template{tid}', db_path_T)
    if t0 == 'detection_time':
        correction_time = T.reference_absolute_time - cfg.buffer_extracted_events
    elif t0 == 'origin_time':
        correction_time = 0.
    else:
        print('t0 should either be detection_time or origin_time')
    # ------------------------------
    CC = np.sort(CC)
    if max_n_events > 0:
        max_n_events = min(max_n_events, len(CC))
        CC_thres = CC[-max_n_events]
    elif best_CC:
        if len(CC) > 300:
            CC_thres = CC[-100] 
        elif len(CC) > 70:
            CC_thres = CC[int(7./10.*len(CC))] # the best 30%
        elif len(CC) > 30:
            CC_thres = np.median(CC) # the best 50%
        elif len(CC) > 10:
            CC_thres = np.percentile(CC, 33.) # the best 66% detections 
        else:
            CC_thres = 0.
    else:
        CC_thres = -1.
    selection = cat.correlation_coefficients >= CC_thres
    CCs = cat.correlation_coefficients[selection]
    OTs = cat.origin_times[selection]
    detection_waveforms  = []
    t1 = give_time()
    for ot in OTs:
        # the OT in the h5 files correspond to the
        # beginning of the windows that were extracted
        # during the matched-filter search
        print('Extracting event from {}'.format(udt(ot)))
        event = event_extraction.extract_event_parallel(
                ot+correction_time,
                net, duration=cfg.multiplet_len,
                offset_start=0., folder='raw',
                attach_response=preprocess_kwargs.get('attach_response', False))
        if integrate:
            event.integrate()
        filtered_ev = event_extraction.preprocess_event(
                event,
                freqmin=freqmin, freqmax=freqmax,
                target_SR=target_SR, target_duration=cfg.multiplet_len,
                **preprocess_kwargs)
        if len(filtered_ev) > 0:
            detection_waveforms.append(get_np_array(
                                  filtered_ev, net,
                                  verbose=False))
        else:
            detection_waveforms.append(np.zeros(
                (len(net.stations), len(net.components),
                sec_to_samp(cfg.multiplet_len, sr=target_SR)), dtype=np.float32))
    detection_waveforms = np.stack(detection_waveforms, axis=0)
    if norm_rms:
        # one normalization factor for each 3-comp seismogram
        norm = np.std(detection_waveforms, axis=(2, 3))[..., np.newaxis, np.newaxis]
        norm[norm == 0.] = 1.
        detection_waveforms /= norm
    n_detections = detection_waveforms.shape[0]
    t2 = give_time()
    print('{:.2f} s to retrieve the waveforms.'.format(t2-t1))
    if ordering is not None:
        # use the requested attribute to order the detections
        if not hasattr(cat, ordering):
            print(f'The catalog does not have the {ordering} attribute, '
                  'return by chronological order.')
        else:
            order = np.argsort(getattr(cat, ordering)[selection])
            if flip_order:
                order = order[::-1]
            detection_waveforms = detection_waveforms[order, ...]
            CCs = CCs[order]
    return detection_waveforms, CCs

def SVDWF_multiplets(tid,
                     db_path=cfg.dbpath,
                     db_path_M='matched_filter_1',
                     db_path_T='template_db_1',
                     best=False,
                     norm_rms=True,
                     max_singular_values=5,
                     expl_var=0.4,
                     freqmin=cfg.min_freq,
                     freqmax=cfg.max_freq,
                     sampling_rate=cfg.sampling_rate,
                     wiener_filter_colsize=None,
                     attach_raw_data=False,
                     detection_waveforms=None):

    """
    Parameters
    -----------
    tid: scalar integer,
        Template id.
    db_path: string, default to cfg.dbpath
        Root path of the database.
    db_path_M: string, default to 'matched_filter_1'
        Name of the folder where matched-filtering results
        are stored.
    db_path_T: string, default to 'template_db_1'
        Name of the folder where template files are stored.
    best: boolean, default to False
        If True, only use the detections with higher
        correlation coefficients.
    norm_rms: boolean, default to True
        If True, individual event are RMS-normalized
        before stacking.
    max_singular_values: scalar integer, default to 5
        Disregarding how many singular values are needed
        to reconstruct 100xexp_var% of the variance of
        the detection matrix, the number of singular values
        will not be larger than max_singular_values.
    max_freq: scalar float, default to cfg.max_freq
        The maximum frequency of the data, or maximum target
        frequency, is used to determined the size in the 
        time axis of the Wiener filter.
    wiener_filter_colsize: scalar integer, default to None,
        Size of the wiener filter in the vertical direction
        (i.e. in the observation axis). If set to None,
        then it will be equal to the number of rows.
    attach_raw_data: boolean, default to False.
        If True, the data extracted during the matched-filter
        search are returned.

    Returns
    --------
    S: obspy Stream,
        Return the SVDWF-processed detections traces in the
        format of an obspy Stream. Stacked traces can be
        found in the obspy Traces, and the filtered (and raw,
        if attach_raw_data is True) data matrix is returned as
        an attribute. See also all the useful metadata in 
        the attributes.
    """

    #-----------------------------------------------------------------------------------------------
    T = dataset.Template('template{:d}'.format(tid),
                         db_path_T,
                         db_path=db_path)
    #-----------------------------------------------------------------------------------------------
    files_all = glob.glob(os.path.join(db_path,
                                       db_path_M,
                                       '*multiplets_*meta.h5'))
    files     = []
    #------------------------------
    stack = dataset.Stack(T.network_stations, T.channels,
                          sampling_rate=sampling_rate, tid=tid)
    n_stations = len(stack.stations)
    n_components = len(stack.components)
    stack.latitude  = T.latitude
    stack.longitude = T.longitude
    stack.depth     = T.depth
    #------------------------------
    if detection_waveforms is None:
        detection_waveforms, CCs = fetch_detection_waveforms(tid, db_path_T, db_path_M,
                                                             best_CC=best, norm_rms=norm_rms,
                                                             db_path=db_path)
    else:
        # provided by the user
        pass
    print('{:d} events.'.format(detection_waveforms.shape[0]))
    filtered_data = np.zeros_like(detection_waveforms)
    for s in range(n_stations):
        for c in range(n_components):
            filtered_data[:, s, c, :] = SVDWF(detection_waveforms[:, s, c, :],
                                              max_singular_values=max_singular_values,
                                              expl_var=expl_var,
                                              freqmin=freqmin,
                                              freqmax=freqmax,
                                              sampling_rate=sampling_rate,
                                              wiener_filter_colsize=wiener_filter_colsize)
            if np.sum(filtered_data[:, s, c, :]) == 0:
                print('Problem with station {} ({:d}), component {} ({:d})'.
                        format(stack.stations[s], s, stack.components[c], c))
    stacked_waveforms = np.mean(filtered_data, axis=0)
    norm = np.max(stacked_waveforms, axis=-1)[..., np.newaxis]
    norm[norm == 0.] = 1.
    stacked_waveforms /= norm
    stack.add_data(stacked_waveforms)
    stack.data = filtered_data
    if attach_raw_data:
        stack.raw_data = detection_waveforms
    stack.n_detections = detection_waveforms.shape[0]
    try:
        stack.correlation_coefficients = CCs
    except:
        stack.correlation_coefficients = 'N/A'
    return stack

# ------------------------------------
#      hierarchical clustering
# ------------------------------------

def extract_colors_from_tree(dendogram, labels, color_singleton):
    """
    Routine to build the map from cluster ids to dendogram colors.

    Parameters
    ----------
    dendogram: dendogram from scipy.hierarchy.dendogram.
    labels: (n_samples) numpy array,
        Labels, or cluster ids, returned by
        scipy.hierarchy.fcluster. For each of the initial
        n_samples samples, this function returns its
        cluster membership. labels[i] = k, means that
        data point i belongs to cluster k.
    color_singleton: string,
        Color given to the singleton when calling
        scipy.hierarchy.dendogram.

    Returns
    ---------
    cluster_colors: dictionary,
        Map between cluster id and plotting color.
        cluster_colors[k] = 'C0' means that cluster k
        is colored in 'C0'.
    """
    from itertools import groupby

    # --------------------
    list_summary = []
    for color, group in groupby(dendogram['color_list']):
        if color == color_singleton:
            continue
        elements = []
        for el in group:
            elements.append(el)
        list_summary.append([color, len(elements)])
    leaf_colors = {}
    leaf_count = 0
    cluster_count = 0
    while True:
        if leaf_count == len(dendogram['leaves']):
            break
        ev_clusterid = labels[dendogram['leaves'][leaf_count]]
        ev_cluster_size = np.sum(labels == ev_clusterid)
        if ev_cluster_size == 1:
            # next leaf is of color "color_singleton"
            leaf_colors[dendogram['leaves'][leaf_count]] = color_singleton
            leaf_count += 1
            continue
        color = list_summary[cluster_count][0]
        n_branches = list_summary[cluster_count][1]
        for j in range(leaf_count, leaf_count+n_branches+1):
            leaf_colors[dendogram['leaves'][j]] = color
            leaf_count += 1
        cluster_count += 1
    # leaf_colors should match what is plotted on the dendogram
    # we are mostly interested in the color of each cluster
    cluster_colors = {}
    for i in range(len(leaf_colors)):
        ev_id = dendogram['leaves'][i]
        cl_id = labels[ev_id]
        cluster_colors[cl_id] = leaf_colors[ev_id]
    return cluster_colors


def find_template_clusters(TpGroup, method='single', metric='correlation',
                           criterion='distance', clustering_threshold=0.33,
                           color_singleton='dimgray', ax_dendogram=None):
    """
    Find non-overlapping groups of similar templates
    with the hierarchical clustering package from scipy.
    """
    from scipy.spatial.distance import squareform
    from scipy.cluster import hierarchy
    # first, transform the CC matrix into a condensed matrix
    np.fill_diagonal(TpGroup.intertp_cc.values, 1.)
    corr_dist = squareform(1.-TpGroup.intertp_cc.values)
    if corr_dist.min() < -1.e-6:
        print('Prob with FMF')
    else:
        # avoid tiny negative values because of numerical imprecision
        corr_dist[corr_dist < 0.] = 0.
    # link the events
    Z = hierarchy.linkage(
            corr_dist, method=method, metric=metric, optimal_ordering=True)
    # get cluster labels
    labels = hierarchy.fcluster(Z, clustering_threshold, criterion=criterion)
    cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    if ax_dendogram is not None:
        # plot dendogram
        dendogram = hierarchy.dendrogram(Z, count_sort=True,
                                         above_threshold_color=color_singleton,
                                         color_threshold=clustering_threshold,
                                         ax=ax_dendogram)
        # get cluster colors from the dendogram
        cluster_colors = extract_colors_from_tree(
                dendogram, labels, color_singleton)
        ## count all singleton clusters as one, for plotting purposes
        #n_clusters = np.sum(cluster_sizes > 1)
        #if np.sum(cluster_sizes == 1) > 0:
        #        n_clusters += 1
        #        sort_by_size = np.argsort(cluster_sizes)
        return labels, cluster_ids, cluster_sizes, dendogram, cluster_colors
    else:
        return labels, cluster_ids, cluster_sizes

# -------------------------------------------------
#           Convert and round times
# -------------------------------------------------

def round_time(t, sr=cfg.sampling_rate):
    """
    Parameters
    -----------
    t: scalar float,
        Time, in seconds, to be rounded so that the number
        of meaningful decimals is consistent with the precision
        allowed by the sampling rate.
    sr: scalar float, default to cfg.sampling_rate,
        Sampling rate of the data. It is used to
        round the time.

    Returns
    --------
    t: scalar float,
        Rounded time.
    """
    # convert t to samples
    t_samp = np.int64(t*sr)
    # get it back to seconds
    t = np.float64(t_samp)/sr
    return t

def sec_to_samp(t, sr=cfg.sampling_rate, epsilon=0.2):
    """Convert seconds to samples taking into account rounding errors.  

    Parameters
    -----------
    """
    # we add epsilon so that we fall onto the right
    # integer number even if there is a small precision
    # error in the floating point number
    sign = np.sign(t)
    t_samp_float = abs(t*sr) + epsilon
    # round and restore sign
    t_samp_int = np.int64(sign*np.int64(t_samp_float))
    return t_samp_int

def time_range(start_time, end_time, dt_sec, unit='ms',
               unit_value={'ms': 1.e3, 'us': 1.e6, 'ns': 1.e9}):
    """Compute a range of datetime64.  

    Parameters
    ------------
    start_time: string or datetime
        Start of the time range.
    end_time: string or datetime
        End of the time range.
    dt_sec: scalar float
        Time step, in seconds, of the time range.
    unit: string, default to 'ms'
        Unit in which dt_sec is converted in order to reach an integer number.
    unit_value: dictionary, optional
        Dictionary with the value of 1 second in different units.

    Returns
    ---------
    time_range: (n_samples,) numpy.ndarray of numpy.datetime64
        The time range computed from the input parameters.
    """
    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    dt = np.timedelta64(int(dt_sec*unit_value[unit]), unit)
    return np.arange(start_time, end_time, dt)

# -------------------------------------------------
#                    Regression
# -------------------------------------------------

def linear_regression(x, y):
    """
    cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

    Returns
    -------
    a: slope
    b: intercept
    r_val: correlation coefficient, usually
           people use the coefficient of determination
           R**2 = r_val**2 to measure the quality of
           the fit
    p_val: two-sided p-value for a hypothesis test whose null 
           hypothesis is that the slope is zero
    std_err: standard error of the estimated slope
    """
    from scipy.stats import linregress
    a, b, r_val, p_val, std_err = linregress(x, y)
    return a, b, r_val, p_val, std_err

def weighted_linear_regression(X, Y, W=None):
    """
    Parameters
    -----------
    X: (n,) numpy array or list
    Y: (n,) numpy array or list
    W: default to None, (n,) numpy array or list
    
    Returns
    --------
    best_slope: scalar float,
        Best slope from the least square formula
    best_intercept: scalar float,
        Best intercept from the least square formula
    std_err: scalar float,
        Error on the slope
    """
    X = np.asarray(X)
    if W is None:
        W = np.ones(X.size)
    W_sum  = W.sum()
    x_mean = np.sum(W*X) / W_sum
    y_mean = np.sum(W*Y) / W_sum
    x_var  = np.sum(W*(X - x_mean)**2)
    xy_cov = np.sum(W*(X - x_mean)*(Y - y_mean))
    best_slope = xy_cov / x_var
    best_intercept = y_mean - best_slope * x_mean
    # errors in best_slope and best_intercept
    estimate = best_intercept + best_slope*X
    s2 = sum(estimate - Y)**2/(Y.size-2)
    s2_intercept = s2 * (1./X.size + x_mean**2/((X.size-1)*x_var))
    s2_slope = s2 * (1./((X.size-1)*x_var))
    return best_slope, best_intercept, np.sqrt(s2_slope)


# -------------------------------------------------
#             Others
# -------------------------------------------------

def compute_distances(source_longitudes, source_latitudes, source_depths,
                      receiver_longitudes, receiver_latitudes, receiver_depths):
    """Fast distance computation between all source points and all receivers.  

    Use `cartopy.geodesic.Geodesic` to compute pair-wise distances.

    Parameters
    ------------
    source_longitudes: (n_sources,) list or numpy.array
        Longitudes, in decimal degrees, of the source points.
    source_latitudes: (n_sources,) list or numpy.array
        Latitudes, in decimal degrees, of the source points.
    source_depths: (n_sources,) list or numpy.array
        Depths, in km, of the source points.
    receiver_longitudes: (n_sources,) list or numpy.array
        Longitudes, in decimal degrees, of the receivers.
    receiver_latitudes: (n_sources,) list or numpy.array
        Latitudes, in decimal degrees, of the receivers.
    receiver_depths: (n_sources,) list or numpy.array
        Depths, in km, of the receivers. E.g., depths are negative if receivers
        are located at the surface.
    """
    from cartopy.geodesic import Geodesic
    # convert types if necessary
    if isinstance(source_longitudes, list):
        source_longitudes = np.asarray(source_longitudes)
    if isinstance(source_latitudes, list):
        source_latitudes = np.asarray(source_latitudes)
    if isinstance(source_depths, list):
        source_depths = np.asarray(source_depths)

    # initialize distance array
    distances = np.zeros((len(source_latitudes), len(receiver_latitudes)),
                         dtype=np.float32)
    # initialize the Geodesic instance
    G = Geodesic()
    for s in range(len(receiver_latitudes)):
        epi_distances = G.inverse(
                np.array([[receiver_longitudes[s], receiver_latitudes[s]]]),
                np.hstack((source_longitudes[:, np.newaxis],
                           source_latitudes[:, np.newaxis])))
        distances[:, s] = np.asarray(epi_distances)[:, 0].squeeze()/1000.
        distances[:, s] = np.sqrt(distances[:, s]**2 \
                + (source_depths - receiver_depths[s])**2)
    return distances

def event_count(event_timings_str, start_date, end_date,
                freq='1D', offset=0., trim_start=True,
                trim_end=False, mode='end'):
    """
    Parameters
    ----------
    event_timings_str: list of array of str
        Timings of the events given as strings of characters.
    start_date: str
        Starting date of the event count time series.
    end_date: str
        End date of the event count time series.
    freq: str, default to '1D'
        Desired frequency of the event count time series.
        Default is one day.
    offset: float, default to 0.
        Fraction of the frequency used for defining
        the beginning of each bin. For example, offset=0.5
        with freq='1D' will return daily event counts
        from noon to noon.
    mode: str, default to 'end'
        Can be 'end' or 'beginning'. This string defines whether
        the seismicity counted between time 1 and time 2 is
        indexed at time 2 ('end') or time 1 ('beginning').

    Returns
    -------
    event_count: Pandas Series
        Pandas Series with temporal indexes defined
        by freq and base, and values given by the 
        event count.
        To get a numpy array from this Pandas Series,
        use: event_count.values
    """
    import pandas as pd

    start_date = pd.to_datetime(start_date.replace(',', '-'))
    end_date = pd.to_datetime(end_date.replace(',', '-'))
    offset_str = '{}{}'.format(offset, freq[-1])
    event_occurrence = pd.Series(data=np.ones(len(event_timings_str), dtype=np.int32),
                                 index=pd.to_datetime(np.asarray(event_timings_str)
                                                      .astype('U'))
                                          .astype('datetime64[ns]'))
    # trick to force a good match between initial indexes and new indexes
    event_occurrence[start_date] = 0
    event_occurrence[end_date] = 0
    if mode == 'end':
        # note: we use mode='end' so that the number of events
        # counted at time t is the event count between t-dt and t
        # this is consistent with the timing convention of pandas diff()
        # namely: diff(t) = x(t)-x(t-dt)
        event_count = event_occurrence.groupby(pd.Grouper(
            freq=freq, offset=offset_str, label='right')).agg('sum')
    elif mode == 'beginning':
        event_count = event_occurrence.groupby(pd.Grouper(
            freq=freq, offset=offset_str, label='left')).agg('sum')
    else:
        print('mode should be end or beginning')
        return
    if event_count.index[0] > pd.Timestamp(start_date):
        event_count[event_count.index[0] - pd.Timedelta(freq)] = 0
    if event_count.index[-1] < pd.Timestamp(end_date):
        event_count[event_count.index[-1] + pd.Timedelta(freq)] = 0
    if trim_start or offset==0.:
        event_count = event_count[event_count.index >= start_date]
    if trim_end or offset==0.:
        if offset > 0.:
            stop_date = pd.to_datetime(end_date) + pd.Timedelta(freq)
        else:
            stop_date = end_date
        event_count = event_count[event_count.index <= stop_date]
    # force the manually added items to be well 
    # located in time
    event_count.sort_index(inplace=True)
    return event_count

def get_np_array(stream, stations, components=['N', 'E', 'Z'],
                 priority='HH', n_samples=None,
                 component_aliases={'N': ['N', '1'],
                                    'E': ['E', '2'],
                                    'Z': ['Z']},
                 verbose=True):
    """Fetch data from Obspy Stream and returns an ndarray.  

    Parameters
    -----------
    stream: Obspy Stream instance
        The Obspy Stream instance with the waveform time series.
    stations: List of strings
        Names of the stations to include in the output array. Define the order
        of the station axis.
    components: List of strings, default to ['N','E','Z']
        Names of the components to include in the output array. Define the order
        of the component axis.
    component_aliases: Dictionary, optional
        Sometimes, components might be named differently than N, E, Z. This
        dictionary tells the function which alternative component names can be
        associated with each "canonical" component. For example,  
        `component_aliases['N'] = ['N', '1']` means that the function will also
        check the '1' component in case the 'N' component doesn't exist.
    priority: string, default to 'HH'
        When a station has multiple instruments, this string tells which
        channel to use in priority.
    n_samples: scalar int, default to None
        Duration, in samples, of the output numpy.ndarray. Select the
        `n_samples` first samples of each trace. If None, take `n_samples` as
        the length of the first trace.
    verbose: boolean, default to True
        If True, print extra output in case the target data cannot be fetched.

    Returns
    ---------
    data: (n_stations, n_components, n_samples) numpy.ndarray
        The waveform time series formatted as an numpy.ndarray.
    """
    if len(stream) == 0:
        print('The input data stream is empty!')
        return
    if n_samples is None:
        n_samples = stream[0].data.size
    data = np.zeros((len(stations), len(components), n_samples),
                    dtype=np.float32)
    for s, sta in enumerate(stations):
        for c, cp in enumerate(components):
            for cp_alias in component_aliases[cp]:
                channel = stream.select(station=sta, component=cp_alias)
                if len(channel) > 0:
                    # succesfully retrieved data
                    break
            if len(channel) > 0:
                try:
                    # try selecting the preferred channel if it exists
                    cha = channel.select(
                            channel=f'{priority}{cp_alias}')[0]
                    #data[s, c, :] = channel.select(
                    #        channel=f'{priority}{cp_alias}')[0].data[:n_samples]
                except IndexError:
                    cha = channel[0]
                    #data[s, c, :] = channel[0].data[:n_samples]
                if len(cha.data) < n_samples:
                    length_diff = n_samples - len(cha.data)
                    data[s, c, :] = np.hstack((cha.data, np.zeros(length_diff,
                        dtype=np.float32)))
                else:
                    data[s, c, :] = cha.data[:n_samples]
    return data

def max_norm(X):
    max_ = np.abs(X).max()
    if max_ != 0.:
        return X/max_
    else:
        return X

def running_mad(time_series,
                window,
                n_mad=10.,
                overlap=0.75):
    from scipy.stats import median_abs_deviation as scimad
    # calculate n_windows given window
    # and overlap
    shift = int((1.-overlap)*window)
    n_windows = int((len(time_series)-window)//shift)+1
    mad_ = np.zeros(n_windows+2, dtype=np.float32)
    med_ = np.zeros(n_windows+2, dtype=np.float32)
    time = np.zeros(n_windows+2, dtype=np.float32)
    for i in range(1, n_windows+1):
        i1 = i*shift
        i2 = min(len(time_series), i1+window)
        sliding_window = time_series[i1:i2]
        #non_zero = cnr_window != 0
        #if sum(non_zero) < 3:
        #    # won't be possible to calculate median
        #    # and mad on that few samples
        #    continue
        #med_[i] = np.median(cnr_window[non_zero])
        #mad_[i] = scimad(cnr_window[non_zero])
        med_[i] = np.median(sliding_window)
        mad_[i] = scimad(sliding_window)
        time[i] = (i1+i2)/2.
    # add boundary cases manually
    time[0] = 0.
    mad_[0] = mad_[1]
    med_[0] = med_[1]
    time[-1] = len(time_series)
    mad_[-1] = mad_[-2]
    med_[-1] = med_[-2]
    running_stat = med_ + n_mad * mad_
    interpolator = interp1d(time,
                            running_stat,
                            kind='slinear',
                            fill_value=(running_stat[0], running_stat[-1]),
                            bounds_error=False)
    full_time = np.arange(0, len(time_series))
    running_stat = interpolator(full_time)
    return running_stat


def two_point_distance(lon_1, lat_1, depth_1,
                       lon_2, lat_2, depth_2):
    """Compute the distance between two points.  


    Parameters
    -----------
    lon_1: scalar, float
        Longitude of Point 1.
    lat_1: scalar, float
        Latitude of Point 1.
    depth_1: scalar, float
        Depth of Point 1 (in km).
    lon_2: scalar, float
        Longitude of Point 2.
    lat_2: scalar, float
        Latitude of Point 2.
    depth_2: scalar, float
        Depth of Point 2 (in km).

    Returns
    ---------
    dist: scalar, float
        Distance between Point 1 and Point 2 in kilometers.
    """

    from obspy.geodetics.base import calc_vincenty_inverse

    dist, az, baz = calc_vincenty_inverse(lat_1, lon_1, lat_2, lon_2)
    dist /= 1000. # from m to km
    dist = np.sqrt(dist**2 + (depth_1 - depth_2)**2)
    return dist

