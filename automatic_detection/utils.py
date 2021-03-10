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
from . import db_h5py


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
#      Earthquake catalog routines
# -------------------------------------------------

def load_earthquake_catalog(tids, 
                            db_path_T='template_db_1',
                            db_path_M='matched_filter_1',
                            db_path=cfg.dbpath,
                            remove_multiples=True):

    # python list version
    OT = []
    OT_string = []
    latitudes         = []
    longitudes        = []
    depths            = []
    dLongitude        = []
    dLatitude         = []
    dDepth            = []
    mining_activity   = []
    stations          = []
    S_travel_times    = []
    P_travel_times    = []
    cat_tids          = []
    Ntot = 0
    for tid in tids:
        try:
            origin_times  = db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid),
                                                            db_path=db_path,
                                                            db_path_M=db_path_M,
                                                            object_from_cat='origin_times')
        except OSError:
            print('Couldn\'t find the catalog file for template {:d}'.format(tid))
            continue
        unique_events = db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid),
                                                        db_path=db_path,
                                                        db_path_M=db_path_M,
                                                        object_from_cat='unique_events')
        network_stations = db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid),
                                                           db_path=db_path,
                                                           db_path_M=db_path_M,
                                                           object_from_cat='stations')
        template = db_h5py.read_template('template{:d}'.format(tid),
                                         db_path=db_path,
                                         db_path_T=db_path_T)
        if remove_multiples:
            mask = unique_events
        else:
            mask = np.ones(origin_times.size, dtype=np.bool)
        print('Template {:d}: {:d}/{:d} events'.format(tid, mask.sum(), mask.size))
        n_events = mask.sum()
        origin_times = origin_times[mask]
        OT.extend(origin_times.tolist())
        OT_string.extend([str(udt(origin_times[k])) for k in range(len(origin_times))])
        latitudes.extend([template.metadata['latitude']] * n_events)
        longitudes.extend([template.metadata['longitude']] * n_events)
        depths.extend([template.metadata['depth']] * n_events)
        dLongitude.extend([template.metadata['cov_mat'][0, 0]] * n_events)
        dLatitude.extend([template.metadata['cov_mat'][1, 1]] * n_events)
        dDepth.extend([template.metadata['cov_mat'][2, 2]] * n_events)
        cat_tids.extend([template.metadata['template_idx']] * n_events)
        st_list = template.metadata['stations'].tolist()
        for i in range(n_events):
            stations.append(st_list)
        # convert network stations to list
        network_stations = network_stations.astype('U').tolist()
        station_indexes = np.int32([network_stations.index(st) for st in st_list])
        p_travel_times = template.metadata['travel_times'][station_indexes, 0]
        p_travel_times = np.repeat(p_travel_times, n_events).reshape( (n_events, p_travel_times.size), order='F' )
        s_travel_times = template.metadata['travel_times'][station_indexes, 1]
        s_travel_times = np.repeat(s_travel_times, n_events).reshape( (n_events, s_travel_times.size), order='F' )
        P_travel_times.extend(p_travel_times)
        S_travel_times.extend(s_travel_times)

    if len(OT) == 0:
        print('No events were found, return None.')
        return None
    
    # python list version: convert to numpy arrays
    OT                = np.asarray(OT).astype(np.float64)
    OT_string         = np.asarray(OT_string).astype('S')
    latitudes         = np.asarray(latitudes).astype(np.float32)
    longitudes        = np.asarray(longitudes).astype(np.float32)
    depths            = np.asarray(depths).astype(np.float32)
    dLongitude        = np.asarray(dLongitude).astype(np.float32)
    dLatitude         = np.asarray(dLatitude).astype(np.float32)
    dDepth            = np.asarray(dDepth).astype(np.float32)
    mining_activity   = np.asarray(mining_activity).astype('bool')
    stations          = np.asarray(stations).astype('|S8')
    S_travel_times    = np.asarray(S_travel_times).astype(np.float32)
    P_travel_times    = np.asarray(P_travel_times).astype(np.float32)
    cat_tids          = np.asarray(cat_tids).astype(np.int32)
    
    # sort by chronological order
    I                 = np.argsort(OT)
    OT                = OT[I]
    OT_string         = OT_string[I]
    latitudes         = latitudes[I]
    longitudes        = longitudes[I]
    depths            = depths[I]
    dLongitude        = dLongitude[I]
    dLatitude         = dLatitude[I]
    dDepth            = dDepth[I]
    stations          = stations[I]
    P_travel_times    = P_travel_times[I, :]
    S_travel_times    = S_travel_times[I, :]
    cat_tids          = cat_tids[I]
    
    catalog = {}
    catalog['origin_times']           = OT
    catalog['origin_times_strings']   = OT_string
    catalog['latitudes']              = latitudes 
    catalog['longitudes']             = longitudes
    catalog['depths']                 = depths
    catalog['latitude_uncertainty']   = dLatitude
    catalog['longitude_uncertainty']  = dLongitude
    catalog['depth_uncertainty']      = dDepth
    catalog['p_travel_times']         = P_travel_times 
    catalog['s_travel_times']         = S_travel_times
    catalog['stations']               = stations
    catalog['template_ids']           = cat_tids

    return catalog

def load_light_earthquake_catalog(tids, 
                                  db_path_T='template_db_1',
                                  db_path_M='matched_filter_1',
                                  db_path=cfg.dbpath,
                                  read_CCs=False,
                                  remove_multiples=True):

    # python list version
    OT = []
    OT_string = []
    latitudes         = []
    longitudes        = []
    depths            = []
    dLongitude        = []
    dLatitude         = []
    dDepth            = []
    cat_tids          = []
    if read_CCs:
        CCs = []
    Ntot = 0
    for tid in tids:
        try:
            origin_times  = db_h5py.read_catalog_multiplets(
                    'multiplets{:d}'.format(tid),
                     db_path=db_path,
                     db_path_M=db_path_M,
                     object_from_cat='origin_times')
            if read_CCs:
                correlation_coefficients = db_h5py.read_catalog_multiplets(
                        'multiplets{:d}'.format(tid),
                        db_path=db_path,
                        db_path_M=db_path_M,
                        object_from_cat='correlation_coefficients')
        except OSError:
            print('Couldn\'t find the catalog file for template {:d}'.format(tid))
            continue
        if remove_multiples:
            unique_events = db_h5py.read_catalog_multiplets('multiplets{:d}'.format(tid),
                                                            db_path=db_path,
                                                            db_path_M=db_path_M,
                                                            object_from_cat='unique_events')
        template = dataset.Template('template{:d}'.format(tid),
                                    db_path_T, db_path=db_path)
        if remove_multiples:
            mask = unique_events
        else:
            mask = np.ones(origin_times.size, dtype=np.bool)
        print('Template {:d}: {:d}/{:d} events'.format(tid, mask.sum(), mask.size))
        n_events = mask.sum()
        origin_times = origin_times[mask]
        if read_CCs:
            correlation_coefficients[mask]
            CCs.extend(correlation_coefficients.tolist())
        OT.extend(origin_times.tolist())
        OT_string.extend([str(udt(origin_times[k])) for k in range(len(origin_times))])
        latitudes.extend([template.latitude] * n_events)
        longitudes.extend([template.longitude] * n_events)
        depths.extend([template.depth] * n_events)
        dLongitude.extend([np.sqrt(template.cov_mat[0, 0])] * n_events)
        dLatitude.extend([np.sqrt(template.cov_mat[1, 1])] * n_events)
        dDepth.extend([np.sqrt(template.cov_mat[2, 2])] * n_events)
        cat_tids.extend([template.template_idx] * n_events)

    if len(OT) == 0:
        print('No events were found, return None.')
        return None
    
    # python list version: convert to numpy arrays
    OT                = np.asarray(OT).astype(np.float64)
    OT_string         = np.asarray(OT_string).astype('S')
    latitudes         = np.asarray(latitudes).astype(np.float32)
    longitudes        = np.asarray(longitudes).astype(np.float32)
    depths            = np.asarray(depths).astype(np.float32)
    dLongitude        = np.asarray(dLongitude).astype(np.float32)
    dLatitude         = np.asarray(dLatitude).astype(np.float32)
    dDepth            = np.asarray(dDepth).astype(np.float32)
    cat_tids          = np.asarray(cat_tids).astype(np.int32)
    
    # sort by chronological order
    I                 = np.argsort(OT)
    OT                = OT[I]
    OT_string         = OT_string[I]
    latitudes         = latitudes[I]
    longitudes        = longitudes[I]
    depths            = depths[I]
    dLongitude        = dLongitude[I]
    dLatitude         = dLatitude[I]
    dDepth            = dDepth[I]
    cat_tids          = cat_tids[I]
    if read_CCs:
        CCs = np.asarray(CCs).astype(np.float32)
        CCs = CCs[I]
   
    catalog = {}
    catalog['origin_times']           = OT
    catalog['origin_times_strings']   = OT_string
    catalog['latitudes']              = latitudes 
    catalog['longitudes']             = longitudes
    catalog['depths']                 = depths
    catalog['latitude_uncertainty']   = dLatitude
    catalog['longitude_uncertainty']  = dLongitude
    catalog['depth_uncertainty']      = dDepth
    catalog['template_ids']           = cat_tids
    if read_CCs:
        catalog['correlation_coefficients'] = CCs
    return catalog

def event_count(event_timings_str,
                start_date,
                end_date,
                freq='1D',
                offset=0.,
                trim_start=True,
                trim_end=False,
                mode='end'):
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
        event_count = event_occurrence.groupby(pd.Grouper(freq=freq, offset=offset_str, label='right')).agg('sum')
    elif mode == 'beginning':
        event_count = event_occurrence.groupby(pd.Grouper(freq=freq, offset=offset_str, label='left')).agg('sum')
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

def fetch_detection_waveforms_old(tid,
                              db_path_T,
                              db_path_M,
                              db_path=cfg.dbpath,
                              best_CC=False,
                              max_n_events=0,
                              norm_rms=True):

    files_all = glob.glob(os.path.join(db_path,
                                       db_path_M,
                                       '*multiplets_*meta.h5'))
    files     = []
    #------------------------------
    CC = []
    valid = []
    tid_str = str(tid)
    t1 = give_time()
    for file_ in files_all:
        with h5.File(file_, mode='r') as f:
            if tid_str in f.keys():
                files.append(file_[:-len('meta.h5')])
                CC.extend(f[tid_str]['correlation_coefficients'][()].tolist())
    CC = np.float32(CC)
    t2 = give_time()
    print('{:.2f} s to retrieve the correlation coefficients.'.format(t2-t1))
    if len(files) == 0:
        print("None multiplet for template {:d} !! Return None".format(tid))
        return None
    #------------------------------
    #----------------------------------------------
    CC = np.sort(CC)
    if max_n_events > 0:
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
    detection_waveforms  = []
    CCs = []
    t1 = give_time()
    for file_ in files:
        with h5.File(file_ + 'meta.h5', mode='r') as fm:
            if tid_str not in fm.keys():
                continue
            selection = np.where(fm[tid_str]['correlation_coefficients'][:] >= CC_thres)[0]
            if selection.size == 0:
                continue
            else:
                CCs.extend(fm[tid_str]['correlation_coefficients'][selection])
        with h5.File(file_ + 'wav.h5', mode='r') as fw:
            if tid_str not in fw.keys():
                continue
            detection_waveforms.append(fw[tid_str]['waveforms'][selection, :, :, :])
    detection_waveforms = np.vstack(detection_waveforms)
    if norm_rms:
        norm = np.std(detection_waveforms, axis=(2, 3))[..., np.newaxis, np.newaxis]
        norm[norm == 0.] = 1.
        detection_waveforms /= norm
    n_detections = detection_waveforms.shape[0]
    t2 = give_time()
    print('{:.2f} s to retrieve the waveforms.'.format(t2-t1))
    # reorder waveforms
    CCs = np.float32(CCs)
    new_order = np.argsort(CCs)[::-1]
    detection_waveforms = detection_waveforms[new_order, ...]
    CCs = CCs[new_order]
    return detection_waveforms, CCs

def fetch_detection_waveforms(tid, db_path_T, db_path_M,
                              db_path=cfg.dbpath, best_CC=False,
                              max_n_events=0, norm_rms=True,
                              ordering='correlation_coefficients',
                              flip_order=True, selection=None,
                              return_event_ids=False):

    from itertools import groupby
    from operator import itemgetter

    cat = dataset.Catalog(f'multiplets{tid}catalog.h5', db_path_M)
    cat.read_data()
    CC = np.sort(cat.correlation_coefficients.copy())
    if max_n_events > 0:
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
        selection = cat.correlation_coefficients > CC_thres
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
        integrate=False, ordering='correlation_coefficients', flip_order=True):

    #sys.path.append(os.path.join(cfg.base, 'earthquake_location_eb'))
    #import relocation_utils
    from . import event_extraction

    cat = dataset.Catalog(f'multiplets{tid}catalog.h5', db_path_M)
    cat.read_data()
    CC = np.sort(cat.correlation_coefficients.copy())

    T = dataset.Template(f'template{tid}', db_path_T)
    correction_time = T.reference_absolute_time - cfg.buffer_extracted_events
    # ------------------------------
    CC = np.sort(CC)
    if max_n_events > 0:
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
    selection = cat.correlation_coefficients > CC_thres
    CCs = cat.correlation_coefficients[selection]
    OTs = cat.origin_times[selection]
    detection_waveforms  = []
    t1 = give_time()
    for ot in OTs:
        # the OT in the h5 files correspond to the
        # beginning of the windows that were extracted
        # during the matched-filter search
        event = event_extraction.extract_event_parallel(
                                       ot+correction_time,
                                       net, duration=cfg.multiplet_len,
                                       offset_start=0., folder='raw')
        if integrate:
            event.integrate()
        filtered_ev = event_extraction.preprocess_event(event,
                                                        freqmin=freqmin,
                                                        freqmax=freqmax,
                                                        target_SR=target_SR)
        detection_waveforms.append(get_np_array(
                              filtered_ev, net,
                              verbose=False))
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
    t_samp = np.int32(t*sr)
    # get it back to seconds
    t = np.float32(t_samp)/sr
    return t

def sec_to_samp(t, sr=cfg.sampling_rate,
                epsilon=0.2):
    # we add epsilon so that we fall onto the right
    # integer number even if there is a small precision
    # error in the floating point number
    sign = np.sign(t)
    t_samp_float = abs(t*sr) + epsilon
    # round and restore sign
    t_samp_int = np.int32(sign*np.int32(t_samp_float))
    return t_samp_int


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

def get_np_array(stream,
                 net,
                 verbose=True):
    n_stations = len(net.stations)
    n_components = len(net.components)
    if len(stream) == 0:
        print('The input data stream is empty!')
        return
    n_samples = stream[0].data.size
    data = np.zeros((n_stations, n_components, n_samples),
                    dtype=np.float32)
    for s, sta in enumerate(net.stations):
        for c, cp in enumerate(net.components):
            try:
                data[s, c, :] = stream.select(station=sta, component=cp)[0].data
            except Exception as e:
                if verbose:
                    print(e)
                    print('Leave blank in the data')
                continue
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


def two_point_distance(lat_1, long_1, depth_1,
                       lat_2, long_2, depth_2):
    """
    Parameters
    -----------
    lat_1: float,
        Latitude of Point 1.
    lon_1: float,
        Longitude of Point 1.
    depth_1: float,
        Depth of Point 1 (in km).
    lat_2: float,
        Latitude of Point 2.
    lon_2: float,
        Longitude of Point 2.
    depth_2: float,
        Depth of Point 2 (in km).

    Returns
    ---------
    dist: float,
        Distance between Point 1 and Point 2 in kilometers.
    """

    from obspy.geodetics.base import calc_vincenty_inverse

    dist, az, baz = calc_vincenty_inverse(lat_1, long_1, lat_2, long_2)
    dist /= 1000. # from m to km
    dist = np.sqrt(dist**2 + (depth_1 - depth_2)**2)
    return dist

