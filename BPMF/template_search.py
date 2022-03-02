import os.path
import sys

from .config import cfg
from . import common as cmn
from . import data, clib, db_h5py, utils

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

try:
    from scipy.stats import median_abs_deviation as scimad
except ImportError:
    from scipy.stats import median_absolute_deviation as scimad
#from scipy.stats import median_abs_deviation as scimad
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import hilbert

import pickle

from time import time as give_time

from math import isnan

from obspy.core import UTCDateTime as udt

class NetworkResponse():
    """Result from calc_network_response.
    """
    
    def __init__(self,
                 stations,
                 components,
                 sampling_rate,
                 date):

        self.stations = stations
        self.components = components
        self.sampling_rate = sampling_rate
        self.date = date

    def remove_baseline(self,
                        window,
                        attribute='composite'):
        
        # convert window from seconds to samples
        window = int(window*self.sampling_rate)
        attr_baseline = self._baseline(getattr(self, attribute),
                                       window)
        setattr(self, attribute, getattr(self, attribute)-attr_baseline)

    def return_pd_series(self, attribute='composite'):
        import pandas as pd
        data = getattr(self, attribute)
        indexes = pd.date_range(start=str(self.date),
                                freq='{}S'.format(1./self.sampling_rate),
                                periods=len(data))
        pd_attr = pd.Series(data=data,
                            index=indexes)
        return pd_attr

    def smooth_cnr(self,
                   window):

        from scipy.ndimage.filters import gaussian_filter1d
        # convert window from seconds to samples
        window = int(window*self.sampling_rate)
        self.smoothed = gaussian_filter1d(self.composite, window)


    def _baseline(self, X, window):
        n_windows = np.int32(np.ceil(X.size/window))
        minima      = np.zeros(n_windows, dtype=X.dtype)
        minima_args = np.zeros(n_windows, dtype=np.int32)
        for i in range(n_windows):
            minima_args[i] = i*window + X[i*window:(i+1)*window].argmin()
            minima[i] = X[minima_args[i]]
        #----------------------------------------
        #--------- build interpolation ----------
        interpolator = interp1d(minima_args,
                                minima,
                                kind='linear',
                                fill_value='extrapolate')
        bline = interpolator(np.arange(X.size))
        return bline


def calc_network_response(data,
                          moveouts,
                          phase='SP',
                          device='gpu', 
                          n_closest_stations=None, 
                          envelopes=True, 
                          test_points=None,
                          saturation=False):
    """
    Calculate the composite network response from Frank et al. 2014
    and Beauce et al. 2019. This systematic backprojection of seismic
    energy detect earthquakes and get an approximative
    idea of where there are located. Seismic energy is not a good
    enough feature for earthquake location, and the location uncertainties
    might be very large.

    Parameters
    -----------
    data: dictionary,
        Dictionary with two fields: 'waveforms', which is an
        (n_stations x n_components x n_samples) numpy array, and
        'metadata', which is a dictionary containing important metadata
        such as the station list, the component list, and the sampling rate.
    moveouts: object,
        Moveout object from automatic_detection.moveouts.
    phase: string, default to 'SP'
        Should be left to 'SP' for now. It means both the P and S
        times will be used to backproject the seismic energy.
    device: string, default to 'gpu'
        Can be 'gpu' or 'cpu'. Determine whether the composite
        network response is computed on CPUs or GPUs.
    n_closest_stations: None or integer, default to None
        If not None, is the number of stations closest to
        each point of the test source grid that are used
        for backprojected the seismic energy.
    envelopes: boolean, default to True
        Whether or not the envelope of the traces is given
        for computing the composite network response.
        The envelope is the modulus of the analytical signal.
    test_points: None or numpy array of integers, default to None
        If not None, the test points are the test sources that are
        used to backproject the seismic energy. This is useful for
        limiting the computation to a subset of the grid.
    saturation: boolean, default to False
        Whether or not the traces are clipped.

    Returns
    --------
    network_response: object,
        A composite network response object with multiple attributes
        calculated by the backprojection of seismic energy.
    """

    ANOMALY_THRESHOLD = 1.e-5 # threshold used to determine if a trace is garbage or not
    MAX_DYNAMIC_RANGE = 5 # if saturation is True, traces are clipped above MADx10^MAX_DYNAMIC_RANGE
    # depends on which unit the trace is
    stations   = data['metadata']['stations']
    components = data['metadata']['components']
    if isinstance(stations, str):
        stations = [stations]
    if isinstance(components, str):
        components = [components]

    traces = np.array(data['waveforms'], copy=True)
    #-----------------------------
    n_stations   = traces.shape[0]
    n_components = traces.shape[1]
    n_samples    = traces.shape[2]
    #-----------------------------

    # Initialize the network response object
    network_response = NetworkResponse(stations,
                                       components,
                                       cfg.sampling_rate,
                                       data['metadata']['date'])

    if phase in ('p', 'P'):
        print('Use the P-wave moveouts to compute the Composite Network Response')
        moveout = moveouts.p_relative_samp
    elif phase in ('s', 'S'):
        print('Use the S-wave moveouts to compute the Composite Network Response')
        moveout = moveouts.s_relative_samp
    elif phase in ('sp', 'SP'):
        print('Use the P- and S-wave moveouts to compute the Composite Network Response')
        moveoutS = moveouts.s_relative_p_samp
        moveoutP = moveouts.p_relative_samp
    
    smooth_win = cmn.to_samples(cfg.smooth, data['metadata']['sampling_rate']) 
    data_availability = np.zeros(n_stations, dtype=np.int32)

    if envelopes:
        window_length = utils.sec_to_samp(cfg.template_len)
        start = give_time()
        detection_traces = envelope_parallel(traces) # take the upper envelope of the traces
        end = give_time()
        print('Computed the envelopes in {:.2f}sec.'.format(end-start))
        #start = give_time()
        #ANOMALY_THRESHOLD = cmn.mad(detection_traces[detection_traces != 0.]) / 1.e5
        #end = give_time()
        #print('{:.2f}s to compute empirically the anomaly threshold (={:.2e})'.format(end-start, ANOMALY_THRESHOLD))
        for s in range(n_stations):
            for c in range(n_components):
                missing_samples = detection_traces[s, c, :] == 0.
                if np.sum(missing_samples) > detection_traces.shape[-1]/2:
                    continue
                median = np.median(detection_traces[s, c, ~missing_samples])
                mad = scimad(detection_traces[s, c, ~missing_samples])
                if mad < ANOMALY_THRESHOLD:
                    detection_traces[s, c, :] = 0.
                    continue
                detection_traces[s, c, :] = (detection_traces[s, c, :] - median) / mad
                detection_traces[s, c, missing_samples] = 0.
                data_availability[s] += 1
    else:
        # compute the daily MADs (Median Absolute Deviation) to normalize the traces
        # this is an empirical way of correcting for instrument's sensitivity
        MADs = np.zeros( (n_stations, n_components), dtype=np.float32)
        for s in range(n_stations):
            for c in range(n_components):
                traces[s, c, :] -= np.median(traces[s, c, :])
                mask = traces[s, c, :] != 0.
                if np.sum(mask) == 0:
                    continue
                mad = scimad(traces[s, c, mask])
                MADs[s, c] = np.float32(mad)
                if MADs[s, c] != 0.:
                    traces[s, c, :] /= MADs[s, c]
                    data_availability[s] += 1
        detection_traces = np.square(traces)

    # we consider data to be available if more than 1 channel were operational
    data_availability = data_availability > 1
    network_response.data_availability = data_availability
    print('{:d} / {:d} available stations'.format(data_availability.sum(), data_availability.size))
    if data_availability.sum() < data_availability.size//2:
        print('Less than half the stations are available, pass!')
        network_response.success = False
        return network_response
    else:
        network_response.success = True
    if n_closest_stations is not None:
        moveouts.get_closest_stations(data_availability, n_closest_stations)
        print('Compute the beamformed network response only with the closest stations to each test seismic source')
    else:
        moveouts.closest_stations_indexes = None

    if saturation:
        print('Clip the traces above 1e{} times the MAD'.format(MAX_DYNAMIC_RANGE))
        for s in range(n_stations):
            for c in range(n_components):
                # clip the traes
                scale = scimad(detection_traces[s, c, :])
                detection_traces[s, c, :] = np.clip(detection_traces[s, c, :],
                                                    detection_traces[s, c, :].min(),
                                                    pow(10, MAX_DYNAMIC_RANGE)*scale)
    #return detection_traces
    #fig = plt.figure('detection_trace')
    #for s in range(n_stations):
    #    for c in range(n_components):
    #        plt.subplot(n_stations, n_components, s*n_components+c+1)
    #        plt.plot(detection_traces[s, c, :])
    #plt.show(block=True)
    #traces = traces.squeeze()
    if phase in ('sp','SP'):
        composite, where = clib.network_response_SP_prestacked(np.sum(detection_traces, axis=1),
                                                               moveoutP,
                                                               moveoutS,
                                                               smooth_win,
                                                               device=device,
                                                               closest_stations=moveouts.closest_stations_indexes,
                                                               test_points=test_points)

        #composite, where = clib.network_response_SP(np.sum(detection_traces[:,:-1,:], axis=1),
        #                                                   detection_traces[:,-1,:],
        #                                                   moveoutP,
        #                                                   moveoutS,
        #                                                   smooth_win,
        #                                                   device=device,
        #                                                   closest_stations=moveouts.closest_stations_indexes,
        #                                                   test_points=test_points)
        network_response.sp = True
    else:
        composite, where = clib.network_response(traces[:, 0, :], # North component
                                                 traces[:, 1, :], # East component
                                                 moveouts.cosine_azimuths,
                                                 moveouts.sine_azimuths,
                                                 moveout,
                                                 smooth_win,
                                                 device=device,
                                                 closest_stations=moveouts.closest_stations_indexes,
                                                 test_points=test_points)
        network_response.sp = False

    # attach results from backprojection to network_response
    #network_response.raw_composite = np.array(composite, copy=True)
    network_response.composite = composite
    network_response.where = where
    # remove the baseline
    baseline_window = 120. # in seconds
    #network_response.remove_baseline(baseline_window, attribute='composite')
    # smooth cnr for easier peak detection
    smoothing_kernel = 5. # in seconds
    #network_response.smooth_cnr(smoothing_kernel)
    return network_response


def find_templates(data,
                   network_response,
                   moveouts,
                   closest=False,
                   detection_threshold=None,
                   search_win=30.):
    """
    Analyze the composite network response to get candidate earthquakes
    from the data.

    Parameters
    -----------
    data: dictionary,
        Dictionary with two fields: 'waveforms', which is an
        (n_stations x n_components x n_samples) numpy array, and
        'metadata', which is a dictionary containing important metadata
        such as the station list, the component list, and the sampling rate.
    network_response: object,
        A composite network response object with multiple attributes
        calculated by the backprojection of seismic energy.
    moveouts: object,
        Moveout object from automatic_detection.moveouts.
    closest: boolean, default to False
        Whether or not seismic energy is backprojected to the grid
        point only using the closest stations.
    detection_threshold: scalar or 1d-array of same length as CNR,
                         default to None
    search_win: scalar, default to 30
        Length of the sliding search window, in seconds,
        for selecting the best detection within these windows.
        
    Returns
    -----------
    detections: dictionary,
        Dictionary with data and metadata of the detected earthquakes.
    """

    search_win = utils.sec_to_samp(search_win)

    #detection_trace = network_response.smoothed
    detection_trace = network_response.composite
    if detection_threshold is None:
        #detection_threshold = np.median(detection_trace) + cfg.CNR_threshold * cmn.mad(detection_trace)
        detection_threshold = time_dependent_threshold(detection_trace,
                                                       min(np.int32(detection_trace.size/4),
                                                           np.int32(0.5 * 3600. * cfg.sampling_rate)),
                                                       overlap=0.75) # sliding mad on 0.5h-long windows
    peaks = _detect_peaks(detection_trace, mpd=search_win)
    peaks = peaks[detection_trace[peaks] > detection_threshold[peaks]]
    #-------------------------------------------------------------------

    # remove peaks from buffer sections + 1/2 template length
    limit = utils.sec_to_samp(cfg.data_buffer + cfg.template_len/2.)
    peaks = peaks[peaks >= limit]
    limit = utils.sec_to_samp(86400 + cfg.data_buffer - cfg.template_len/2.)
    peaks = peaks[peaks < limit]

    # keep the largest peak for grouped detection
    for i in range(peaks.size):
        idx = np.int32(np.arange(max(0, peaks[i] - search_win/2),
                                 min(peaks[i] + search_win/2, network_response.composite.size)))
        idx_to_update = np.where(peaks == peaks[i])[0]
        peaks[idx_to_update] = np.argmax(network_response.composite[idx]) + idx[0]

    peaks, idx = np.unique(peaks, return_index=True)

    peaks   = np.asarray(peaks)
    sources = network_response.where[peaks]
    CNR     = detection_trace[peaks]
    #----------------------------------------------
    if network_response.sp:
        method = 'SP'
    else:
        method = 'S'
    print(peaks.size)
    detections = extract_detections(data, peaks, sources, moveouts, CNR, method=method)

    print("------------------------")
    print("{:d} templates".format(peaks.size))   

    return detections

def extract_detections(data,
                       peaks,
                       test_sources,
                       moveouts,
                       CNR,
                       method='S'):
    """
    Extract the events identified by the composite network response.
    Returns the output in a dictionnary, which can easily be converted into
    a h5 databse.

    Parameters
    -----------
    data: dictionary,
        Dictionary with two fields: 'waveforms', which is an
        (n_stations x n_components x n_samples) numpy array, and
        'metadata', which is a dictionary containing important metadata
        such as the station list, the component list, and the sampling rate.
    peaks: numpy array of integers,
        Array of peak positions, in samples, where candidate earthquakes
        should be extracted.
    test_sources: numpy array of integers,
        Source indexes where the candidate earthquakes were backprojected.
    moveouts: object,
        Moveout object from automatic_detection.moveouts.
    CNR: numpy array,
        Array of the composite network response values associated
        with the peak indexes.
    method: string, default to 'S'
        'S' or 'SP'. This string should be prescribed according
        to the method that was due when backprojecting the energy.

    Returns
    --------
    detections: dictionary,
        A dictionary with the data and metadata of the candidate
        earthquakes.
    """
    detections = {}
    if moveouts.closest_stations_indexes is not None:
        n_stations = moveouts.closest_stations_indexes.shape[-1]
    else:
        n_stations = data['waveforms'].shape[0]
    n_components  = data['waveforms'].shape[1]
    n_samples     = np.int32(cfg.template_len * cfg.sampling_rate)
    n_detections  = peaks.size
    #--------------------------------------------------
    waveforms                   =  np.zeros((n_detections, n_stations, n_components, n_samples), dtype=np.float32)
    origin_times                =  np.zeros( n_detections,                                       dtype=np.float64)
    test_source_indexes         =  np.zeros( n_detections,                                       dtype=np.int32)
    relative_travel_times       =  np.zeros((n_detections, n_stations, n_components),            dtype=np.float32)
    kurtosis                    =  np.zeros((n_detections, n_stations, n_components),            dtype=np.float32)
    composite_network_response  =  np.zeros(n_detections,                                        dtype=np.float32)
    locations                   =  np.zeros((n_detections, 3),                                   dtype=np.float32)
    stations                    =  []
    for i in range(n_detections):
        if moveouts.closest_stations_indexes is not None:
            indexes_stations = moveouts.closest_stations_indexes[test_sources[i], :]
            #stations.append(np.asarray(data['metadata']['stations'])[moveouts.closest_stations_indexes[test_sources[i], :]])
            stations.append(np.asarray(data['metadata']['stations'])[indexes_stations])
        else:
            indexes_stations = np.arange(n_stations)
            stations.append(data['metadata']['stations'])
        if method.upper() == 'S':
            mvs = moveouts.s_relative_samp[test_sources[i], indexes_stations]
            # make sure the moveouts are the relative travel times
            mvs -= mvs.min() # relative to the shortest S-wave travel time
            # reshape mvs to get a (n_stations x n_components) matrix
            mvs = np.repeat(mvs, n_components).reshape(n_stations, n_components)
        elif method.upper() == 'SP':
            #mvs = np.hstack( (moveouts.s_relative_p_samp[test_sources[i], indexes_stations].reshape(-1, 1), \
            #                  moveouts.s_relative_p_samp[test_sources[i], indexes_stations].reshape(-1, 1), \
            #                  moveouts.p_relative_samp[test_sources[i],   indexes_stations].reshape(-1, 1)) )
            mvs = np.stack([moveouts.s_relative_p_samp[test_sources[i], indexes_stations], 
                            moveouts.s_relative_p_samp[test_sources[i], indexes_stations], 
                            moveouts.p_relative_samp[test_sources[i],   indexes_stations]], axis=1)

            # make sure the moveouts are the relative travel times
            # to be coherent with the NR computation routine
            mvs -= mvs[:, -1].min() # relative to the shortest P-wave travel time
        for s in range(n_stations):
            #ss = moveouts.closest_stations_indexes[test_sources[i], s]
            ss = indexes_stations[s]
            for c in range(n_components):
                # extract the waveforms between t1 and t2
                t1 = np.int32(peaks[i] + mvs[s, c] - n_samples//2)
                t2 = t1 + n_samples
                if (t2 < data['waveforms'].shape[-1]) and (t1 > 0):
                    waveforms[i, s, c, :] = data['waveforms'][ss, c, t1:t2]
        #--------------------------------
        timing = data['metadata']['date'] + peaks[i]/cfg.sampling_rate - cfg.data_buffer
        origin_times[i]                 = timing.timestamp
        #--------------------------------
        test_source_indexes[i]          = test_sources[i]
        #--------------------------------
        relative_travel_times[i, :, :]  = np.float32(mvs / cfg.sampling_rate)
        #--------------------------------
        composite_network_response[i]   = CNR[i]
        #--------------------------------
        locations[i,:] = np.array([moveouts.longitude[test_sources[i]],\
                                   moveouts.latitude[test_sources[i]],\
                                   moveouts.depth[test_sources[i]]])
    detections.update({'waveforms'                    :   waveforms})
    detections.update({'origin_times'                 :   origin_times})
    detections.update({'test_source_indexes'          :   test_source_indexes})
    detections.update({'moveouts'                     :   relative_travel_times})
    detections.update({'composite_network_response'   :   composite_network_response})
    detections.update({'stations'                     :   np.asarray(stations).astype('S')})
    detections.update({'components'                   :   np.asarray(data['metadata']['components']).astype('S')})
    detections.update({'locations'                    :   locations})
    return detections


def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                  kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1]:http://nbviewer.ipython.org/github/demotu/BMC/blob/master/
        notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan,
                                                    indnan - 1, indnan + 1))),
                          invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]),
                    axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='{:d} {}'.format(ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("{} (mph={}, mpd={:d}, threshold={}, edge='{}')"
                     .format(mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

def envelope_parallel(traces):
    """Compute the envelope of traces.  

    Compute the traces' envelopes using a simple
    parallelization of the envelope function.
    """
    from multiprocessing import cpu_count, Pool
    traces_reshape = traces.reshape(-1, traces.shape[-1])
    p = Pool(cpu_count())
    iterable = (traces_reshape[i,:] for i in range(traces_reshape.shape[0]))
    tr_analytic = np.asarray(p.map(envelope, iterable, chunksize=1), dtype=np.float32)\
                            .reshape(traces.shape)
    p.close()
    p.join()
    return tr_analytic

def envelope(trace):
    """Compute the envelope of trace.  

    Returns the trace's envelope using the hilbert transform from Scipy.
    """
    return np.float32(np.abs(hilbert(trace)))

def compute_envelopes(traces):
        start = give_time()
        traces = envelope_parallel(traces) # take the upper envelope of the traces
        end = give_time()
        print('Computed the envelopes in {:.2f}sec.'.format(end-start))
        return traces

def baseline(X, w):
    n_windows = np.int32(np.ceil(X.size/w))
    minima      = np.zeros(n_windows, dtype=X.dtype)
    minima_args = np.zeros(n_windows, dtype=np.int32)
    for i in range(n_windows):
        minima_args[i] = i*w + X[i*w:(i+1)*w].argmin()
        minima[i]      = X[minima_args[i]]
    #----------------------------------------
    #--------- build interpolation ----------
    interpolator = interp1d(minima_args, minima, kind='linear', fill_value='extrapolate')
    bline = interpolator(np.arange(X.size))
    return bline

def time_dependent_threshold(network_response,
                             window,
                             overlap=0.75):

    # calculate n_windows given window
    # and overlap
    shift = int((1.-overlap)*window)
    n_windows = int((len(network_response)-window)//shift)+1
    mad_ = np.zeros(n_windows+2, dtype=np.float32)
    med_ = np.zeros(n_windows+2, dtype=np.float32)
    time = np.zeros(n_windows+2, dtype=np.float32)
    for i in range(1, n_windows+1):
        i1 = i*shift
        i2 = min(network_response.size, i1+window)
        cnr_window = network_response[i1:i2]
        #non_zero = cnr_window != 0
        #if sum(non_zero) < 3:
        #    # won't be possible to calculate median
        #    # and mad on that few samples
        #    continue
        #med_[i] = np.median(cnr_window[non_zero])
        #mad_[i] = scimad(cnr_window[non_zero])
        med_[i] = np.median(cnr_window)
        mad_[i] = scimad(cnr_window)
        time[i] = (i1+i2)/2.
    # add boundary cases manually
    time[0] = 0.
    mad_[0] = mad_[1]
    med_[0] = med_[1]
    time[-1] = network_response.size
    mad_[-1] = mad_[-2]
    med_[-1] = med_[-2]
    threshold = med_ + cfg.CNR_threshold * mad_
    interpolator = interp1d(time,
                            threshold,
                            kind='slinear',
                            fill_value=(threshold[0], threshold[-1]),
                            bounds_error=False)
    full_time = np.arange(0, network_response.size)
    threshold = interpolator(full_time)
    return threshold

def time_dependent_threshold_pd(network_response,
                                window):
    """
    Calculate a time dependent detection threshold
    using the rolling function from pandas.

    Parameters
    -----------
    network_response: numpy array,
        Composite network response on which we calculate
        the detection threshold.
    window: scalar, integer
        Length of the sliding window, in samples, over
        which we calculate the running statistics used
        in the detection threshold.
    Returns
    --------
    detection_threshold: numpy array,
        Detection threshold that will serve to select
        the well backprojected events.
    """
    network_response_pd = pd.Series(network_response)
    r = network_response_pd.rolling(window=window)
    # get running median and running mad
    run_med = r.median().shift(1)
    run_mad = r.apply(scimad).shift(1)
    # combine these into a detection threshold
    detection_threshold = run_med + cfg.CNR_threshold*run_mad
    return detection_threshold.values

