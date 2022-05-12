import os
import sys

from .config import cfg
from . import clib, utils, dataset

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import beamnetresponse as bnr

try:
    from scipy.stats import median_abs_deviation as scimad
except ImportError:
    from scipy.stats import median_absolute_deviation as scimad
#from scipy.stats import median_abs_deviation as scimad
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import hilbert

from time import time as give_time

from math import isnan

from obspy.core import UTCDateTime as udt

class NetworkResponse(object):
    """Class for computing and post-processing the network response.  

    """
    
    def __init__(self, data, network, tt_filename='tts.h5',
            path_tts=cfg.moveouts_path, phases=['P', 'S'],
            starttime=None):
        """Initialize the essential attributes.  

        Parameters
        -----------
        data: dataset.Data instance
            The Data instance with the waveform time series and metadata
            required for the computation of the (composite) network response.
        network: dataset.Network instance
            The Network instance with the station network information.
            `network` can force the network response to be computed only on
            a subset of the data stored in `data`.
        tt_filename: string, default to 'tts.h5'
            Name of the hdf5 file with travel-times.
        path_tts: string, default to `cfg.moveouts_path`
            Path to the directory with the travel-time files.
        phases: list, default to ['P', 'S']
            List of seismic phases used in the computation of the network
            response.
        starttime: string or Datetime, default to None
            Start time of the network response time series. If None, takes
            `self.data.date` as the start time.
        """
        self.data = data
        self.network = network
        self.path_tts = os.path.join(path_tts, tt_filename)
        self.phases = phases
        if starttime is None:
            self.starttime = self.data.date
        else:
            self.starttime = udt(starttime)

    @property
    def n_sources(self):
        if hasattr(self, 'moveouts'):
            return self.moveouts.shape[0]
        else:
            print('You should attribute moveouts to the class instance first.')
            return

    @property
    def n_stations(self):
        return self.network.n_stations

    @property
    def moveouts(self):
        if not hasattr(self, '_moveouts'):
            print('You need to call `load_moveouts` or `set_moveouts` first.')
            return
        else:
            return self._moveouts

    @property
    def source_coordinates(self):
        if not hasattr(self, '_source_coordinates'):
            print('You need to call `load_moveouts` or `set_source_coords`'
                  ' first.')
            return
        else:
            return self._source_coordinates

    def compute_network_response(self, detection_traces,
            composite=True, device='cpu'):
        """Compute the network response.  

        Parameters
        --------------
        detection_traces: (n_stations, n_components, n_samples) numpy.ndarray
            Some characteristic function of the waveform time series to
            backproject onto the grid of theoretical sources.
        device: string, default to 'cpu'
            Either 'cpu' or 'gpu', depending on the available hardware and
            user's preferences.
        composite: boolean, default to True
            If True, compute the composite network response. That is, search
            for the one source that produces the largest network response at
            each time step.
        """
        if not hasattr(self, 'weights_phases'):
            print('You need to set self.weights_phases first.')
            return
        if not hasattr(self, 'weights_sources'):
            print('You need to set self.weights_sources first.')
            return
        if not hasattr(self, '_moveouts'):
            print('You need to call `load_moveouts` or `set_moveouts` first.')
            return
        elif self.moveouts.dtype not in (np.int32, np.int64):
            print('Moveouts should be integer typed and in unit of samples.')
            return
        if composite:
            self.cnr, self.cnr_sources = \
                    bnr.beamformed_nr.composite_network_response(
                            detection_traces, self.moveouts, self.weights_phases,
                            self.weights_sources, device=device)
        else:
            self.nr = \
                    bnr.beamformed_nr.network_response(
                            detection_traces, self.moveouts, self.weights_phases,
                            self.weights_sources, device=device)

    def find_detections(self, detection_threshold,
                        minimum_interevent_time, n_max_stations=None):
        """Analyze the composite network response to find detections.

        Parameters
        -----------
        detection_threshold: scalar or (n_samples,) numpy.ndarray, float
            The number of running MADs taken above the running median
            to define detections.
        minimum_interevent_time: scalar, float
            The shortest duration, in seconds, allowed between two
            consecutive detections.
        n_max_stations: integer, default to None
            If not None and if smaller than the total number of stations in the
            network, only extract the `n_max_stations` closest stations for
            each theoretical source.
            
        Returns
        -----------
        detections: dictionary,
            Dictionary with data and metadata of the detected earthquakes.
        """
        from obspy import Stream
        self.detection_threshold = detection_threshold
        self.minimum_interevent_time = minimum_interevent_time
        sr = self.data.sr
        minimum_interevent_time = utils.sec_to_samp(minimum_interevent_time, sr=sr)

        # select peaks
        peak_indexes = _detect_peaks(self.cnr, mpd=minimum_interevent_time)
        # only keep peaks above detection threshold
        peak_indexes = peak_indexes[self.cnr[peak_indexes] > detection_threshold[peak_indexes]]

        # keep the largest peak for grouped detection
        for i in range(len(peak_indexes)):
            idx = np.int32(np.arange(max(0, peak_indexes[i] - minimum_interevent_time/2),
                                     min(peak_indexes[i] +
                                         minimum_interevent_time/2, len(self.cnr))))
            idx_to_update = np.where(peak_indexes == peak_indexes[i])[0]
            peak_indexes[idx_to_update] = np.argmax(self.cnr[idx]) + idx[0]

        peak_indexes = np.unique(peak_indexes)

        peak_indexes = np.asarray(peak_indexes)
        source_indexes = self.cnr_sources[peak_indexes]

        # extract waveforms
        detections = []
        data_path, data_filename = os.path.split(self.data.where)
        for i in range(len(peak_indexes)):
            src_idx = source_indexes[i]
            event = Stream()
            ot_i = self.data.date + peak_indexes[i]/sr
            mv = self.moveouts[src_idx, ...]/sr
            if n_max_stations is not None:
                # use moveouts as a proxy for distance
                # keep only the n_max_stations closest stations
                mv_max = np.sort(mv[:, 0])[n_max_stations-1]
            else:
                mv_max = np.finfo(np.float32).max
            stations_in = np.asarray(self.network.stations)[mv[:, 0] < mv_max]
            latitude = self.source_coordinates['latitude'][src_idx]
            longitude = self.source_coordinates['longitude'][src_idx]
            depth = self.source_coordinates['depth'][src_idx]
            event = dataset.Event(ot_i, mv, stations_in, self.phases,
                    data_filename, data_path, latitude=latitude,
                    longitude=longitude, depth=depth, sampling_rate=sr)
            aux_data = {}
            aux_data['cnr'] = self.cnr[peak_indexes[i]]
            aux_data['source_index'] = src_idx
            event.set_aux_data(aux_data)
            detections.append(event)
        
        print(f'Extracted {len(detections):d} events.')

        self.peak_indexes = peak_indexes
        self.source_indexes = source_indexes
        return detections, peak_indexes, source_indexes

    def load_moveouts(self, source_indexes=None, remove_min=True):
        """Load the moveouts, in units of samples.  

        Call `utils.load_travel_times` and `utils.get_moveout_array` using the
        instance's attributes and the optional parameters given here. Add the
        attribute `self.moveouts` to the instance. The moveouts are converted
        to samples using the sampling rate `self.data.sampling_rate`.

        Parameters
        ------------
        source_indexes: (n_sources,) int numpy.ndarray, default to None
            If not None, this is used to select a subset of sources from the
            grid.
        remove_min: boolean, default to True
            If True, remove the smallest travel time from the collection of
            travel times for each source of the grid. The network response only
            depends on the relative travel times -- the moveouts -- and
            therefore it is unnecessary to carry potentially very large travel
            times.
        """
        tts, self._source_coordinates = utils.load_travel_times(
                self.path_tts, phases, source_indexes=source_indexes,
                return_coords=True)
        self._moveouts = utils.get_moveout_array(tts, self.networks.stations,
                self.phases)
        del tts
        if remove_min:
            self._moveouts -= np.min(self._moveouts, axis=(1, 2), keepdims=True)
        self._moveouts = utils.sec_to_samp(self._moveouts, sr=self.data.sr)

    def remove_baseline(self, window, attribute='composite'):
        """Remove baseline from network response.  

        """
        # convert window from seconds to samples
        window = int(window*self.sampling_rate)
        attr_baseline = self._baseline(getattr(self, attribute), window)
        setattr(self, attribute, getattr(self, attribute)-attr_baseline)

    def return_pd_series(self, attribute='cnr'):
        """Return the network response as a Pandas.Series.  

        """
        import pandas as pd
        time_series = getattr(self, attribute)
        indexes = pd.date_range(start=str(self.starttime),
                                freq='{}S'.format(1./self.data.sr),
                                periods=len(time_series))
        pd_attr = pd.Series(data=time_series, index=indexes)
        return pd_attr

    def smooth_cnr(self, window):
        """Smooth the network response with a gaussian kernel.  

        """
        from scipy.ndimage.filters import gaussian_filter1d
        # convert window from seconds to samples
        window = int(window*self.sampling_rate)
        self.smoothed = gaussian_filter1d(self.composite, window)

    def set_moveouts(self, moveouts):
        """Attribute `_moveouts` to the class instance.  

        Parameters
        -----------
        moveouts: (n_sources, n_stations, n_phases) numpy.ndarray
            The moveouts to use for backprojection.
        """
        self._moveouts = moveouts

    def set_source_coordinates(self, source_coords):
        """Attribute `_source_coordinates` to the class instance.  

        Parameters
        ------------
        source_coords: dictionary
            Dictionary with 3 fields: 'latitude', 'longitude' and 'depth'
        """
        self._source_coordinates = source_coords

    def set_weights(self, weights_phases=None, weights_sources=None):
        """Set the weights required by `beamnetresponse`.  

        weights_phases: (n_stations, n_channels, n_phases) np.ndarray, default
        to None
            Weight given to each station and channel for a given phase. For
            example, horizontal components might be given a small or zero
            weight for the P-wave stacking.
        weights_sources: (n_sources, n_stations) np.ndarray, default to None
            Source-receiver-specific weights. For example, based on the
            source-receiver distance.
        """
        if weights_phases is not None:
            self.weights_phases = weights_phases
        if weights_sources is not None:
            self.weights_sources = weights_sources

    def set_weights_sources(self, n_max_stations):
        """Set network-geometry-based weights of each source-receiver pair.  

        Parameters
        ------------
        n_max_stations: scalar, int
            Maximum number of stations used at each theoretical source. Only
            the `n_max_stations` stations will be set a weight > 0.
        """
        weights_sources = np.ones((self.n_sources, self.n_stations),
                dtype=np.float32)
        if n_max_stations < self.n_stations:
            cutoff_mv = np.max(np.partition(self.moveouts[:, :, 0], n_max_stations)
                    [:, :n_max_stations], axis=1, keepdims=True) 
            weights_sources[self.moveouts[:, :, 0] > cutoff_mv] = 0.
        self.weights_sources = weights_sources

    def weights_station_density(self, cutoff_dist=None, lower_percentile=0.,
            upper_percentile=100.):
        """Compute station weights to balance station density.

        Areas of high station density produce stronger network responses than
        areas with sparse coverage, and thus might prevent detecting earthquakes
        where stations are scarcer.

        Parameters
        -----------
        cutoff_dist: scalar float, default to None
            All station pairs (i,j) are attributed a number from a gaussian
            distribution with standard deviation `cutoff_dist`, in km. The
            weight of station i is: `w_i = 1/(sum_j(exp(-D_ij**2/cutoff_dist**2)))`.
            If None, `cutoff_dist` is set to the median interstation distance.
        lower_percentile: scalar float, default to 0
            If `lower_percentile > 0`, the weights are clipped above the
            `lower_percentile`-th percentile.
        upper_percentile: scalar float, default to 100
            If `upper_percentile < 100`, the weights are clipped below the
            `upper_percentile`-th percentile.

        Returns
        -------
        weights_sta_density: (n_stations,) `numpy.ndarray`
            The station density weights.
        """
        if cutoff_dist is None:
            cutoff_dist = np.median(
                    self.network.interstation_distances.values
                    [self.network.interstation_distances.values != 0.])
        weights_sta_density = np.zeros(self.network.n_stations, dtype=np.float32)
        for s, sta in enumerate(self.network.stations):
            dist_sj = self.network.interstation_distances.loc[sta]
            weights_sta_density[s] = 1./np.sum(np.exp(-dist_sj**2/cutoff_dist**2))
        if lower_percentile > 0.:
            weights_sta_density = np.clip(
                    weights_sta_density,
                    np.percentile(weights_sta_density, lower_percentile),
                    weights_sta_density.max())
        if upper_percentile < 100.:
            weights_sta_density = np.clip(
                    weights_sta_density,
                    weights_sta_density.min(),
                    np.percentile(weights_sta_density, upper_percentile))
        return weights_sta_density


    def _baseline(self, X, window):
        """Compute a baseline.  

        The baseline is a curve connecting the local minima. Removing the
        baseline is equivalent to some kind of low-pass filtering.
        """
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

    # -------------------------------------------
    #       Plotting methods
    # -------------------------------------------
    def plot_cnr(self, ax=None, detection=None, figsize=(20, 7)):
        """Plot the composite network response.  

        Parameters
        -----------
        ax: plt.Axes, default to None
            If not None, plot in this axis.
        detection: dataset.Event, default to None
            A dataset.Event instance of a given detection.
        figsize: tuple, default to (20, 7)
            Width and height of the figure, in inches.

        Returns
        --------
        fig: plt.Figure
            The Figure instance produced by this method.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        if ax is None:
            # plot the composite network response
            fig = plt.figure('composite_network_response', figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        ax.plot(self.data.time, self.cnr)
        ax.plot(self.data.time, self.detection_threshold, color='C3', ls='--',
                label='Detection Threshold')
        ax.plot(self.data.time[self.peak_indexes], self.cnr[self.peak_indexes],
                marker='o', ls='', color='C3')
        ax.legend(loc='upper right')
        ax.set_xlabel('Time of the day')
        ax.set_ylabel('Composite Network Response')

        ax.set_xlim(self.data.time.min(), self.data.time.max())
        #ax.set_ylim(-0.1*(detection_threshold.max() - cnr.min()), 1.2*detection_threshold.max())
        #ax.set_ylim(-0.1*(detection_threshold.max() - cnr.min()), 1.2*detection_threshold.max())
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        ax.legend(loc='upper right')
        
        if detection is not None:
            ot = np.datetime64(detection.origin_time)
            ax.annotate('detection', (ot, detection.aux_data['cnr']),
                        (ot + np.timedelta64(15, 'm'), min(ax.get_ylim()[1],
                            2.*detection.aux_data['cnr'])),
                        arrowprops={'width': 2, 'headwidth': 5, 'color': 'k'})
        return fig

    def plot_detection(self, detection, figsize=(20, 20),
            component_aliases={'N': ['N', '1'], 'E': ['E', '2'], 'Z': ['Z']},
            n_stations=None):
        """Plot a detection and the composite network response.  

        Parameters
        -----------
        detection: dataset.Event
            A dataset.Event instance of a given detection.
        figsize: tuple, default to (20, 20)
            Widht and height of the figure, in inches.
        component_aliases: Dictionary, optional
            Sometimes, components might be named differently than N, E, Z. This
            dictionary tells the function which alternative component names can be
            associated with each "canonical" component. For example,  
            `component_aliases['N'] = ['N', '1']` means that the function will also
            check the '1' component in case the 'N' component doesn't exist.
        n_stations: scalar int or None, default to None
            If not None, is the number of stations to plot. The closest
            stations will be plotted.

        Returns
        -------
        fig: plt.Figure
            The Figure instance produced by this method.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if n_stations is None:
            stations = self.network.stations
        else:
            stations = detection.moveouts[detection.moveouts.columns[0]]\
                    .sort_values()[:n_stations].index
   
        sr = self.data.sr
        fig = plt.figure(f'detection_{detection.origin_time}', figsize=figsize)
        grid = fig.add_gridspec(nrows=len(stations)+2,
                ncols=len(self.network.components), hspace=0.35)
        start_times, end_times = [], []
        wav_axes = []
        ax_cnr = fig.add_subplot(grid[:2, :])
        self.plot_cnr(ax=ax_cnr, detection=detection)
        ax_cnr.set_ylim(-0.5, 2.*detection.aux_data['cnr'])
        beam = 0.
        for s, sta in enumerate(stations):
            for c, cp in enumerate(self.network.components):
                ax = fig.add_subplot(grid[2+s, c])
                for cp_alias in component_aliases[cp]:
                    tr = detection.traces.select(station=sta, component=cp_alias)
                    if len(tr) > 0:
                        # succesfully retrieved data
                        break
                if len(tr) == 0:
                    continue
                else:
                    tr = tr[0]
                time = utils.time_range(tr.stats.starttime,
                        tr.stats.endtime+1./sr, 1./sr, unit='ms')
                start_times.append(time[0])
                end_times.append(time[-1])
                ax.plot(time[:detection.n_samples], tr.data[:detection.n_samples], color='k')
                # plot the theoretical pick
                phase = detection.aux_data[f'phase_on_comp{cp_alias}'].upper()
                offset_ph = detection.aux_data[f'offset_{phase}']
                ax.axvline(time[0] + np.timedelta64(int(1000.*offset_ph), 'ms'), color='C3')
                ax.text(0.05, 0.05, f'{sta}.{cp_alias}', transform=ax.transAxes)
                wav_axes.append(ax)
        for ax in wav_axes:
            ax.set_xlim(min(start_times), max(end_times))
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.subplots_adjust(top=0.95, bottom=0.06, right=0.98, left=0.06)
        return fig



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

def time_dependent_threshold(network_response, window,
                             overlap=0.75, CNR_threshold=cfg.CNR_threshold):
    """Compute a time-dependent detection threshold.  


    Parameters
    -----------
    network_response: (n_samples,) numpy.ndarray, float
        Composite network response on which we calculate
        the detection threshold.
    window: scalar, integer
        Length of the sliding window, in samples, over
        which we calculate the running statistics used
        in the detection threshold.
    overlap: scalar, float, default to 0.75
        Ratio of overlap between two contiguous windows.
    CNR_threshold: scalar, float, default to 10
        Number of running MADs above running median that
        defines the detection threshold.
    Returns
    --------
    detection_threshold: (n_samples,) numpy.ndarray
        Detection threshold on the network response.
    """
    try:
        from scipy.stats import median_abs_deviation as scimad
    except ImportError:
        from scipy.stats import median_absolute_deviation as scimad
    from scipy.interpolate import interp1d

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
    time[-1] = len(network_response)
    mad_[-1] = mad_[-2]
    med_[-1] = med_[-2]
    threshold = med_ + CNR_threshold * mad_
    interpolator = interp1d(
            time, threshold, kind='slinear',
            fill_value=(threshold[0], threshold[-1]),
            bounds_error=False)
    full_time = np.arange(0, len(network_response))
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


# ---------------------------------------------------------------
#                      Detection traces
# ---------------------------------------------------------------

def saturated_envelopes(traces, anomaly_threshold=1.e-11, max_dynamic_range=1.e5):
    """Compute the saturated envelopes.  

    Parameters
    ------------
    traces: (n_stations, n_components, n_samples) numpy.ndarray
        Input waveform time series.
    anomaly_threshold: scalar, float, default to 1e-11
        Scalar threshold below which the MAD is suspicious. It should be a very
        small number if you are working on physical unit seismograms.
    max_dynamic_range: scalar, float, default to 1e5
        Higher cutoff on the standardized envelopes. This mitigates the
        contamination of the network response by transient, undesired high
        energy signals such as spikes.
    """
    n_stations, n_components, n_samples = traces.shape
    tstart = give_time()
    detection_traces = envelope_parallel(traces) # take the upper envelope of the traces
    tend = give_time()
    print(f'Computed the envelopes in {tend-tstart:.2f}sec.')
    data_availability = np.zeros(n_stations, dtype=np.int32)
    for s in range(n_stations):
        for c in range(n_components):
            missing_samples = detection_traces[s, c, :] == 0.
            if np.sum(missing_samples) > detection_traces.shape[-1]/2:
                # too many samples are missing, don't use this trace
                # do not increment data_availability
                detection_traces[s, c, :] = 0.
                continue
            median = np.median(detection_traces[s, c, ~missing_samples])
            mad = scimad(detection_traces[s, c, ~missing_samples])
            if mad < anomaly_threshold:
                detection_traces[s, c, :] = 0.
                continue
            detection_traces[s, c, :] = (detection_traces[s, c, :] - median) / mad
            detection_traces[s, c, missing_samples] = 0.
            # saturate traces
            detection_traces[s, c, :] = np.clip(
                    detection_traces[s, c, :], detection_traces[s, c, :],
                    max_dynamic_range)
            data_availability[s] += 1
    return detection_traces, data_availability

def envelope_parallel(traces):
    """Compute the envelope of traces.  

    The envelope is defined as the modulus of the complex
    analytical signal (a signal whose Fourier transform only has
    energy in positive frequencies).

    Parameters
    -------------
    traces: (n_stations, n_channels, n_samples) numpy.ndarray, float
        The input time series.

    Returns
    -------------
    envelopes: (n_stations, n_channels, n_samples) numpy.ndarray, float
        The moduli of the analytical signal of the input traces.
    """
    import concurrent.futures
    traces_reshaped = traces.reshape(-1, traces.shape[-1])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        envelopes = np.float32(list(executor.map(
           envelope, traces_reshaped)))
    return envelopes.reshape(traces.shape)

def envelope(trace):
    """Compute the envelope of trace.  

    The envelope is defined as the modulus of the complex
    analytical signal (a signal whose Fourier transform only has
    energy in positive frequencies).

    Parameters
    -------------
    trace: (n_samples) numpy.ndarray, float
        The input time series.

    Returns
    -------------
    envelope: (n_samples) numpy.ndarray, float
        The modulus of the analytical signal of the input traces.
    """
    from scipy.signal import hilbert
    return np.float32(np.abs(hilbert(trace)))

