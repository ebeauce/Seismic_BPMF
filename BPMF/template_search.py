import os
import sys

from .config import cfg
from . import clib, utils, dataset

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import beampower as bp

try:
    from scipy.stats import median_abs_deviation as scimad
except ImportError:
    from scipy.stats import median_absolute_deviation as scimad
# from scipy.stats import median_abs_deviation as scimad
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import hilbert

from time import time as give_time

from math import isnan

from obspy.core import UTCDateTime as udt


class Beamformer(object):
    """Class for backprojecting waveform features with beamforming."""

    def __init__(
        self,
        travel_times,
        source_coordinates,
        sampling_rate=cfg.SAMPLING_RATE_HZ,
        data=None,
        network=None,
        phases=None,
        remove_tt_min=True,
    ):
        """Initialize the essential attributes.

        Parameters
        -----------
        tt_filename: string, default to 'tts.h5'
            Name of the hdf5 file with travel-times.
        path_tts: string, default to `cfg.MOVEOUTS_PATH`
            Path to the directory with the travel-time files.
        phases: list, default to ['P', 'S']
            List of seismic phases used in the computation of the network
            response.
        """
        self.travel_times = travel_times
        self.travel_times_samp = self.travel_times.copy()
        for sta in self.travel_times_samp.index:
            for ph in self.travel_times_samp.columns:
                self.travel_times_samp.loc[sta, ph] = utils.sec_to_samp(
                    self.travel_times_samp.loc[sta, ph],
                    sr=sampling_rate,
                )
        self.source_coordinates = source_coordinates
        self.n_sources = len(self.source_coordinates["depth"])
        self.data = data
        self.network = network
        self.phases = phases
        if remove_tt_min:
            # find min travel-time for each source
            tts = np.asarray(
                [
                    [
                        self.travel_times_samp.loc[sta, ph]
                        for sta in self.travel_times_samp.index
                    ]
                    for ph in self.phases
                ]
            ).T
            tt_min = np.min(tts, axis=(1, 2))
            for sta in self.travel_times_samp.index:
                for ph in self.phases:
                    self.travel_times_samp.loc[sta, ph] -= tt_min

    @property
    def n_stations(self):
        if not hasattr(self, "network") or self.network == None:
            print("You need to call set_network first.")
            return
        return self.network.n_stations

    @property
    def n_phases(self):
        if not hasattr(self, "phases") or self.phases == None:
            print("You need to call set_phases first.")
            return
        return len(self.phases)

    @property
    def moveouts(self):
        return np.asarray(
            [
                [self.travel_times_samp.loc[sta, ph] for sta in self.stations]
                for ph in self.phases
            ]
        ).T

    @property
    def stations(self):
        if not hasattr(self, "network") or self.network == None:
            print("You need to call set_network first.")
            return
        return self.network.stations

    @staticmethod
    def _likelihood(beam_volume):
        likelihood = (beam_volume - beam_volume.min()) / (
            beam_volume.max() - beam_volume.min()
        )
        # likelihood is not meant to be outside [0, 1] beside numerical
        # imprecisions
        likelihood = np.clip(likelihood, a_min=0., a_max=1.)
        return likelihood

    def backproject(self, waveform_features, reduce="max", device="cpu"):
        """Backproject the waveform features.

        Parameters
        --------------
        waveform_features: (n_stations, n_components, n_samples) numpy.ndarray
            Features of the waveform time series used for the
            backprojection onto the grid of theoretical sources.
        device: string, default to 'cpu'
            Either 'cpu' or 'gpu', depending on the available hardware and
            user's preferences.
        reduce: string, default to 'max'
            Either 'max' or 'none'. If 'max', returns the maximum beam at every
            time. If 'none', returns the full space-time beam.
        """
        if not hasattr(self, "weights_phases"):
            print("You need to set self.weights_phases first.")
            return
        if not hasattr(self, "weights_sources"):
            print("You need to set self.weights_sources first.")
            return
        if reduce == "max":
            self.maxbeam, self.maxbeam_sources = bp.beampower.beamform(
                waveform_features,
                self.moveouts,
                self.weights_phases,
                self.weights_sources,
                device=device,
                reduce=reduce,
            )
        elif reduce == "none":
            self.beam = bp.beampower.beamform(
                waveform_features,
                self.moveouts,
                self.weights_phases,
                self.weights_sources,
                device=device,
                reduce=reduce,
            )
        else:
            print(f"'reduce' should be 'max' or 'none' but {reduce} was given.")
            self.beam = None

    def find_detections(
        self, detection_threshold, minimum_interevent_time, n_max_stations=None
    ):
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
        peak_indexes = _detect_peaks(self.maxbeam, mpd=minimum_interevent_time)
        # only keep peaks above detection threshold
        peak_indexes = peak_indexes[
            self.maxbeam[peak_indexes] > detection_threshold[peak_indexes]
        ]

        # keep the largest peak for grouped detection
        for i in range(len(peak_indexes)):
            idx = np.int32(
                np.arange(
                    max(0, peak_indexes[i] - minimum_interevent_time / 2),
                    min(
                        peak_indexes[i] + minimum_interevent_time / 2, len(self.maxbeam)
                    ),
                )
            )
            idx_to_update = np.where(peak_indexes == peak_indexes[i])[0]
            peak_indexes[idx_to_update] = np.argmax(self.maxbeam[idx]) + idx[0]

        peak_indexes = np.unique(peak_indexes)

        peak_indexes = np.asarray(peak_indexes)
        source_indexes = self.maxbeam_sources[peak_indexes]

        # extract waveforms
        detections = []
        data_path, data_filename = os.path.split(self.data.where)
        for i in range(len(peak_indexes)):
            src_idx = self.source_coordinates.index[source_indexes[i]]
            event = Stream()
            ot_i = self.data.date + peak_indexes[i] / sr
            mv = self.moveouts[source_indexes[i], ...] / sr
            if n_max_stations is not None:
                # use moveouts as a proxy for distance
                # keep only the n_max_stations closest stations
                mv_max = np.sort(mv[:, 0])[n_max_stations - 1]
            else:
                mv_max = np.finfo(np.float32).max
            stations_in = np.asarray(self.network.stations)[mv[:, 0] < mv_max]
            latitude = self.source_coordinates.loc[src_idx, "latitude"]
            longitude = self.source_coordinates.loc[src_idx, "longitude"]
            depth = self.source_coordinates.loc[src_idx, "depth"]
            event = dataset.Event(
                ot_i,
                mv,
                stations_in,
                self.phases,
                data_filename,
                data_path,
                latitude=latitude,
                longitude=longitude,
                depth=depth,
                sampling_rate=sr,
                data_reader=self.data.data_reader,
            )
            aux_data = {}
            aux_data["maxbeam"] = self.maxbeam[peak_indexes[i]]
            aux_data["source_index"] = src_idx
            event.set_aux_data(aux_data)
            detections.append(event)

        print(f"Extracted {len(detections):d} events.")

        self.peak_indexes = peak_indexes
        self.source_indexes = source_indexes
        return detections, peak_indexes, source_indexes

    def remove_baseline(self, window, attribute="composite"):
        """Remove baseline from network response."""
        # convert window from seconds to samples
        window = int(window * self.sampling_rate)
        attr_baseline = self._baseline(getattr(self, attribute), window)
        setattr(self, attribute, getattr(self, attribute) - attr_baseline)

    def return_pd_series(self, attribute="maxbeam"):
        """Return the network response as a Pandas.Series."""
        import pandas as pd

        time_series = getattr(self, attribute)
        indexes = pd.date_range(
            start=str(self.starttime),
            freq="{}S".format(1.0 / self.data.sr),
            periods=len(time_series),
        )
        pd_attr = pd.Series(data=time_series, index=indexes)
        return pd_attr

    def smooth_maxbeam(self, window):
        """Smooth the network response with a gaussian kernel."""
        from scipy.ndimage.filters import gaussian_filter1d

        # convert window from seconds to samples
        window = int(window * self.sampling_rate)
        self.smoothed = gaussian_filter1d(self.composite, window)

    def set_data(self, data):
        """Attribute `data` to the class instance.

        Parameters
        ----------
        data: `dataset.Data`
            Instance of `dataset.Data`.
        """
        self.data = data
        self.starttime = self.data.date

    def set_moveouts(self, moveouts, stations, phases):
        """Attribute `_moveouts` to the class instance.

        Parameters
        -----------
        moveouts: (n_sources, n_stations, n_phases) numpy.ndarray
            The moveouts to use for backprojection.
        stations: (n_stations,) list or numpy.ndarray of strings
            Station names corresponding with the second axis of `moveouts`.
        phases: (n_phases,) list or numpy.ndarray of strings
            Phase names corresponing with the third axis of `moveouts`.
        """
        self._moveouts = pd.DataFrame(
            index=stations,
            columns=phases,
        )
        for s, sta in enumerate(stations):
            for ph in enumerate(phases):
                self._moveouts.loc[sta, ph] = moveouts[:, s, p]

    def set_network(self, network):
        """Attribute `network` to the class instance.

        `network` determines which stations are included in the computation
        of the beams. All arrays with a station axis are ordered according to
        `network.stations`.

        Parameters
        ----------
        network: dataset.Network instance
            The Network instance with the station network information.
            `network` can force the network response to be computed only on
            a subset of the data stored in `data`.

        """
        self.network = network

    def set_source_coordinates(self, source_coords):
        """Attribute `_source_coordinates` to the class instance.

        Parameters
        ------------
        source_coords: dictionary
            Dictionary with 3 fields: 'latitude', 'longitude' and 'depth'
        """
        self._source_coordinates = source_coords

    def set_weights(self, weights_phases=None, weights_sources=None):
        """Set the weights required by `beampower`.

        weights_phases: (n_stations, n_channels, n_phases) np.ndarray, optional
            Weight given to each station and channel for a given phase. For
            example, horizontal components might be given a small or zero
            weight for the P-wave stacking.
        weights_sources: (n_sources, n_stations) np.ndarray, optional
            Source-receiver-specific weights. For example, based on the
            source-receiver distance.
        """
        if weights_phases is not None:
            self.weights_phases = weights_phases
        if weights_sources is not None:
            self.weights_sources = weights_sources

    def set_weights_sources(self, n_max_stations, n_min_stations=0):
        """Set network-geometry-based weights of each source-receiver pair.

        Parameters
        ------------
        n_max_stations: scalar, int
            Maximum number of stations used at each theoretical source. Only
            the `n_max_stations` stations will be set a weight > 0.
        """
        weights_sources = np.ones((self.n_sources, self.n_stations), dtype=np.float32)
        self.data.set_availability(self.stations)
        operational_stations = self.data.availability_per_sta.loc[self.stations].values
        mv = self.moveouts[:, operational_stations, 0]
        n_max_stations = min(mv.shape[1], n_max_stations)
        if (n_max_stations < self.n_stations) and (n_max_stations > 0):
            cutoff_mv = np.max(
                np.partition(mv, n_max_stations - 1)[:, :n_max_stations],
                axis=1,
                keepdims=True,
            )
            weights_sources[self.moveouts[:, :, 0] > cutoff_mv] = 0.0
        if n_min_stations > 0:
            n_stations_per_source = np.sum(weights_sources > 0.0, axis=-1)
            weights_sources[n_stations_per_source < n_min_stations, :] = 0.0
        self.weights_sources = weights_sources

    def weights_station_density(
        self, cutoff_dist=None, lower_percentile=0.0, upper_percentile=100.0
    ):
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
                self.network.interstation_distances.values[
                    self.network.interstation_distances.values != 0.0
                ]
            )
        weights_sta_density = np.zeros(self.network.n_stations, dtype=np.float32)
        for s, sta in enumerate(self.network.stations):
            dist_sj = self.network.interstation_distances.loc[sta]
            weights_sta_density[s] = 1.0 / np.sum(
                np.exp(-(dist_sj**2) / cutoff_dist**2)
            )
        if lower_percentile > 0.0:
            weights_sta_density = np.clip(
                weights_sta_density,
                np.percentile(weights_sta_density, lower_percentile),
                weights_sta_density.max(),
            )
        if upper_percentile < 100.0:
            weights_sta_density = np.clip(
                weights_sta_density,
                weights_sta_density.min(),
                np.percentile(weights_sta_density, upper_percentile),
            )
        return weights_sta_density

    def _baseline(self, X, window):
        """Compute a baseline.

        The baseline is a curve connecting the local minima. Removing the
        baseline is equivalent to some kind of low-pass filtering.
        """
        n_windows = np.int32(np.ceil(X.size / window))
        minima = np.zeros(n_windows, dtype=X.dtype)
        minima_args = np.zeros(n_windows, dtype=np.int32)
        for i in range(n_windows):
            minima_args[i] = i * window + X[i * window : (i + 1) * window].argmin()
            minima[i] = X[minima_args[i]]
        # ----------------------------------------
        # --------- build interpolation ----------
        interpolator = interp1d(
            minima_args, minima, kind="linear", fill_value="extrapolate"
        )
        bline = interpolator(np.arange(X.size))
        return bline

    # -------------------------------------------
    #       Plotting methods
    # -------------------------------------------
    def plot_maxbeam(self, ax=None, detection=None, **kwargs):
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
            # plot the maximum beam
            fig = plt.figure(
                    "maximum_beam", figsize=kwargs.get("figsize", (15, 10))
                    )
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        ax.plot(
                self.data.time, self.maxbeam, rasterized=kwargs.get("rasterized", True)
                )
        if hasattr(self, "detection_threshold"):
            ax.plot(
                self.data.time,
                self.detection_threshold,
                color="C3",
                ls="--",
                label="Detection Threshold",
            )
            ax.plot(
                self.data.time[self.peak_indexes],
                self.maxbeam[self.peak_indexes],
                marker="o",
                ls="",
                color="C3",
            )
        ax.legend(loc="upper right")
        ax.set_xlabel("Time of the day")
        ax.set_ylabel("Maximum Beam")

        ax.set_xlim(self.data.time.min(), self.data.time.max())

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax.legend(loc="upper right")

        if detection is not None:
            ot = np.datetime64(detection.origin_time)
            ax.annotate(
                "detection",
                (ot, detection.aux_data["maxbeam"]),
                (
                    ot + np.timedelta64(15, "m"),
                    min(ax.get_ylim()[1], 2.0 * detection.aux_data["maxbeam"]),
                ),
                arrowprops={"width": 2, "headwidth": 5, "color": "k"},
            )
        return fig

    def plot_detection(
        self,
        detection,
        figsize=(20, 20),
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
        n_stations=None,
    ):
        """Plot a detection and the maximum beam.

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
            stations = (
                detection.moveouts[detection.moveouts.columns[0]]
                .sort_values()[:n_stations]
                .index
            )

        sr = self.data.sr
        fig = plt.figure(f"detection_{detection.origin_time}", figsize=figsize)
        grid = fig.add_gridspec(
            nrows=len(stations) + 2, ncols=len(self.network.components), hspace=0.35
        )
        start_times, end_times = [], []
        wav_axes = []
        ax_maxbeam = fig.add_subplot(grid[:2, :])
        self.plot_maxbeam(ax=ax_maxbeam, detection=detection)
        ax_maxbeam.set_ylim(
                max(-0.5, ax_maxbeam.get_ylim()[0]), 2.0 * detection.aux_data["maxbeam"]
                )
        beam = 0.0
        for s, sta in enumerate(stations):
            for c, cp in enumerate(self.network.components):
                ax = fig.add_subplot(grid[2 + s, c])
                for cp_alias in component_aliases[cp]:
                    tr = detection.traces.select(station=sta, component=cp_alias)
                    if len(tr) > 0:
                        # succesfully retrieved data
                        break
                if len(tr) == 0:
                    continue
                else:
                    tr = tr[0]
                time = utils.time_range(
                    tr.stats.starttime, tr.stats.endtime + 1.0 / sr, 1.0 / sr, unit="ms"
                )
                start_times.append(time[0])
                end_times.append(time[-1])
                ax.plot(
                    time[: detection.n_samples],
                    tr.data[: detection.n_samples],
                    color="k",
                )
                # plot the theoretical pick
                phase = detection.aux_data[f"phase_on_comp{cp_alias}"].upper()
                offset_ph = detection.aux_data[f"offset_{phase}"]
                ax.axvline(
                    time[0] + np.timedelta64(int(1000.0 * offset_ph), "ms"), color="C3"
                )
                ax.text(0.05, 0.05, f"{sta}.{cp_alias}", transform=ax.transAxes)
                wav_axes.append(ax)
        for ax in wav_axes:
            ax.set_xlim(min(start_times), max(end_times))
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
            )
        plt.subplots_adjust(top=0.95, bottom=0.06, right=0.98, left=0.06)
        return fig

    def plot_likelihood(self, likelihood=None, time_index=None, **kwargs):
        """Plot likelihood (beam) slices at a given time."""
        from cartopy.crs import PlateCarree
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from . import plotting_utils

        if time_index is None:
            src_idx, time_index = np.unravel_index(self.beam.argmax(), self.beam.shape)
        if likelihood is None:
            likelihood = self._likelihood(self.beam[:, time_index])
        # define slices
        longitude = self.source_coordinates["longitude"].iloc[src_idx]
        latitude = self.source_coordinates["latitude"].iloc[src_idx]
        depth = self.source_coordinates["depth"].iloc[src_idx]
        hor_slice = np.where(self.source_coordinates["depth"] == depth)[0]
        lon_slice = np.where(self.source_coordinates["latitude"] == latitude)[0]
        lat_slice = np.where(self.source_coordinates["longitude"] == longitude)[0]
        # initialize map
        data_coords = PlateCarree()
        lat_min = np.min(self.source_coordinates["latitude"].iloc[hor_slice])
        lat_max = np.max(self.source_coordinates["latitude"].iloc[hor_slice])
        lon_min = np.min(self.source_coordinates["longitude"].iloc[hor_slice])
        lon_max = np.max(self.source_coordinates["longitude"].iloc[hor_slice])
        ax = plotting_utils.initialize_map(
            [lon_min, lon_max],
            [lat_min, lat_max],
            **kwargs,
        )
        fig = ax.get_figure()
        mappable = ax.tricontourf(
            self.source_coordinates["longitude"].iloc[hor_slice],
            self.source_coordinates["latitude"].iloc[hor_slice],
            likelihood[hor_slice],
            levels=np.linspace(0., 1.0, 10),
            cmap="inferno",
            alpha=0.50,
            transform=data_coords,
            zorder=-1,
        )
        ax.scatter(
            self.network.longitude,
            self.network.latitude,
            marker="v",
            color="k",
            s=50,
            transform=data_coords,
            zorder=-1.5,
        )
        # add slices
        divider = make_axes_locatable(ax)
        ax_lon = divider.append_axes("bottom", size="50%", pad=0.2, axes_class=plt.Axes)
        ax_lat = divider.append_axes("right", size="50%", pad=0.2, axes_class=plt.Axes)
        projected_coords = ax.projection.transform_points(
            data_coords,
            self.source_coordinates["longitude"].iloc[lon_slice],
            self.source_coordinates["latitude"].iloc[lon_slice],
        )
        ax_lon.tricontourf(
            projected_coords[..., 0],
            self.source_coordinates["depth"].iloc[lon_slice],
            likelihood[lon_slice],
            levels=np.linspace(0.0, 1.0, 10),
            cmap="inferno",
            alpha=0.50,
            zorder=-1,
        )
        plt.setp(ax_lon.get_xticklabels(), visible=False)
        ax_lon.invert_yaxis()
        ax_lon.set_ylabel("Depth (km)")
        projected_coords = ax.projection.transform_points(
            data_coords,
            self.source_coordinates["longitude"].iloc[lat_slice],
            self.source_coordinates["latitude"].iloc[lat_slice],
        )
        ax_lat.tricontourf(
            self.source_coordinates["depth"].iloc[lat_slice],
            projected_coords[..., 1],
            likelihood[lat_slice],
            levels=np.linspace(0.0, 1.0, 10),
            cmap="inferno",
            alpha=0.50,
            zorder=-1,
        )
        plt.setp(ax_lat.get_yticklabels(), visible=False)
        ax_lat.set_xlabel("Depth (km)")
        cax = divider.append_axes("top", size="3%", pad=0.1, axes_class=plt.Axes)
        plt.colorbar(
            mappable, cax=cax, label="Location Likelihood", orientation="horizontal"
        )
        cax.xaxis.set_label_position("top")
        cax.xaxis.tick_top()

        return fig

    def _rectangular_domain(self, lon0, lat0, side_km=100.0):
        """Return a boolean array indicating which points in a given grid are inside
        a rectangular domain centered at the given longitude and latitude.

        Parameters
        ----------
        lon0 : float
            Longitude of the center of the domain, in degrees.
        lat0 : float
            Latitude of the center of the domain, in degrees.
        side_km : float, optional
            Length of the sides of the rectangular domain, in kilometers.
            Default is 100km.

        Returns
        -------
        selection : ndarray
            1-D boolean array indicating which grid points are inside the domain.

        Notes
        -----
        This function uses the Haversine formula to compute the distances between
        the center of the domain and each grid point. It assumes a spherical Earth
        of radius 6371.0 km.
        """
        R_earth_km = 6371.0  # km
        colat0 = 90.0 - lat0
        Rlat = R_earth_km * np.sin(np.deg2rad(colat0))
        dist_per_lat = 2.0 * np.pi * (1.0 / 360.0) * Rlat
        dist_per_lon = 2.0 * np.pi * (1.0 / 360.0) * R_earth_km
        longitudes = self.source_coordinates["longitude"].values
        latitudes = self.source_coordinates["latitude"].values
        selection = (np.abs(longitudes - lon0) * dist_per_lon < side_km / 2.0) & (
            np.abs(latitudes - lat0) * dist_per_lat < side_km / 2.0
        )
        return selection

    def _compute_location_uncertainty(
            self, event_longitude, event_latitude, event_depth, likelihood, domain
            ):
        """
        Compute the horizontal and vertical uncertainties of an event location.

        Parameters
        ----------
        event_longitude : float
            The longitude, in decimal system, of the event located with the
            present Beamformer instance.
        event_latitude : float
            The latitude, in decimal system, of the event located with the
            present Beamformer instance.
        event_depth : float
            The depth, in km, of the event located with the present
            Beamformer instance.
        likelihood : ndarray
            A 1D numpy array containing the likelihood of each source location in
            the beamformer domain.
        domain : ndarray
            A 1D numpy array containing the indices of the beamformer domain.

        Returns
        -------
        tuple
            A tuple of two floats representing the horizontal and vertical
            uncertainties of the event location in km, respectively.

        Notes
        -----
        The horizontal uncertainty is computed as the weighted average of the
        distance from each source in the domain to the event location, with the
        weight being the likelihood of each source. The vertical uncertainty is
        computed as the weighted average of the absolute depth difference between
        each source in the domain and the event depth, with the weight being the
        likelihood of each source. The distance is calculated using the Geodesic
        library from the Cartopy package.

        """
        from cartopy.geodesic import Geodesic
        # initialize the Geodesic instance
        G = Geodesic()
        epi_distances = G.inverse(
            np.array([event_longitude, event_latitude]),
            np.hstack(
                (
                    self.source_coordinates["longitude"].values[domain, None],
                    self.source_coordinates["latitude"].values[domain, None],
                )
            ),
        )
        pointwise_distances = np.asarray(epi_distances)[:, 0].squeeze() / 1000.0

        # horizontal uncertainty
        hunc = np.sum(likelihood[domain]*pointwise_distances)/np.sum(likelihood[domain])

        # vertical uncertainty
        depth_diff = np.abs(event_depth - self.source_coordinates["depth"].values[domain])
        vunc = np.sum(likelihood[domain]*depth_diff)/np.sum(likelihood[domain])
        
        return hunc, vunc

def _detect_peaks(
    x,
    mph=None,
    mpd=1,
    threshold=0,
    edge="rising",
    kpsh=False,
    valley=False,
    show=False,
    ax=None,
):
    """see `utils._detect_peaks`.

    """
    return utils._detect_peaks(
            x, mph=mph, mpd=mpd, threshold=threshold, edge=edge,
            kpsh=kpsh, valley=valley, show=show, ax=ax
            )


def _plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available.")
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, "b", lw=1)
        if ind.size:
            label = "valley" if valley else "peak"
            label = label + "s" if ind.size > 1 else label
            ax.plot(
                ind,
                x[ind],
                "+",
                mfc=None,
                mec="r",
                mew=2,
                ms=8,
                label="{:d} {}".format(ind.size, label),
            )
            ax.legend(loc="best", framealpha=0.5, numpoints=1)
        ax.set_xlim(-0.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel("Data #", fontsize=14)
        ax.set_ylabel("Amplitude", fontsize=14)
        mode = "Valley detection" if valley else "Peak detection"
        ax.set_title(
            "{} (mph={}, mpd={:d}, threshold={}, edge='{}')".format(
                mode, str(mph), mpd, str(threshold), edge
            )
        )
        # plt.grid()
        plt.show()


def baseline(X, w):
    n_windows = np.int32(np.ceil(X.size / w))
    minima = np.zeros(n_windows, dtype=X.dtype)
    minima_args = np.zeros(n_windows, dtype=np.int32)
    for i in range(n_windows):
        minima_args[i] = i * w + X[i * w : (i + 1) * w].argmin()
        minima[i] = X[minima_args[i]]
    # ----------------------------------------
    # --------- build interpolation ----------
    interpolator = interp1d(
        minima_args, minima, kind="linear", fill_value="extrapolate"
    )
    bline = interpolator(np.arange(X.size))
    return bline


def time_dependent_threshold(
    network_response, window, overlap=0.75, CNR_threshold=cfg.N_DEV_BP_THRESHOLD
):
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
    shift = int((1.0 - overlap) * window)
    n_windows = int((len(network_response) - window) // shift) + 1
    mad_ = np.zeros(n_windows + 2, dtype=np.float32)
    med_ = np.zeros(n_windows + 2, dtype=np.float32)
    time = np.zeros(n_windows + 2, dtype=np.float32)
    for i in range(1, n_windows + 1):
        i1 = i * shift
        i2 = min(network_response.size, i1 + window)
        maxbeam_window = network_response[i1:i2]
        # non_zero = maxbeam_window != 0
        # if sum(non_zero) < 3:
        #    # won't be possible to calculate median
        #    # and mad on that few samples
        #    continue
        # med_[i] = np.median(maxbeam_window[non_zero])
        # mad_[i] = scimad(maxbeam_window[non_zero])
        med_[i] = np.median(maxbeam_window)
        mad_[i] = scimad(maxbeam_window)
        time[i] = (i1 + i2) / 2.0
    # add boundary cases manually
    time[0] = 0.0
    mad_[0] = mad_[1]
    med_[0] = med_[1]
    time[-1] = len(network_response)
    mad_[-1] = mad_[-2]
    med_[-1] = med_[-2]
    threshold = med_ + CNR_threshold * mad_
    interpolator = interp1d(
        time,
        threshold,
        kind="slinear",
        fill_value=(threshold[0], threshold[-1]),
        bounds_error=False,
    )
    full_time = np.arange(0, len(network_response))
    threshold = interpolator(full_time)
    return threshold


def time_dependent_threshold_pd(network_response, window):
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
    detection_threshold = run_med + cfg.N_DEV_BP_THRESHOLD * run_mad
    return detection_threshold.values


# ---------------------------------------------------------------
#                      Detection traces
# ---------------------------------------------------------------


def saturated_envelopes(traces, anomaly_threshold=1.0e-11, max_dynamic_range=1.0e5):
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
    waveform_features = envelope_parallel(
        traces
    )  # take the upper envelope of the traces
    tend = give_time()
    print(f"Computed the envelopes in {tend-tstart:.2f}sec.")
    data_availability = np.zeros(n_stations, dtype=np.int32)
    for s in range(n_stations):
        for c in range(n_components):
            missing_samples = waveform_features[s, c, :] == 0.0
            if np.sum(missing_samples) > waveform_features.shape[-1] / 2:
                # too many samples are missing, don't use this trace
                # do not increment data_availability
                waveform_features[s, c, :] = 0.0
                continue
            median = np.median(waveform_features[s, c, ~missing_samples])
            mad = scimad(waveform_features[s, c, ~missing_samples])
            if mad < anomaly_threshold:
                waveform_features[s, c, :] = 0.0
                continue
            waveform_features[s, c, :] = (waveform_features[s, c, :] - median) / mad
            waveform_features[s, c, missing_samples] = 0.0
            # saturate traces
            waveform_features[s, c, :] = np.clip(
                waveform_features[s, c, :],
                waveform_features[s, c, :],
                max_dynamic_range,
            )
            data_availability[s] += 1
    return waveform_features, data_availability


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
        envelopes = np.float32(list(executor.map(envelope, traces_reshaped)))
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
