import os

import numpy as np
import h5py as h5

from .config import cfg
from . import utils

import obspy as obs
import pandas as pd
import pathlib
import copy
import datetime
from obspy import UTCDateTime as udt

from abc import ABC, abstractmethod
from tqdm import tqdm

from time import time as give_time
from time import sleep


class Network(object):
    """Station metadata.

    Contains station metadata.
    """

    def __init__(self, network_file):
        """
        Parameters
        -----------
        network_file: string
            Name of the station metadata file.
        """
        self.where = os.path.join(cfg.NETWORK_PATH, network_file)

    @property
    def n_stations(self):
        return np.int32(len(self.stations))

    @property
    def n_components(self):
        return np.int32(len(self.components))

    @property
    def stations(self):
        return self.metadata["stations"]

    @property
    def station_indexes(self):
        return pd.Series(index=self.stations, data=np.arange(len(self.stations)))

    @property
    def networks(self):
        return self.metadata["networks"]

    @property
    def latitude(self):
        return self.metadata["latitude"]

    @property
    def longitude(self):
        return self.metadata["longitude"]

    @property
    def depth(self):
        return self.metadata["depth_km"]

    @property
    def elevation(self):
        return self.metadata["elevation_m"]

    def box(self, lat_min, lat_max, lon_min, lon_max):
        """Geographical selection of sub-network.

        Parameters
        ------------
        lat_min: scalar, float
            Minimum latitude of the box.
        lat_max: scalar, float
            Maximum latitude of the box.
        lon_min: scalar, float
            Minimum longitude of the box.
        lon_max: scalar, float
            Maximum longitude of the box.

        Returns
        ---------------
        subnet: Network instance
            The Network instance restricted to the relevant stations.
        """
        selection = (
            (self.latitude > lat_min)
            & (self.latitude < lat_max)
            & (self.longitude > lon_min)
            & (self.longitude < lon_max)
        )
        new_stations = np.asarray(self.stations[selection]).astype("U")
        subnet = self.subset(new_stations, self.components, method="keep")
        return subnet

    def datelist(self):
        return pd.date_range(start=str(self.start_date), end=str(self.end_date))

    def read(self):
        """
        Reads the metadata from the file at self.where

        Note: This function can be modified to match the user's
        data convention.
        """
        with open(self.where, "r") as fin:
            line1 = fin.readline()[:-1].split()
            self.start_date = udt(line1[0])
            self.end_date = udt(line1[1])
            line2 = fin.readline()[:-1].split()
            self.components = line2
        metadata = pd.read_csv(self.where, sep="\t", skiprows=2)
        metadata.rename(
            columns={"station_code": "stations", "network_code": "networks"},
            inplace=True,
        )
        metadata["depth_km"] = -1.0 * metadata["elevation_m"] / 1000.0  # depth in km
        self.metadata = metadata
        self.metadata.set_index("stations", inplace=True, drop=False)

    def stations_idx(self, stations):
        # if not isinstance(stations, list) and not isinstance(stations, np.ndarray):
        #    stations = [stations]
        # idx = []
        # for station in stations:
        #    idx.append(self.stations.index(station))
        self.station_indexes.loc[stations]
        return idx

    def subset(self, stations, components, method="keep"):
        """
        Parameters
        -----------
        stations: list or array of strings
            Stations to keep or discard, depending on the method.
        components: list or array of strings
            Components to keep or discard, depending on the method.
        method: string, default to 'keep'
            Should be 'keep' or 'discard'.
            If 'keep', the stations and components provided to
            this function are what will be left in the subnetwork.
            If 'discard', the stations and components provided to
            this function are what won't be featured in the
            subnetwork.

        Returns
        ----------
        subnetwork
        """
        subnetwork = copy.deepcopy(self)

        if isinstance(stations, np.ndarray):
            stations = stations.tolist()
        elif not isinstance(stations, list):
            stations = [stations]
        if isinstance(components, np.ndarray):
            components = components.tolist()
        elif not isinstance(components, list):
            components = [components]

        if method == "discard":
            subnetwork.metadata.drop(stations, axis="rows", inplace=True)
            for component in components:
                if component in self.components:
                    idx = subnetwork.components.index(component)
                    subnetwork.components.remove(component)
                else:
                    print("{} not a network component".format(station))
        elif method == "keep":
            subnetwork.metadata = subnetwork.metadata.loc[stations]
            subnetwork.components = components
        else:
            print('method should be "keep" or "discard"!')
            return
        return subnetwork

    @property
    def interstation_distances(self):
        """Compute the distance between all station pairs."""
        # should update code to reuse utils.compute_distamces
        if (
            hasattr(self, "_interstation_distances")
            and self._interstation_distances.shape[0] == self.n_stations
        ):
            # was already computed and the size of the network was unchanged
            return self._interstation_distances
        else:
            from cartopy.geodesic import Geodesic

            G = Geodesic()

            intersta_dist = np.zeros(
                (len(self.stations), len(self.stations)), dtype=np.float64
            )
            for s in range(len(self.stations)):
                d = G.inverse(
                    np.array([[self.longitude[s], self.latitude[s]]]),
                    np.hstack(
                        (
                            self.longitude.values.reshape(-1, 1),
                            self.latitude.values.reshape(-1, 1),
                        )
                    ),
                )
                # d is in m, convert it to km
                d = np.asarray(d)[:, 0] / 1000.0
                intersta_dist[s, :] = np.sqrt(
                    d.squeeze() ** 2 + (self.depth[s] - self.depth)
                )

            # return distance in km
            self._interstation_distances = pd.DataFrame(
                index=self.stations, columns=self.stations, data=intersta_dist
            )
            return self._interstation_distances

    # plotting method
    def plot_map(self, ax=None, figsize=(20, 10), **kwargs):
        """Plot stations on map.

        Parameters
        ------------
        ax: `plt.Axes`, default to None
            If None, create a new `plt.Figure` and `plt.Axes` instances. If
            speficied by user, use the provided instance to plot.
        figsize: tuple of floats, default to (20, 10)
            Size, in inches, of the figure (width, height).

        Returns
        ----------
        fig: `plt.Figure`
            The map with seismic stations.
        """
        from . import plotting_utils
        import matplotlib.pyplot as plt
        from cartopy.crs import PlateCarree

        cmap = kwargs.get("cmap", None)
        if cmap is None:
            try:
                import colorcet as cc

                cmap = cc.cm.fire_r
            except Exception as e:
                print(e)
                cmap = "hot_r"
        data_coords = PlateCarree()
        lat_margin = kwargs.get("lat_margin", 0.1)
        lon_margin = kwargs.get("lon_margin", 0.1)
        # ------------------------------------------------
        #           Scattering plot kwargs
        scatter_kwargs = {}
        scatter_kwargs["edgecolor"] = kwargs.get("edgecolor", "k")
        scatter_kwargs["linewidths"] = kwargs.get("linewidths", 0.5)
        scatter_kwargs["s"] = kwargs.get("s", 10)
        scatter_kwargs["zorder"] = kwargs.get("zorder", 1)
        # ------------------------------------------------
        map_longitudes = [
            min(self.longitude) - lon_margin,
            max(self.longitude) + lon_margin,
        ]
        map_latitudes = [
            min(self.latitude) - lat_margin,
            max(self.latitude) + lat_margin,
        ]
        seismic_stations = {
            "longitude": self.longitude,
            "latitude": self.latitude,
            "stations": self.stations,
        }
        ax = plotting_utils.initialize_map(
            map_longitudes,
            map_latitudes,
            figsize=figsize,
            map_axis=ax,
            seismic_stations=seismic_stations,
            **kwargs,
        )
        return ax.get_figure()


class Catalog(object):
    """A class for catalog data, and basic plotting."""

    def __init__(
        self, longitudes, latitudes, depths, origin_times, event_ids=None, **kwargs
    ):
        """Initialize a catalog attribute as a pandas.DataFrame.

        Parameters
        -----------
        longitudes: List or numpy.ndarray of floats
            Event longitudes.
        latitudes: List or numpy.ndarray of floats
            Event latitudes.
        depths: List or numpy.ndarray of floats
            Event depths.
        origin_times: List or numpy.ndarray of strings or datetimes
            Event origin times.
        event_ids: List or numpy.ndarray, default to None
            If not None, is used to define named indexes of the rows.
        """
        catalog = {}
        catalog["longitude"] = longitudes
        catalog["latitude"] = latitudes
        catalog["depth"] = depths
        catalog["origin_time"] = pd.to_datetime(np.datetime64(origin_times))
        catalog.update(kwargs)
        if event_ids is not None:
            catalog["event_id"] = event_ids
        self.catalog = pd.DataFrame(catalog)
        if event_ids is not None:
            self.catalog.set_index("event_id", inplace=True)
        self.catalog.sort_values("origin_time", inplace=True)

    @property
    def latitude(self):
        return self.catalog.latitude

    @property
    def longitude(self):
        return self.catalog.longitude

    @property
    def depth(self):
        return self.catalog.depth

    @property
    def origin_time(self):
        return self.catalog.origin_time

    @classmethod
    def concatenate(cls, catalogs, ignore_index=True):
        """Build catalog from list of `pandas.DataFrame`.

        Parameters
        -----------
        catalogs: list of `pandas.DataFrame`
            List of `pandas.DataFrame` with consistent columns.
        """
        cat = pd.concat(catalogs, ignore_index=ignore_index)
        cat.sort_values("origin_time", inplace=True)
        base = ["longitude", "latitude", "depth", "origin_time"]
        return cls(
            cat.longitude,
            cat.latitude,
            cat.depth,
            cat.origin_time,
            **cat.drop(columns=base),
        )

    @classmethod
    def read_from_events(cls, events, extra_attributes=[], fill_value=np.nan):
        """Build catalog from list of `Event` instances."""
        longitudes, latitudes, depths, origin_times = [], [], [], []
        extra_attr = {}
        # initialize empty lists for extra requested attributes
        for attr in extra_attributes:
            extra_attr[attr] = []
        for event in events:
            longitudes.append(event.longitude)
            latitudes.append(event.latitude)
            depths.append(event.depth)
            origin_times.append(str(event.origin_time))
            for attr in extra_attributes:
                if hasattr(event, attr):
                    extra_attr[attr].append(getattr(event, attr))
                elif hasattr(event, "aux_data") and attr in event.aux_data:
                    # check if attribute is in aux_data
                    extra_attr[attr].append(event.aux_data[attr])
                else:
                    # attribute was not found, fill with default value
                    extra_attr[attr].append(fill_value)
        return cls(longitudes, latitudes, depths, origin_times, **extra_attr)

    @classmethod
    def read_from_dataframe(cls, dataframe):
        """Initialize a Catalog instance from a `pandas.DataFrame` instance.
        """
        catalog = cls(
                dataframe["longitude"],
                dataframe["latitude"],
                dataframe["depth"],
                dataframe["origin_time"],
                **dataframe.drop(
                    columns=["longitude", "latitude", "depth", "origin_time"]
                    )
                )
        return catalog

    @classmethod
    def read_from_detection_file(
        cls,
        filename,
        db_path=cfg.OUTPUT_PATH,
        gid=None,
        extra_attributes=[],
        fill_value=np.nan,
        return_events=False,
    ):
        """Read all detected events and build catalog."""
        events = []
        try:
            with h5.File(os.path.join(db_path, filename), mode="r") as f:
                if gid is not None:
                    f = f[gid]
                keys = list(f.keys())
                for key in f.keys():
                    events.append(Event.read_from_file(hdf5_file=f[key]))
        except Exception as e:
            print(e)
            print(
                "Error while trying to read the detected events "
                "(perhaps there are none)."
            )
            pass
        if return_events:
            return (
                cls.read_from_events(
                    events, extra_attributes=extra_attributes, fill_value=fill_value
                ),
                events,
            )
        else:
            return cls.read_from_events(
                events, extra_attributes=extra_attributes, fill_value=fill_value
            )

    # ---------------------------------------------------------
    #                  Plotting methods
    # ---------------------------------------------------------
    def plot_time_statistics(self, figsize=(16, 7), **kwargs):
        """Plot the histograms of time of the day and day of the week.

        Parameters
        ------------
        figsize: tuple of floats, default to (16, 7)
            Size, in inches, of the figure (width, height).

        Returns
        ---------
        fig: `plt.Figure`
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            num="time_statistics", ncols=2, nrows=1, figsize=figsize
        )
        self.catalog["origin_time"].dt.dayofweek.hist(bins=np.arange(8), ax=axes[0])
        axes[0].set_xticks(0.5 + np.arange(7))
        axes[0].set_xticklabels(["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"])
        axes[0].set_xlabel("Day of the Week")
        axes[0].set_ylabel("Event Count")

        self.catalog["origin_time"].dt.hour.hist(bins=np.arange(25), ax=axes[1])
        axes[1].set_xlabel("Hour of the Day")
        axes[1].set_ylabel("Event Count")
        return fig

    def plot_map(
        self,
        ax=None,
        figsize=(20, 10),
        depth_min=0.0,
        depth_max=20.0,
        network=None,
        **kwargs
    ):
        """Plot epicenters on map.

        Parameters
        ------------
        ax: `plt.Axes`, default to None
            If None, create a new `plt.Figure` and `plt.Axes` instances. If
            speficied by user, use the provided instance to plot.
        figsize: tuple of floats, default to (20, 10)
            Size, in inches, of the figure (width, height).
        depth_min: scalar float, default to 0
            Smallest depth, in km, in the depth colormap.
        depth_max: scalar float, default to 20
            Largest depth, in km, in the depth colormap.

        Returns
        ----------
        fig: `plt.Figure`
            The figure with depth color-coded epicenters.
        """
        from . import plotting_utils
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from cartopy.crs import PlateCarree

        cmap = kwargs.get("cmap", None)
        if cmap is None:
            try:
                import colorcet as cc

                cmap = cc.cm.fire_r
            except Exception as e:
                print(e)
                cmap = "hot_r"
        data_coords = PlateCarree()
        lat_margin = kwargs.get("lat_margin", 0.1)
        lon_margin = kwargs.get("lon_margin", 0.1)
        # ------------------------------------------------
        #           Scattering plot kwargs
        scatter_kwargs = {}
        scatter_kwargs["edgecolor"] = kwargs.get("edgecolor", "k")
        scatter_kwargs["linewidths"] = kwargs.get("linewidths", 0.5)
        scatter_kwargs["s"] = kwargs.get("s", 10)
        scatter_kwargs["zorder"] = kwargs.get("zorder", 1)
        # ------------------------------------------------
        if network is None:
            map_longitudes = [
                min(self.longitude) - lon_margin,
                max(self.longitude) + lon_margin,
            ]
            map_latitudes = [
                min(self.latitude) - lat_margin,
                max(self.latitude) + lat_margin,
            ]
        else:
            map_longitudes = [
                min(self.longitude.tolist() + network.longitude.tolist()) - lon_margin,
                max(self.longitude.tolist() + network.longitude.tolist()) + lon_margin,
            ]
            map_latitudes = [
                min(self.latitude.tolist() + network.latitude.tolist()) - lat_margin,
                max(self.latitude.tolist() + network.latitude.tolist()) + lat_margin,
            ]
        ax = plotting_utils.initialize_map(
            map_longitudes, map_latitudes, figsize=figsize, map_axis=ax, **kwargs
        )
        # plot epicenters
        cNorm = Normalize(vmin=depth_min, vmax=depth_max)
        scalar_map = ScalarMappable(norm=cNorm, cmap=cmap)

        ax.scatter(
            self.longitude,
            self.latitude,
            c=scalar_map.to_rgba(self.depth.values),
            label="Earthquakes",
            **scatter_kwargs,
            transform=data_coords,
        )
        if network is not None:
            ax.scatter(
                network.longitude,
                network.latitude,
                c="magenta",
                marker="v",
                label="Seismic stations",
                s=kwargs.get("markersize_station", 10),
                transform=data_coords,
            )
        ax.legend(loc="lower right")
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="2%", pad=0.08, axes_class=plt.Axes)
        plt.colorbar(scalar_map, cax, orientation="vertical", label="Depth (km)")
        return ax.get_figure()

    def plot_space_time(
        self,
        ax=None,
        figsize=(20, 10),
        color_coded="longitude",
        y_axis="latitude",
        **kwargs,
    ):
        """Plot the space-time event distribution.

        Parameters
        ------------
        ax: `plt.Axes`, default to None
            If None, create a new `plt.Figure` and `plt.Axes` instances. If
            speficied by user, use the provided instance to plot.
        figsize: tuple of floats, default to (20, 10)
            Size, in inches, of the figure (width, height).
        color_coded: string, default to 'longitude'
            Can be either 'longitude', 'latitude', or 'depth'. This is the
            attribute used to define the color scale of each dot.
        y_axis: string, default to 'latitude'
            Can be either 'longitude', 'latitude', or 'depth'. This is the
            attribute used to define the y-axis coordinates.

        Returns
        --------
        fig: `plt.Figure`
            The figure with color coded latitudes or longitudes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        cmap = kwargs.get("cmap", None)
        if cmap is None:
            try:
                import colorcet as cc

                cmap = cc.cm.bjy
            except Exception as e:
                print(e)
                cmap = "viridis"
        # ------------------------------------------------
        #           Scattering plot kwargs
        scatter_kwargs = {}
        scatter_kwargs["edgecolor"] = kwargs.get("edgecolor", "k")
        scatter_kwargs["linewidths"] = kwargs.get("linewidths", 0.5)
        scatter_kwargs["s"] = kwargs.get("s", 10)
        scatter_kwargs["zorder"] = kwargs.get("zorder", 0)
        # ------------------------------------------------
        if ax is None:
            fig = plt.figure(kwargs.get("figname", "space_time"), figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        cNorm = Normalize(
            vmin=self.catalog[color_coded].min(), vmax=self.catalog[color_coded].max()
        )
        scalar_map = ScalarMappable(norm=cNorm, cmap=cmap)
        scalar_map.set_array([])

        ax.set_title(f"{len(self.catalog):d} events")
        ax.set_xlabel("Calendar Time")
        ax.set_ylabel(y_axis.capitalize())
        ax.scatter(
            self.catalog["origin_time"],
            self.catalog[y_axis],
            color=scalar_map.to_rgba(self.catalog[color_coded]),
            rasterized=True,
            **scatter_kwargs,
        )
        ax.set_xlim(min(self.catalog["origin_time"]), max(self.catalog["origin_time"]))

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="2%", pad=0.08)
        plt.colorbar(
            scalar_map, cax, orientation="vertical", label=color_coded.capitalize()
        )

        return fig


class Data(object):
    """A Data class to manipulate waveforms and metadata."""

    def __init__(
        self,
        date,
        where,
        data_reader,
        duration=24.0 * 3600.0,
        sampling_rate=None,
    ):
        """
        Parameters
        -----------
        date: string
            Date of the requested day. Example: '2016-01-23'.
        where: string
            Path to root folder or data file itself (depending on the data
            reader you are using).
        data_reader: function
            Function that takes a path and optional key-word arguments to read
            data from this path and returns an `obspy.Stream` instance.
        duration: float, default to 24*3600
            Target duration, in seconds, of the waveform time series. Waveforms
            will be trimmed and zero-padded to match this duration.
        sampling_rate: float or int, default to None
            Sampling rate of the data. This variable should be left to None if
            this Data instance aims at dealing with raw data and multiple
            sampling rates.
        """
        self.date = udt(date)
        # full path:
        self.where = where
        # data reader
        self.data_reader = data_reader
        self.duration = duration
        # fetch metadata
        # self._read_metadata()
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
            self.n_samples = utils.sec_to_samp(duration, sr=self.sampling_rate)

    @property
    def sr(self):
        return self.sampling_rate

    @property
    def time(self):
        if not hasattr(self, "sampling_rate"):
            print("You need to define the instance's sampling rate first.")
            return
        if not hasattr(self, "_time"):
            self._time = utils.time_range(
                self.date, self.date + self.duration, 1.0 / self.sr, unit="ms"
            )
        return self._time

    def get_np_array(
        self,
        stations,
        components=["N", "E", "Z"],
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
        priority="HH",
        verbose=True,
    ):
        if not hasattr(self, "traces"):
            print("You should call read_waveforms first.")
            return None
        return utils.get_np_array(
            self.traces,
            stations,
            components=components,
            priority=priority,
            component_aliases=component_aliases,
            n_samples=self.n_samples,
            verbose=verbose,
        )

    def read_waveforms(self, trim_traces=True, **reader_kwargs):
        """Read the waveform time series.

        Parameters
        -----------
        trim_traces: boolean, default to True
            If True, call `trim_waveforms` to make sure all traces have the same
            start time.
        """
        reader_kwargs.setdefault("starttime", self.date)
        reader_kwargs.setdefault("endtime", self.date + self.duration)
        self.traces = self.data_reader(self.where, **reader_kwargs)
        if trim_traces:
            self.trim_waveforms()

    def set_availability(
        self,
        stations,
        components=["N", "E", "Z"],
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
    ):
        """Set the data availability.

        A station is available if at least one station has non-zero data. The
        availability is then accessed via the property `self.availability`.

        Parameters
        -----------
        stations: list of strings or numpy.ndarray
            Names of the stations on which we check availability. If None, use
            `self.stations`.
        """
        if not hasattr(self, "traces"):
            print("Call `self.read_waveforms` first.")
            return
        self.availability_per_sta = pd.Series(
            index=stations,
            data=np.zeros(len(stations), dtype=bool),
        )
        self.availability_per_cha = pd.DataFrame(index=stations)
        for c, cp in enumerate(components):
            availability = np.zeros(len(stations), dtype=bool)
            for s, sta in enumerate(stations):
                for cp_alias in component_aliases[cp]:
                    tr = self.traces.select(station=sta, component=cp_alias)
                    if len(tr) > 0:
                        tr = tr[0]
                    else:
                        continue
                    if np.sum(tr.data != 0.0):
                        availability[s] = True
                        break
            self.availability_per_cha[cp] = availability
            self.availability_per_sta = np.bitwise_or(
                self.availability_per_sta, availability
            )
        self.availability = self.availability_per_sta

    def trim_waveforms(self, starttime=None, endtime=None):
        """Trim waveforms.

        Start times might differ of one sample on different traces. Use this
        method to make sure all traces have the same start time.

        Parameters
        -----------
        starttime: string or datetime, default to None
            If None, use `self.date` as the start time.
        endtime: string or datetime, default to None
            If None, use `self.date` + `self.duration` as the end time.
        """
        if not hasattr(self, "traces"):
            print("You should call `read_waveforms` first.")
            return
        if starttime is None:
            starttime = self.date
        if endtime is None:
            endtime = self.date + self.duration
        for tr in self.traces:
            tr.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0.0)


class Event(object):
    """An Event class to describe *any* collection of waveforms."""

    def __init__(
        self,
        origin_time,
        moveouts,
        stations,
        phases,
        data_filename,
        data_path,
        latitude=None,
        longitude=None,
        depth=None,
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
        sampling_rate=None,
        components=["N", "E", "Z"],
        id=None,
        data_reader=None,
    ):
        """Initialize an Event instance with basic attributes.

        Parameters
        -----------
        origin_time: string
            Origin time, or detection time, of the event. Phase picks are
            defined by origin_time + moveout.
        moveouts: (n_stations, n_phases) float numpy.ndarray
            Moveouts, in seconds, for each station and each phase.
        stations: List of strings
            List of station names corresponding to `moveouts`.
        phases: List of strings
            List of phase names corresponding to `moveouts`.
        data_filename: string
            Name of the data file.
        data_path: string
            Path to the data directory.
        latitude: scalar float, default to None
            Event latitude.
        longitude: scalar float, default to None
            Event longitude.
        depth: scalar float, default to None
            Event depth.
        sampling_rate: scalar float, default to None
            Sampling rate (Hz) of the waveforms. It should be different from
            None only if you plan on reading preprocessed data with a fixed
            sampling rate.
        components: List of strings, default to ['N','E','Z']
            List of the components to use in reading and plotting methods.
        component_aliases: Dictionary, optional
            Each entry of the dictionary is a list of strings.
            `component_aliases[comp]` is the list of all aliases used for
            the same component 'comp'. For example, `component_aliases['N'] =
            ['N', '1']` means that both the 'N' and '1' channels will be mapped
            to the Event's 'N' channel.
        id: string, default to None
            Identifying label.
        data_reader: function, default to None
            Function that takes a path and optional key-word arguments to read
            data from this path and returns an `obspy.Stream` instance. If None,
            `data_reader` has to be specified when calling `read_waveforms`.
        """
        self.origin_time = udt(origin_time)
        self.date = self.origin_time  # for compatibility with Data class
        self.where = os.path.join(data_path, data_filename)
        self.stations = np.asarray(stations).astype("U")
        self.components = np.asarray(components).astype("U")
        self.component_aliases = component_aliases
        self.phases = np.asarray(phases).astype("U")
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.sampling_rate = sampling_rate
        if moveouts.dtype in (np.int32, np.int64):
            print(
                "Integer data type detected for moveouts. Are you sure these"
                " are in seconds?"
            )
        # format moveouts in a Pandas data frame
        mv_table = {"stations": self.stations}
        for p, ph in enumerate(self.phases):
            mv_table[f"moveouts_{ph.upper()}"] = moveouts[:, p]
        self.moveouts = pd.DataFrame(mv_table)
        self.moveouts.set_index("stations", inplace=True)
        if id is None:
            self.id = self.origin_time.strftime("%Y%m%d_%H%M%S")
        else:
            self.id = id
        self.data_reader = data_reader

    @classmethod
    def read_from_file(
        cls, filename=None, db_path=cfg.INPUT_PATH, hdf5_file=None, gid=None,
        data_reader=None,
    ):
        """Initialize an Event instance from `filename`.

        Parameters
        ------------
        filename: string, default to None
            Name of the hdf5 file with the event's data. If None, then
            `hdf5_file` should be specified.
        db_path: string, default to `cfg.INPUT_PATH`
            Name of the directory where `filename` is located.
        gid: string, default to None
            If not None, this string is the hdf5's group name of the event.
        hdf5_file: `h5py.File`, default to None
            If not None, is an opened file pointing directly at the subfolder of
            interest.
        data_reader: function, default to None
            Function that takes a path and optional key-word arguments to read
            data from this path and returns an `obspy.Stream` instance. If None,
            `data_reader` has to be specified when calling `read_waveforms`.

        Returns
        ----------
        event: `Event` instance
            The `Event` instance defined by the data in `filename`.
        """
        attributes = ["origin_time", "moveouts", "stations", "phases"]
        optional_attr = [
            "latitude",
            "longitude",
            "depth",
            "sampling_rate",
            "compoments",
            "id",
        ]
        args = []
        kwargs = {}
        has_picks = False
        has_arrivals = False
        close_file = False
        if filename is not None:
            parent_file = h5.File(os.path.join(db_path, filename), mode="r")
            if gid is not None:
                # go to specified group
                f = parent_file[gid]
            else:
                f = parent_file
            close_file = True  # remember to close file at the end
        else:
            f = hdf5_file
        for attr in attributes:
            args.append(f[attr][()])
        if type(f["where"][()]) == bytes:
            # if h5py.version >= 3
            data_path, data_filename = os.path.split(f["where"][()].decode("utf-8"))
        else:
            # more recent versions of h5py seems to decode automatically
            data_path, data_filename = os.path.split(f["where"][()])
        args.extend([data_filename, data_path])
        for opt_attr in optional_attr:
            if opt_attr in f:
                kwargs[opt_attr] = f[opt_attr][()]
        aux_data = {}
        if "aux_data" in f:
            for key in f["aux_data"].keys():
                aux_data[key] = f["aux_data"][key][()]
                if type(aux_data[key]) == bytes:
                    aux_data[key] = aux_data[key].decode("utf-8")
        if "picks" in f:
            picks = {}
            for key in f["picks"].keys():
                picks[key] = f["picks"][key][()]
                if picks[key].dtype.kind == "S":
                    picks[key] = picks[key].astype("U")
                    if key != "stations":
                        picks[key] = pd.to_datetime(picks[key])
            picks = pd.DataFrame(picks)
            picks.set_index("stations", inplace=True)
            has_picks = True
        if "arrival_times" in f:
            arrival_times = {}
            for key in f["arrival_times"].keys():
                arrival_times[key] = f["arrival_times"][key][()]
                if arrival_times[key].dtype.kind == "S":
                    arrival_times[key] = arrival_times[key].astype("U")
                    if key != "stations":
                        arrival_times[key] = pd.to_datetime(arrival_times[key])
            arrival_times = pd.DataFrame(arrival_times)
            arrival_times.set_index("stations", inplace=True)
            has_arrivals = True
        if close_file:
            # close the file
            parent_file.close()
        # ! the order of args is important !
        kwargs["data_reader"] = data_reader
        event = cls(*args, **kwargs)
        if "cov_mat" in aux_data:
            event.cov_mat = aux_data["cov_mat"]
        event.set_aux_data(aux_data)
        if has_picks:
            event.picks = picks
        if has_arrivals:
            event.arrival_times = arrival_times
        if gid is not None:
            # keep trace that we read from a group
            event.hdf5_gid = gid
        return event

    @property
    def availability(self):
        if hasattr(self, "aux_data") and "availability" in self.aux_data:
            return self.aux_data["availability"].loc[self.stations]
        else:
            print("Call `self.set_availability` first.")
            return

    @property
    def availability_per_sta(self):
        if hasattr(self, "_availability_per_sta"):
            return self._availability_per_sta
        if hasattr(self, "aux_data") and "availability_per_sta" in self.aux_data:
            return self.aux_data["availability_per_sta"].loc[self.stations]
        else:
            print("Call `self.set_availability` first.")
            return

    @property
    def availability_per_cha(self):
        if hasattr(self, "_availability_per_cha"):
            return self._availability_per_cha
        availability = pd.DataFrame(index=self.stations, columns=self.components)
        for cp in self.components:
            if hasattr(self, "aux_data") and f"availability_{cp}" in self.aux_data:
                availability.loc[self.stations, cp] = self.aux_data[
                    f"availability_{cp}"
                ].loc[self.stations]
            else:
                print("Call `self.set_availability` first.")
                return
        return availability

    @property
    def hmax_unc(self):
        if hasattr(self, "_hmax_unc"):
            return self._hmax_unc
        elif hasattr(self, "aux_data") and "hmax_unc" in self.aux_data:
            return self.aux_data["hmax_unc"]
        else:
            self.hor_ver_uncertainties()
            return self._hmax_unc

    @property
    def hmin_unc(self):
        if hasattr(self, "_hmin_unc"):
            return self._hmin_unc
        elif hasattr(self, "aux_data") and "hmax_unc" in self.aux_data:
            return self.aux_data["hmax_unc"]
        else:
            self.hor_ver_uncertainties()
            return self._hmin_unc

    @property
    def vmax_unc(self):
        if hasattr(self, "_vmax_unc"):
            return self._vmax_unc
        else:
            self.hor_ver_uncertainties()
            return self._vmax_unc

    @property
    def az_hmax_unc(self):
        if hasattr(self, "_az_hmax_unc"):
            return self._az_hmax_unc
        else:
            self.hor_ver_uncertainties()
            return self._az_hmax_unc

    @property
    def az_hmin_unc(self):
        if hasattr(self, "_az_hmin_unc"):
            return self._az_hmin_unc
        else:
            self.hor_ver_uncertainties()
            return self._az_hmin_unc

    @property
    def source_receiver_dist(self):
        if hasattr(self, "_source_receiver_dist"):
            return self._source_receiver_dist
        else:
            print(
                "You need to set source_receiver_dist before."
                " Call self.set_source_receiver_dist(network)"
            )
            return

    @property
    def sr(self):
        return self.sampling_rate

    def get_np_array(self, stations, components=None, priority="HH", verbose=True):
        if not hasattr(self, "traces"):
            print("You should call read_waveforms first.")
            return None
        if components is None:
            components = self.components
        return utils.get_np_array(
            self.traces,
            stations,
            components=components,
            priority=priority,
            component_aliases=self.component_aliases,
            n_samples=self.n_samples,
            verbose=verbose,
        )

    def hor_ver_uncertainties(self, mode="intersection"):
        """Compute the horizontal and vertical uncertainties on location.

        Return errors as given by the 68% confidence ellipsoid.

        Parameters
        ---------------
        mode: string, default to 'intersection'
            Either 'intersection' or 'projection'. If `mode` is 'intersection', the
            horizontal uncertainties are the lengths of the semi-axes of the ellipse
            defined by the intersection between the confidence ellipsoid and the
            horizontal plane. This is consistent with the horizontal errors returned
            by NLLoc. If mode is 'projection', the horizontal uncertainties are the
            max and min span of the confidence ellipsoid in the horizontal
            directions.

        New Attributes
        ----------------
        hmax_unc: scalar, float
            The maximum horizontal uncertainty, in km.
        hmin_unc: scalar, float
            The minimum horizontal uncertainty, in km.
        vmax_unc: scalar, float
            The maximum vertical uncertainty, in km.
        az_hmax_unc: scalar, float
            The azimuth (angle from north) of the maximum horizontal
            uncertainty, in degrees.
        az_hmin_unc: scalar, float
            The azimuth (angle from north) of the minimum horizontal
            uncertainty, in degrees.

        Note: hmax + vmax does not have to be equal to the
        max_loc, the latter simply being the length of the
        longest semi-axis of the uncertainty ellipsoid.
        """
        if not hasattr(self, "cov_mat"):
            print("Class instance does not have a `cov_mat` attribute.")
            # these private attributes should be called via their property names
            self._hmax_unc = 15.0
            self._hmin_unc = 15.0
            self._vmax_unc = 15.0
            self._az_hmax_unc = 0.0
            self._az_hmin_unc = 0.0
            return
        # X: west, Y: south, Z: downward
        s_68_3df = 3.52
        s_68_2df = 2.28
        # eigendecomposition of whole matrix
        w, v = np.linalg.eigh(self.cov_mat)
        semi_axis_length = np.sqrt(s_68_3df * w)
        # check the vertical components of all semi-axes:
        vertical_unc = np.abs(semi_axis_length * v[2, :])
        if mode == "intersection":
            # eigendecomposition of cov mat restricted to horizontal components
            wh, vh = np.linalg.eigh(self.cov_mat[:2, :2])
            semi_axis_length_h = np.sqrt(s_68_2df * wh)
            hmax_unc = np.max(semi_axis_length_h)
            hmin_unc = np.min(semi_axis_length_h)
            hmax_dir = vh[:, wh.argmax()]
            hmin_dir = vh[:, wh.argmin()]
            az_hmax = np.arctan2(-hmax_dir[0], -hmax_dir[1]) * 180.0 / np.pi
            az_hmin = np.arctan2(-hmin_dir[0], -hmin_dir[1]) * 180.0 / np.pi
        elif mode == "projection":
            # check the horizontal components of all semi-axes:
            horizontal_unc = np.sqrt(
                np.sum((semi_axis_length[np.newaxis, :] * v[:2, :]) ** 2, axis=0)
            )
            hmax_unc = np.max(horizontal_unc)
            hmin_unc = np.min(horizontal_unc)
            hmax_dir = v[:, horizontal_unc.argmax()]
            hmin_dir = v[:, horizontal_unc.argmin()]
            az_hmax = np.arctan2(-hmax_dir[0], -hmax_dir[1]) * 180.0 / np.pi
            az_hmin = np.arctan2(-hmin_dir[0], -hmin_dir[1]) * 180.0 / np.pi
        # these private attributes should be called via their property names
        self._hmax_unc = hmax_unc
        self._hmin_unc = hmin_unc
        self._vmax_unc = np.max(vertical_unc)
        self._az_hmax_unc = az_hmax
        self._az_hmin_unc = az_hmin

    def n_closest_stations(self, n, available_stations=None):
        """Adjust `self.stations` to the `n` closest stations.


        Find the `n` closest stations and modify `self.stations` accordingly.
        The instance's properties will also change accordingly.

        Parameters
        ----------------
        n: scalar int
            The `n` closest stations to fetch.
        available_stations: list of strings, default to None
            The list of stations from which we search the closest stations.
            If some stations are known to lack data, the user
            may choose to not include these in the closest stations.
        """
        if not hasattr(self, "network_stations"):
            # typically, an Event instance has no network_stations
            # attribute, but a Template instance does
            self.network_stations = self.stations.copy()
        # re-initialize the stations attribute
        self.stations = np.array(self.network_stations, copy=True)
        index_pool = np.arange(len(self.network_stations))
        # limit the index pool to available stations
        if available_stations is not None:
            availability = np.in1d(
                self.network_stations, np.asarray(available_stations).astype("U")
            )
            valid = availability & self.availability
        else:
            valid = self.availability
        station_pool = self.network_stations[valid]
        closest_stations = (
            self.source_receiver_dist.loc[station_pool].sort_values().index[:n]
        )
        # make sure we return a n-vector
        if len(closest_stations) < n:
            missing = n - len(closest_stations)
            closest_stations = np.hstack(
                (
                    closest_stations,
                    self.source_receiver_dist.drop(closest_stations, axis="rows")
                    .sort_values()
                    .index[:missing],
                )
            )
        self.stations = np.asarray(closest_stations).astype("U")

    def pick_PS_phases_EQTransformer(
       self,
       duration,
       threshold_P=0.60,
       threshold_S=0.60,
       offset_ot=cfg.BUFFER_EXTRACTED_EVENTS_SEC,
       phase_on_comp={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"},
       component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
       **kwargs,
    ):
       """Use PhaseNet (Zhu et al., 2019) to pick P and S waves (Event class).

       Note1: PhaseNet must be used with 3-comp data.
       Note2: Extra kwargs are passed to
       `phasenet.wrapper.automatic_detection`.

       Parameters
       -----------
       duration: scalar float
           Duration, in seconds, of the time window to process to search for P
           and S wave arrivals.
       tag: string
           Tag name of the target data. For example: 'preprocessed_1_12'.
       threshold_P: scalar float, default to 0.60
           Threshold on PhaseNet's probabilities to trigger the identification
           of a P-wave arrival.
       threshold_S: scalar float, default to 0.60
           Threshold on PhaseNet's probabilities to trigger the identification
           of a S-wave arrival.
       mini_batch_size: scalar int, default to 126
           Number of traces processed in a single batch by PhaseNet. This
           shouldn't have to be tuned.
       phase_on_comp: dictionary, optional
           Dictionary defining which seismic phase is extracted on each
           component. For example, phase_on_comp['N'] gives the phase that is
           extracted on the north component.
       component_aliases: Dictionary
           Each entry of the dictionary is a list of strings.
           `component_aliases[comp]` is the list of all aliases used for
           the same component 'comp'. For example, `component_aliases['N'] =
           ['N', '1']` means that both the 'N' and '1' channels will be mapped
           to the Event's 'N' channel.
       """
       import seisbench.models as sbm

       # load model
       model = sbm.EQTransformer.from_pretrained("original")

       if kwargs.get("read_waveforms", True):
           # read waveforms in picking mode, i.e. with `time_shifted`=False
           self.read_waveforms(
               duration,
               offset_ot=offset_ot,
               phase_on_comp=phase_on_comp,
               component_aliases=component_aliases,
               time_shifted=False,
               **kwargs,
           )
       ML_picks = model.classify(
           self.traces,
           P_threshold=threshold_P,
           S_threshold=threshold_S,
       )
       # add picks to auxiliary data
       # self.set_aux_data(PhaseNet_picks)
       # format picks in pandas DataFrame
       pandas_picks = pd.DataFrame(
           index=self.stations,
           columns=[
               "P_picks_sec",
               "P_probas",
               "P_abs_picks",
               "S_picks_sec",
               "S_probas",
               "S_abs_picks",
           ],
       )
       for pick in ML_picks[0]:
           sta = pick.trace_id.split(".")[1]
           if (
               ~pd.isna(pandas_picks.loc[sta, f"{pick.phase}_probas"])
               and pick.peak_value > pandas_picks.loc[sta, f"{pick.phase}_probas"]
           ):
               pandas_picks.loc[sta, f"{pick.phase}_probas"] = pick.peak_value
               pandas_picks.loc[sta, f"{pick.phase}_picks_sec"] = (
                   pick.peak_time - self.origin_time
               )
               pandas_picks.loc[sta, f"{pick.phase}_abs_picks"] = pick.peak_time
           else:
               pandas_picks.loc[sta, f"{pick.phase}_probas"] = pick.peak_value
               pandas_picks.loc[sta, f"{pick.phase}_picks_sec"] = (
                   pick.peak_time - self.origin_time
               )
               pandas_picks.loc[sta, f"{pick.phase}_abs_picks"] = pick.peak_time
       self.picks = pandas_picks

    def pick_PS_phases(
        self,
        duration,
        threshold_P=0.60,
        threshold_S=0.60,
        offset_ot=cfg.BUFFER_EXTRACTED_EVENTS_SEC,
        mini_batch_size=126,
        phase_on_comp={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"},
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
        upsampling=1,
        downsampling=1,
        use_apriori_picks=False,
        search_win_sec=2.,
        **kwargs,
    ):
        """Use PhaseNet (Zhu et al., 2019) to pick P and S waves (Event class).

        Note1: PhaseNet must be used with 3-comp data.
        Note2: Extra kwargs are passed to
        `phasenet.wrapper.automatic_detection`.

        Parameters
        -----------
        duration: scalar float
            Duration, in seconds, of the time window to process to search for P
            and S wave arrivals.
        tag: string
            Tag name of the target data. For example: 'preprocessed_1_12'.
        threshold_P: scalar float, default to 0.60
            Threshold on PhaseNet's probabilities to trigger the identification
            of a P-wave arrival.
        threshold_S: scalar float, default to 0.60
            Threshold on PhaseNet's probabilities to trigger the identification
            of a S-wave arrival.
        mini_batch_size: scalar int, default to 126
            Number of traces processed in a single batch by PhaseNet. This
            shouldn't have to be tuned.
        phase_on_comp: dictionary, optional
            Dictionary defining which seismic phase is extracted on each
            component. For example, phase_on_comp['N'] gives the phase that is
            extracted on the north component.
        component_aliases: Dictionary
            Each entry of the dictionary is a list of strings.
            `component_aliases[comp]` is the list of all aliases used for
            the same component 'comp'. For example, `component_aliases['N'] =
            ['N', '1']` means that both the 'N' and '1' channels will be mapped
            to the Event's 'N' channel.
        upsampling: scalar integer, default to 1
            Upsampling factor applied before calling PhaseNet.
        downsampling: scalar integer, default to 1
            Downsampling factor applied before calling PhaseNet.
        """
        from phasenet import wrapper as PN

        if kwargs.get("read_waveforms", True):
            # read waveforms in picking mode, i.e. with `time_shifted`=False
            self.read_waveforms(
                duration,
                offset_ot=offset_ot,
                phase_on_comp=phase_on_comp,
                component_aliases=component_aliases,
                time_shifted=False,
                **kwargs,
            )
        data_arr = self.get_np_array(self.stations, components=["N", "E", "Z"])
        if upsampling > 1 or downsampling > 1:
            from scipy.signal import resample_poly
            data_arr = resample_poly(
                    data_arr,
                    upsampling,
                    downsampling,
                    axis=-1
                    )
            # momentarily update samping_rate
            sampling_rate0 = float(self.sampling_rate)
            self.sampling_rate = self.sr * upsampling / downsampling
        # call PhaseNet
        PhaseNet_probas, PhaseNet_picks = PN.automatic_picking(
            data_arr[np.newaxis, ...],
            self.stations,
            mini_batch_size=mini_batch_size,
            format="ram",
            threshold_P=threshold_P,
            threshold_S=threshold_S,
            **kwargs,
        )
        if use_apriori_picks and hasattr(self, "arrival_times"):
            columns = []
            if "P" in self.phases:
                columns.append("P")
            if "S" in self.phases:
                columns.append("S")
            prior_knowledge = pd.DataFrame(columns=columns)
            for sta in self.stations:
                for ph in prior_knowledge.columns:
                    prior_knowledge.loc[sta, ph] = utils.sec_to_samp(
                            udt(self.arrival_times.loc[
                                sta, f"{ph}_abs_arrival_times"
                                ])
                            -
                            self.origin_time,
                            sr=self.sampling_rate
                            )
        else:
            prior_knowledge = None
        # only used if use_apriori_picks is True
        search_win_samp = utils.sec_to_samp(
                search_win_sec, sr=self.sampling_rate
                )
        # keep best P- and S-wave pick on each 3-comp seismogram
        PhaseNet_picks = PN.get_picks(
                PhaseNet_picks,
                prior_knowledge=prior_knowledge,
                search_win_samp=search_win_samp
                )
        # add picks to auxiliary data
        # self.set_aux_data(PhaseNet_picks)
        # format picks in pandas DataFrame
        pandas_picks = {"stations": self.stations}
        for ph in ["P", "S"]:
            rel_picks_sec = np.zeros(len(self.stations), dtype=np.float32)
            proba_picks = np.zeros(len(self.stations), dtype=np.float32)
            abs_picks = np.zeros(len(self.stations), dtype=object)
            for s, sta in enumerate(self.stations):
                if sta in PhaseNet_picks[f"{ph}_picks"].keys():
                    rel_picks_sec[s] = PhaseNet_picks[f"{ph}_picks"][sta][0] / self.sr
                    proba_picks[s] = PhaseNet_picks[f"{ph}_proba"][sta][0]
                    if proba_picks[s] > 0.0:
                        abs_picks[s] = (
                            self.traces.select(station=sta)[0].stats.starttime
                            + rel_picks_sec[s]
                        )
            pandas_picks[f"{ph}_picks_sec"] = rel_picks_sec
            pandas_picks[f"{ph}_probas"] = proba_picks
            pandas_picks[f"{ph}_abs_picks"] = abs_picks
        self.picks = pd.DataFrame(pandas_picks)
        self.picks.set_index("stations", inplace=True)
        self.picks.replace(0.0, np.nan, inplace=True)
        if upsampling > 1 or downsampling > 1:
            # reset the sampling rate to initial value
            self.sampling_rate = sampling_rate0

    def read_waveforms(
        self,
        duration,
        phase_on_comp={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"},
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
        offset_phase={"P": 1.0, "S": 4.0},
        time_shifted=True,
        offset_ot=cfg.BUFFER_EXTRACTED_EVENTS_SEC,
        data_reader=None,
        **reader_kwargs,
    ):
        """Read waveform data (Event class).

        Parameters
        -----------
        duration: scalar float
            Duration, in seconds, of the extracted time windows.
        phase_on_comp: dictionary, optional
            Dictionary defining which seismic phase is extracted on each
            component. For example, phase_on_comp['N'] gives the phase that is
            extracted on the north component.
        component_aliases: Dictionary
            Each entry of the dictionary is a list of strings.
            `component_aliases[comp]` is the list of all aliases used for
            the same component 'comp'. For example, `component_aliases['N'] =
            ['N', '1']` means that both the 'N' and '1' channels will be mapped
            to the Event's 'N' channel.
        offset_phase: dictionary, optional
            Dictionary defining when the time window starts with respect to the
            pick. A positive offset means the window starts before the pick. Not
            used if `time_shifted` is False.
        time_shifted: boolean, default to True
            If True, the moveouts are used to extract time windows from specific
            seismic phases. If False, windows are simply extracted with respect to
            the origin time.
        offset_ot: scalar float, default to `cfg.BUFFER_EXTRACTED_EVENTS_SEC`
            Only used if `time_shifted` is False. Time, in seconds, taken before
            `origin_time`.
        data_reader: function, default to None
            Function that takes a path and optional key-word arguments to read
            data from this path and returns an `obspy.Stream` instance. If None,
            use `self.data_reader` and return None if `self.data_reader=None`.
        """
        # from pyasdf import ASDFDataSet
        from obspy import Stream

        if data_reader is None:
            data_reader = self.data_reader
        if data_reader is None:
            print("You need to specify a data reader for the class instance.")
            return
        self.traces = Stream()
        self.duration = duration
        self.n_samples = utils.sec_to_samp(self.duration, sr=self.sr)
        for sta in self.stations:
            for comp in self.components:
                ph = phase_on_comp[comp]
                if time_shifted:
                    pick = (
                        self.origin_time
                        + self.moveouts[f"moveouts_{ph.upper()}"].loc[sta]
                        - offset_phase[ph.upper()]
                    )
                else:
                    pick = self.origin_time - offset_ot
                for cp_alias in component_aliases[comp]:
                    self.traces += data_reader(
                        self.where,
                        station=sta,
                        channel=cp_alias,
                        starttime=pick,
                        endtime=pick + duration,
                        **reader_kwargs,
                    )
        for ph in offset_phase.keys():
            self.set_aux_data({f"offset_{ph.upper()}": offset_phase[ph]})
        for comp in phase_on_comp.keys():
            self.set_aux_data({f"phase_on_comp{comp}": phase_on_comp[comp]})
        if not time_shifted:
            self.trim_waveforms(
                starttime=self.origin_time - offset_ot,
                endtime=self.origin_time - offset_ot + self.duration,
            )

    def relocate(self, *args, routine="NLLoc", **kwargs):
        """Wrapper function for relocation with multiple methods.

        This single function interfaces the earthquake relocation with
        multiple relocation routines. All key-word arguments go the
        routine corresponding to `routine`.

        Parameters
        ----------
        routine: string, default to 'NLLoc'
            Method used for relocation. 'NLLoc' calls `relocated_NLLoc` and
            requires `self` to have the attribute `picks`.
        """
        if routine.lower() == "nlloc":
            self.relocate_NLLoc(**kwargs)
        elif routine.lower() == "beam":
            self.relocate_beam(*args, **kwargs)

    def relocate_beam(
        self,
        beamformer,
        duration=60.0,
        offset_ot=cfg.BUFFER_EXTRACTED_EVENTS_SEC,
        phase_on_comp={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"},
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
        waveform_features=None,
        device="cpu",
        **kwargs,
    ):
        """ """
        from .template_search import Beamformer, envelope

        if kwargs.get("read_waveforms", True):
            # read waveforms in picking mode, i.e. with `time_shifted`=False
            self.read_waveforms(
                duration,
                offset_ot=offset_ot,
                phase_on_comp=phase_on_comp,
                component_aliases=component_aliases,
                time_shifted=False,
                **kwargs,
            )
        if waveform_features is None:
            data_arr = self.get_np_array(
                beamformer.network.stations, components=["N", "E", "Z"]
            )
            norm = np.std(data_arr, axis=(1, 2), keepdims=True)
            norm[norm == 0.] = 1.
            data_arr /= norm
            waveform_features = envelope(data_arr)
        #print(waveform_features)
        beamformer.backproject(waveform_features, device=device, reduce="none")
        # find where the maximum focusing occurred
        src_idx, time_idx = np.unravel_index(
                beamformer.beam.argmax(),
                beamformer.beam.shape
                )
        # update hypocenter
        self.origin_time = (
            self.traces[0].stats.starttime + time_idx / self.sampling_rate
        )
        self.longitude = beamformer.source_coordinates["longitude"].iloc[src_idx]
        self.latitude = beamformer.source_coordinates["latitude"].iloc[src_idx]
        self.depth = beamformer.source_coordinates["depth"].iloc[src_idx]
        # estimate location uncertainty
        # fill arrival time attribute
        self.arrival_times = pd.DataFrame(
            index=beamformer.network.stations,
            columns=[
                "P_tt_sec",
                "P_abs_arrival_times",
                "S_tt_sec",
                "S_abs_arrival_times",
            ],
        )
        travel_times = beamformer.moveouts[src_idx, ...]
        beamformer.phases = [ph.upper() for ph in beamformer.phases]
        for s, sta in enumerate(beamformer.network.stations):
            for p, ph in enumerate(["P", "S"]):
                pp = beamformer.phases.index(ph)
                self.arrival_times.loc[sta, f"{ph}_tt_sec"] = (
                    travel_times[s, pp] / self.sampling_rate
                )
                self.arrival_times.loc[sta, f"{ph}_abs_arrival_times"] = (
                    self.origin_time + self.arrival_times.loc[sta, f"{ph}_tt_sec"]
                )

    def relocate_NLLoc(self, stations=None, method="EDT", verbose=0, **kwargs):
        """Relocate with NLLoc using `self.picks`.

        Parameters
        -----------
        stations: list of strings, default to None
            Names of the stations to include in the relocation process. If None,
            `stations` is set to `self.stations`.
        method: string, default to 'EDT'
            Optimization algorithm used by NonLinLoc. Either 'GAU_ANALYTIC',
            'EDT', 'EDT_OT', 'EDT_OT_WT_ML'. See NonLinLoc's documentation for
            more information.
        verbose: scalar int, default to 0
            If more than 0, print NLLoc's outputs to the standard output.
        """
        import subprocess
        import glob
        from . import NLLoc_utils

        if stations is None:
            stations = self.stations
        # create folder for input/output files
        input_dir = os.path.join(cfg.NLLOC_INPUT_PATH, self.id)
        output_dir = os.path.join(cfg.NLLOC_OUTPUT_PATH, self.id)
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # file names:
        ctrl_fn = os.path.join(self.id, self.id + ".in")
        out_basename = os.path.join(self.id, self.id + "_out")
        obs_fn = os.path.join(self.id, self.id + ".obs")
        # write obs file
        if os.path.isfile(os.path.join(cfg.NLLOC_INPUT_PATH, obs_fn)):
            os.remove(os.path.join(cfg.NLLOC_INPUT_PATH, obs_fn))
        NLLoc_utils.write_NLLoc_obs(self.origin_time, self.picks, stations, obs_fn)
        # write control file
        NLLoc_utils.write_NLLoc_control(
                ctrl_fn, out_basename, obs_fn, method=method, **kwargs
                )
        if verbose == 0:
            # run NLLoc
            subprocess.run(
                f"NLLoc {os.path.join(cfg.NLLOC_INPUT_PATH, ctrl_fn)} "
                f"> {os.devnull}",
                shell=True,
            )
        else:
            # run NLLoc
            subprocess.run(
                "NLLoc " + os.path.join(cfg.NLLOC_INPUT_PATH, ctrl_fn), shell=True
            )
        # read results
        try:
            out_fn = os.path.basename(
                glob.glob(
                    os.path.join(cfg.NLLOC_OUTPUT_PATH, out_basename + ".[!s]*hyp")
                )[0]
            )
        except IndexError:
            # relocation failed
            for fn in glob.glob(os.path.join(input_dir, "*")):
                pathlib.Path(fn).unlink()
            if os.path.isdir(input_dir):
                # add this protection against unexpected
                # external change
                pathlib.Path(input_dir).rmdir()
            for fn in glob.glob(os.path.join(output_dir, "*")):
                pathlib.Path(fn).unlink()
            if os.path.isdir(output_dir):
                # add this protection against unexpected
                # external change
                pathlib.Path(output_dir).rmdir()
            return
        hypocenter, predicted_times = NLLoc_utils.read_NLLoc_outputs(
            out_fn, os.path.join(cfg.NLLOC_OUTPUT_PATH, self.id)
        )
        if hypocenter is None:
            # problem when reading the output
            for fn in glob.glob(os.path.join(input_dir, "*")):
                pathlib.Path(fn).unlink()
            if os.path.isdir(input_dir):
                # add this protection against unexpected
                # external change
                pathlib.Path(input_dir).rmdir()
            for fn in glob.glob(os.path.join(output_dir, "*")):
                pathlib.Path(fn).unlink()
            if os.path.isdir(output_dir):
                # add this protection against unexpected
                # external change
                pathlib.Path(output_dir).rmdir()
            return
        hypocenter["origin_time"] = udt(hypocenter["origin_time"])
        # round seconds to reasonable precision to avoid producing
        # origin times that are in between samples
        hypocenter["origin_time"] = udt(
            utils.round_time(hypocenter["origin_time"].timestamp, sr=self.sr)
        )
        # update event's attributes
        for key in hypocenter.keys():
            setattr(self, key, hypocenter[key])
        # add absolute arrival times to predicted_times
        P_abs_arrivals = np.zeros(len(predicted_times), dtype=object)
        S_abs_arrivals = np.zeros(len(predicted_times), dtype=object)
        for s, sta in enumerate(predicted_times.index):
            P_abs_arrivals[s] = self.origin_time + predicted_times.loc[sta, "P_tt_sec"]
            S_abs_arrivals[s] = self.origin_time + predicted_times.loc[sta, "S_tt_sec"]
        predicted_times["P_abs_arrival_times"] = P_abs_arrivals
        predicted_times["S_abs_arrival_times"] = S_abs_arrivals
        # attach the theoretical arrival times
        self.arrival_times = predicted_times
        self.set_aux_data({"NLLoc_reloc": True})
        self.set_aux_data({"cov_mat": self.cov_mat, "tt_rms": self.tt_rms})
        # clean the temporary control and pick files
        for fn in glob.glob(os.path.join(input_dir, "*")):
            pathlib.Path(fn).unlink()
        if os.path.isdir(input_dir):
            # add this protection against unexpected
            # external change
            pathlib.Path(input_dir).rmdir()
        for fn in glob.glob(os.path.join(output_dir, "*")):
            pathlib.Path(fn).unlink()
        if os.path.isdir(output_dir):
            # add this protection against unexpected
            # external change
            pathlib.Path(output_dir).rmdir()

    def remove_outlier_picks(self, max_diff_percent=25.):
        """Remove picks that are too far from predicted arrival times.

        Parameters
        ----------
        max_diff_percent: float, default to 25
            Maximum difference, in percentage, between the picked and predicted
            arrival time.
        """
        stations_outlier = []
        for sta in self.stations:
            for ph in self.phases:
                pick = pd.Timestamp(str(self.picks.loc[sta, f"{ph}_abs_picks"]))
                predicted = pd.Timestamp(str(self.arrival_times.loc[sta, f"{ph}_abs_arrival_times"]))
                predicted_tt = self.arrival_times.loc[sta, f"{ph}_tt_sec"]
                diff_percent = (
                        100. * abs((pick - predicted).total_seconds())/predicted_tt
                        )
                if diff_percent > max_diff_percent:
                    stations_outlier.append(sta)
                    self.picks.loc[sta, f"{ph}_abs_picks"] = np.nan
                    self.picks.loc[sta, f"{ph}_picks_sec"] = np.nan
                    self.picks.loc[sta, f"{ph}_probas"] = np.nan

    def zero_out_clipped_waveforms(self, kurtosis_threshold=-1.):
        """Find waveforms with anomalous statistic and zero them out.

        Parameters
        ----------
        kurtosis_threshold: scalar float, optional
            Threshold below which the kurtosis is considered anomalous.
            Note that the Fischer definition of the kurtosis is used,
            that is, kurtosis=0 for gaussian distribution.
        """
        from scipy.stats import kurtosis
        if not hasattr(self, "traces"):
            return
        for tr in self.traces:
            if kurtosis(tr.data) < kurtosis_threshold:
                tr.data = np.zeros(len(tr.data), dtype=tr.data.dtype)

    def remove_distant_stations(self, max_distance_km=50.):
        """Remove picks on stations that are further than given distance.

        Parameters
        ----------
        max_distance_km: float, default to 50
            Maximum distance, in km, beyond which picks are set to NaN.
        """
        if self.source_receiver_dist is None:
            print("Call self.set_source_receiver_dist(network) before "
                  "using self.remove_distant_stations.")
            return
        for sta in self.stations:
            if self.source_receiver_dist.loc[sta] > max_distance_km:
                self.picks.loc[sta] = np.nan

    def set_aux_data(self, aux_data):
        """Adds any extra data to the Event instance.

        Parameters
        ------------
        aux_data: dictionary
            Dictionary with any auxiliary data.
        """
        if not hasattr(self, "aux_data"):
            self.aux_data = {}
        for field in aux_data:
            self.aux_data[field] = aux_data[field]

    def set_availability(
        self,
        stations=None,
        components=["N", "E", "Z"],
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
    ):
        """Set the data availability.

        A station is available if at least one station has non-zero data. The
        availability is then accessed via the property `self.availability`.

        Parameters
        -----------
        stations: list of strings or numpy.ndarray, default to None
            Names of the stations on which we check availability. If None, use
            `self.stations`.
        """
        if stations is None:
            stations = self.stations
        if not hasattr(self, "traces"):
            print("Call `self.read_waveforms` first.")
            return
        self._availability_per_sta = pd.Series(
            index=stations,
            data=np.zeros(len(stations), dtype=bool),
        )
        self._availability_per_cha = pd.DataFrame(index=stations)
        for c, cp in enumerate(components):
            availability = np.zeros(len(stations), dtype=bool)
            for s, sta in enumerate(stations):
                for cp_alias in component_aliases[cp]:
                    tr = self.traces.select(station=sta, component=cp_alias)
                    if len(tr) > 0:
                        tr = tr[0]
                    else:
                        continue
                    if np.sum(tr.data != 0.0):
                        availability[s] = True
                        break
            self._availability_per_cha[cp] = availability
            self._availability_per_sta = np.bitwise_or(
                self._availability_per_sta, availability
            )
        self.set_aux_data(
            {
                "availability": self._availability_per_sta,
                "availability_per_sta": self._availability_per_sta,
            }
        )
        self.set_aux_data(
            {f"availability_{cp}": self._availability_per_cha[cp] for cp in components}
        )

    def set_components(self, components):
        """Set the list of components.

        Parameters
        -----------
        components: list of strings
            The names of the components on which the `Template` instance will
            work after this call to `self.set_components`.
        """
        self.components = components

    def set_component_aliases(self, component_aliases):
        """Set or modify the `component_aliases` attribute.

        Parameters
        -----------
        component_aliases: Dictionary
            Each entry of the dictionary is a list of strings.
            `component_aliases[comp]` is the list of all aliases used for
            the same component 'comp'. For example, `component_aliases['N'] =
            ['N', '1']` means that both the 'N' and '1' channels will be mapped
            to the Event's 'N' channel.
        """
        self.component_aliases = component_aliases

    def set_arrival_times_from_moveouts(self, verbose=1):
        """Build arrival times assuming at = ot + mv."""
        if verbose > 0:
            print("Make sure origin_time + moveout points at the phase arrival!")
        self.arrival_times = pd.DataFrame(
                columns=[
                    f"{ph.upper()}_abs_arrival_times" for ph in self.phases
                    ]
                )
        for ph in self.phases:
            ph = ph.upper()
            field = f"{ph}_abs_arrival_times"
            for sta in self.moveouts.index:
                self.arrival_times.loc[sta, field] = (
                        self.origin_time + self.moveouts.loc[sta, f"moveouts_{ph}"]
                        )

    def set_moveouts_to_empirical_times(self):
        """Set moveouts equal to picks, if available."""
        if not hasattr(self, "picks"):
            print("Does not have a `picks` attribute.")
            return
        # make sure picks are consistent with the current origin time
        self.update_picks()
        self.origin_time = udt(utils.round_time(self.origin_time.timestamp, sr=self.sr))
        for station in self.picks.index:
            for ph in self.phases:
                if not pd.isnull(self.picks.loc[station, f"{ph.upper()}_picks_sec"]):
                    self.moveouts.loc[
                        station, f"moveouts_{ph.upper()}"
                    ] = utils.round_time(
                        self.picks.loc[station, f"{ph.upper()}_picks_sec"], sr=self.sr
                    )

    def set_moveouts_to_theoretical_times(self):
        """Set moveouts equal to theoretical arrival times, if available."""
        if not hasattr(self, "arrival_times"):
            print("Does not have a `arrival_times` attribute.")
            return
        # make sure travel times are consistent with the current origin time
        self.update_travel_times()
        self.origin_time = udt(utils.round_time(self.origin_time.timestamp, sr=self.sr))
        for station in self.arrival_times.index:
            for ph in self.phases:
                if not pd.isnull(
                    self.arrival_times.loc[station, f"{ph.upper()}_tt_sec"]
                ):
                    self.moveouts.loc[
                        station, f"moveouts_{ph.upper()}"
                    ] = utils.round_time(
                        self.arrival_times.loc[station, f"{ph.upper()}_tt_sec"],
                        sr=self.sr,
                    )

    def set_source_receiver_dist(self, network):
        """Set source-receiver distances, given `network`.

        Parameters
        -----------
        network: `dataset.Network` instance
            The `Network` instance with the station coordinates to use
            in the source-receiver computation.
        """
        distances = utils.compute_distances(
            [self.longitude],
            [self.latitude],
            [self.depth],
            network.longitude,
            network.latitude,
            network.depth,
        )
        self._source_receiver_dist = pd.Series(
            index=network.stations,
            data=distances.squeeze(),
            name="source-receiver dist (km)",
        )
        if not hasattr(self, "network_stations"):
            self.network_stations = self.stations.copy()
        self._source_receiver_dist = self.source_receiver_dist.loc[
            self.network_stations
        ]

    def trim_waveforms(self, starttime=None, endtime=None):
        """Trim waveforms.

        Start times might differ of one sample on different traces. Use this
        method to make sure all traces have the same start time.

        Parameters
        -----------
        starttime: string or datetime, default to None
            If None, use `self.date` as the start time.
        endtime: string or datetime, default to None
            If None, use `self.date` + `self.duration` as the end time.
        """
        if not hasattr(self, "traces"):
            print("You should call `read_waveforms` first.")
            return
        if starttime is None:
            starttime = self.date
        if endtime is None:
            endtime = self.date + self.duration
        for tr in self.traces:
            fill_value = np.zeros(1, dtype=tr.data.dtype)[0]
            tr.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=fill_value)

    def update_picks(self):
        """Update the picks w.r.t the current origin time."""
        if not hasattr(self, "picks"):
            print("Does not have a `picks` attribute.")
            return
        for station in self.picks.index:
            for ph in self.phases:
                if pd.isnull(self.picks.loc[station, f"{ph.upper()}_abs_picks"]):
                    continue
                self.picks.loc[station, f"{ph.upper()}_picks_sec"] = udt(
                    self.picks.loc[station, f"{ph.upper()}_abs_picks"]
                ) - udt(self.origin_time)

    def update_travel_times(self):
        """Update travel times w.r.t the current origin time."""
        if not hasattr(self, "arrival_times"):
            print("Does not have an `arrival_times` attribute.")
            return
        for station in self.arrival_times.index:
            for ph in self.phases:
                self.arrival_times.loc[station, f"{ph.upper()}_tt_sec"] = udt(
                    self.arrival_times.loc[station, f"{ph.upper()}_abs_arrival_times"]
                ) - udt(self.origin_time)

    def write(
        self,
        db_filename,
        db_path=cfg.OUTPUT_PATH,
        save_waveforms=False,
        gid=None,
        hdf5_file=None,
    ):
        """Write to hdf5 file.

        Parameters
        ------------
        db_filename: string
            Name of the hdf5 file storing the event information.
        db_path: string, default to `cfg.OUTPUT_PATH`
            Name of the directory with `db_filename`.
        save_waveforms: boolean, default to False
            If True, save the waveforms.
        gid: string, default to None
            Name of the hdf5 group where Event will be stored. If `gid=None`
            then Event is directly stored at the root.
        hdf5_file: `h5py.File`, default to None
            If not None, is an opened file pointing directly at the subfolder of
            interest.
        """
        output_where = os.path.join(db_path, db_filename)
        attributes = [
            "origin_time",
            "latitude",
            "longitude",
            "depth",
            "moveouts",
            "stations",
            "components",
            "phases",
            "where",
            "sampling_rate",
        ]
        # moveouts' indexes may have been re-ordered
        # because writing moveouts as an array will forget about the current
        # row indexes and assume that they are in the same order as
        # self.stations, it is critical to make sure this is true
        self.moveouts = self.moveouts.loc[self.stations]
        lock_file = output_where + ".lock"
        while os.path.isfile(lock_file):
            # another process is already writing in this file
            # wait a bit a check again
            sleep(1.0)
        # create empty lock file
        open(lock_file, "w").close()
        try:
            if hdf5_file is None:
                hdf5_file = h5.File(output_where, mode="a")
                close_file = True
            else:
                close_file = False
            if gid is not None:
                if gid in hdf5_file:
                    # overwrite existing detection with same id
                    print(
                        f"Found existing event {gid} in {output_where}. Overwrite it."
                    )
                    del hdf5_file[gid]
                hdf5_file.create_group(gid)
                f = hdf5_file[gid]
            else:
                f = hdf5_file
            # with h5.File(output_where, mode='a') as f:
            #    if gid is not None:
            #        if gid in f:
            #            # overwrite existing detection with same id
            #            print(f'Found existing event {gid} in {output_where}. Overwrite it.')
            #            del f[gid]
            #        f.create_group(gid)
            #        f = f[gid]
            for attr in attributes:
                if not hasattr(self, attr):
                    continue
                attr_ = getattr(self, attr)
                if attr == "origin_time":
                    attr_ = str(attr_)
                if isinstance(attr_, list):
                    attr_ = np.asarray(attr_)
                if isinstance(attr_, np.ndarray) and (
                    attr_.dtype.kind == np.dtype("U").kind
                ):
                    attr_ = attr_.astype("S")
                f.create_dataset(attr, data=attr_)
            if hasattr(self, "aux_data"):
                f.create_group("aux_data")
                for key in self.aux_data.keys():
                    f["aux_data"].create_dataset(key, data=self.aux_data[key])
            if hasattr(self, "picks"):
                f.create_group("picks")
                f["picks"].create_dataset(
                    "stations", data=np.asarray(self.picks.index).astype("S")
                )
                for column in self.picks.columns:
                    data = self.picks[column]
                    if data.dtype.kind == "M":
                        # pandas datetime format
                        data = data.dt.strftime("%Y-%m-%d %H:%M:%S.%f %z")
                    if data.dtype == np.dtype("O"):
                        data = data.astype("S")
                    f["picks"].create_dataset(column, data=data)
            if hasattr(self, "arrival_times"):
                f.create_group("arrival_times")
                f["arrival_times"].create_dataset(
                    "stations", data=np.asarray(self.arrival_times.index).astype("S")
                )
                for column in self.arrival_times.columns:
                    data = self.arrival_times[column]
                    if data.dtype.kind == "M":
                        # pandas datetime format
                        data = data.dt.strftime("%Y-%m-%d %H:%M:%S.%f %z")
                    if data.dtype == np.dtype("O"):
                        data = data.astype("S")
                    f["arrival_times"].create_dataset(column, data=data)
            if save_waveforms:
                if hasattr(self, "traces"):
                    f.create_group("waveforms")
                    for tr in self.traces:
                        sta = tr.stats.station
                        cha = tr.stats.channel
                        if sta not in f["waveforms"]:
                            f["waveforms"].create_group(sta)
                        if cha in f["waveforms"][sta]:
                            print(f"{sta}.{cha} already exists!")
                        else:
                            f["waveforms"][sta].create_dataset(cha, data=tr.data)
                else:
                    print(
                        "You are trying to save the waveforms whereas you did"
                        " not read them!"
                    )
            if close_file:
                hdf5_file.close()
        except Exception as e:
            os.remove(lock_file)
            raise (e)
        # remove lock file
        os.remove(lock_file)

    # -----------------------------------------------------------
    #            plotting method(s)
    # -----------------------------------------------------------

    def plot(
        self, figsize=(20, 15), gain=1.0e6, stations=None, ylabel=r"Velocity ($\mu$m/s)", **kwargs
    ):
        """Plot the waveforms of the Event instance.

        Parameters
        ------------

        Returns
        ----------
        fig: plt.Figure
            Figure instance produced by this method.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if stations is None:
            stations = self.stations
        start_times, end_times = [], []
        fig, axes = plt.subplots(
            num=f"event_{str(self.origin_time)}",
            figsize=figsize,
            nrows=len(stations),
            ncols=len(self.components),
        )
        fig.suptitle(f'Event at {self.origin_time.strftime("%Y-%m-%d %H:%M:%S")}')
        for s, sta in enumerate(stations):
            for c, cp in enumerate(self.components):
                for cp_alias in self.component_aliases[cp]:
                    tr = self.traces.select(station=sta, component=cp_alias)
                    if len(tr) > 0:
                        # succesfully retrieved data
                        break
                if len(tr) == 0:
                    continue
                else:
                    tr = tr[0]
                time = utils.time_range(
                    tr.stats.starttime,
                    tr.stats.endtime + tr.stats.delta,
                    tr.stats.delta,
                    unit="ms",
                )
                start_times.append(time[0])
                end_times.append(time[-1])
                axes[s, c].plot(
                    time[: self.n_samples], tr.data[: self.n_samples] * gain, color="k"
                )
                # plot the picks
                if hasattr(self, "picks") and (
                    sta in self.picks["P_abs_picks"].dropna().index
                ):
                    P_pick = np.datetime64(self.picks.loc[sta]["P_abs_picks"])
                    axes[s, c].axvline(P_pick, color="C0", lw=1.00, ls="--")
                if hasattr(self, "picks") and (
                    sta in self.picks["S_abs_picks"].dropna().index
                ):
                    S_pick = np.datetime64(self.picks.loc[sta]["S_abs_picks"])
                    axes[s, c].axvline(S_pick, color="C3", lw=1.00, ls="--")
                # plot the theoretical arrival times
                if hasattr(self, "arrival_times") and (sta in self.arrival_times.index):
                    P_pick = np.datetime64(
                        self.arrival_times.loc[sta]["P_abs_arrival_times"]
                    )
                    axes[s, c].axvline(P_pick, color="C4", lw=1.25)
                if hasattr(self, "arrival_times") and (sta in self.arrival_times.index):
                    S_pick = np.datetime64(
                        self.arrival_times.loc[sta]["S_abs_arrival_times"]
                    )
                    axes[s, c].axvline(S_pick, color="C1", lw=1.25)
                axes[s, c].text(
                    0.05, 0.05, f"{sta}.{cp_alias}", transform=axes[s, c].transAxes
                )
        for ax in axes.flatten():
            ax.set_xlim(min(start_times), max(end_times))
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
            )
        plt.subplots_adjust(top=0.95, bottom=0.06, right=0.98, left=0.06)
        fig.text(0.03, 0.40, ylabel, rotation="vertical")
        return fig


class Template(Event):
    """A class for template events."""

    def __init__(
        self,
        origin_time,
        moveouts,
        stations,
        phases,
        template_filename,
        template_path,
        latitude=None,
        longitude=None,
        depth=None,
        sampling_rate=None,
        components=["N", "E", "Z"],
        id=None,
    ):
        super().__init__(
            origin_time,
            moveouts,
            stations,
            phases,
            template_filename,
            template_path,
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            sampling_rate=sampling_rate,
            components=components,
            id=id,
        )
        # network_stations is a copy of the original list of stations
        # self.stations may change if the user wants to use the template
        # on a subset of stations
        self.network_stations = np.array(stations, copy=True)

    @classmethod
    def init_from_event(cls, event, attach_waveforms=True):
        """Instanciate a `Template` object from an `Event` object.

        Parameters
        -----------
        event: `Event` instance
            The `Event` instance to convert to a `Template` instance.
        attach_waveforms: boolean, default to True
            Should not be turned to False when used directly.

        Returns
        ---------
        template: Template instance
            `Template` instance base on `event`.
        """
        db_path, db_filename = os.path.split(event.where)
        template = cls(
            event.origin_time,
            event.moveouts.values,
            event.stations,
            event.phases,
            db_filename,
            db_path,
            latitude=event.latitude,
            longitude=event.longitude,
            depth=event.depth,
            sampling_rate=event.sampling_rate,
            components=event.components,
            id=event.id,
        )
        if hasattr(event, "picks"):
            template.picks = event.picks
        if hasattr(event, "arrival_times"):
            template.arrival_times = event.arrival_times
        aux_data_to_keep = [f"offset_{ph}" for ph in event.phases] + [
            "tt_rms",
            "tid",
            "cov_mat",
        ]
        select = lambda str: str.startswith("phase_on_comp")
        aux_data_to_keep += list(filter(select, event.aux_data.keys()))
        aux_data = {
            key: event.aux_data[key]
            for key in aux_data_to_keep
            if key in event.aux_data
        }
        if attach_waveforms:
            # ----------------------------------
            # attach traces
            if not hasattr(event, "traces"):
                print(
                    "You need to call `event.read_waveforms` before instanciating"
                    " a Template object with this Event instance."
                )
                return
            template.traces = event.traces
            template.n_samples = event.n_samples
            template.set_availability()
        else:
            template.n_samples = event.aux_data["n_samples"]
        # ----------------------------------
        aux_data["n_samples"] = template.n_samples
        if "cov_mat" in aux_data:
            template.cov_mat = aux_data["cov_mat"]
        template.set_aux_data(aux_data)
        return template

    @classmethod
    def read_from_file(cls, filename, db_path=cfg.INPUT_PATH, gid=None):
        """Initialize a `Template` instance from a file."""
        template = cls.init_from_event(
            Event.read_from_file(filename, db_path=db_path, gid=gid),
            attach_waveforms=False,
        )
        template.n_samples = template.aux_data["n_samples"]
        template.id = template.aux_data["tid"]
        # overwrite any path that was stored in aux_data, because what matters
        # for the template is only the file it was associated with
        template.where = os.path.join(db_path, filename)
        return template

    # properties
    @property
    def template_idx(self):
        return self.id

    @property
    def tid(self):
        return self.id

    @property
    def moveouts_arr(self):
        """Return a moveout array given self.components and phase_on_comp."""
        return utils.sec_to_samp(
            self.moveouts_win.loc[self.stations][
                [
                    f'moveouts_{self.aux_data[f"phase_on_comp{cp}"].upper()}'
                    for cp in self.components
                ]
            ].values,
            sr=self.sr,
        )

    @property
    def moveouts_win(self):
        if not hasattr(self, "_moveouts_win"):
            # this new moveout table store the moveouts of the
            # time windows rather than the phases
            self._moveouts_win = self.moveouts.copy()
            for ph in self.phases:
                self._moveouts_win[f"moveouts_{ph.upper()}"] = (
                    self.moveouts[f"moveouts_{ph.upper()}"]
                    - self.aux_data[f"offset_{ph.upper()}"]
                )
        return self._moveouts_win

    @property
    def waveforms_arr(self):
        """Return traces in numpy.ndarray."""
        return utils.get_np_array(
            self.traces,
            self.stations,
            components=self.components,
            priority="HH",
            component_aliases=self.component_aliases,
            n_samples=self.n_samples,
            verbose=True,
        )

    # methods
    def distance(self, longitude, latitude, depth):
        """Compute distance between template and a given location.

        Parameters
        -----------
        longitude: scalar, float
            Longitude of the target location.
        latitude: scalar, float
            Latitude of the target location.
        depth: scalar, float
            Depth of the target location, in km.
        """
        from .utils import two_point_distance

        return two_point_distance(
            self.longitude, self.latitude, self.depth, longitude, latitude, depth
        )

    def n_best_SNR_stations(self, n, available_stations=None):
        """Adjust `self.stations` to the `n` best SNR stations.


        Find the `n` best stations and modify `self.stations` accordingly.
        The instance's properties will also change accordingly.

        Parameters
        ----------------
        n: scalar int
            The `n` closest stations.
        available_stations: list of strings, default to None
            The list of stations from which we search the closest stations.
            If some stations are known to lack data, the user
            may choose to not include these in the closest stations.
        """
        # re-initialize the stations attribute
        self.stations = np.array(self.network_stations, copy=True)
        index_pool = np.arange(len(self.network_stations))
        # limit the index pool to available stations
        if available_stations is not None:
            availability = np.in1d(
                self.network_stations, available_stations.astype("U")
            )
            valid = availability & self.availability
        else:
            valid = self.availability
        index_pool = index_pool[valid]
        best_SNR_stations = index_pool[np.argsort(self.SNR[index_pool])[::-1]]
        # make sure we return a n-vector
        if len(best_SNR_stations) < n:
            missing = n - len(best_SNR_stations)
            remaining_indexes = np.setdiff1d(
                np.argsort(self.SNR)[::-1], best_SNR_stations
            )
            best_SNR_stations = np.hstack(
                (best_SNR_stations, remaining_indexes[:missing])
            )
        self.stations = self.network_stations[best_SNR_stations[:n]]

    def read_waveforms(self, stations=None, components=None):
        """Read the waveforms time series."""
        if stations is None:
            stations = self.stations
        if components is None:
            components = self.components

        def find_channel(keys, cp):
            for cp_alias in self.component_aliases[cp]:
                channel = list(filter(lambda x: x.endswith(cp_alias), keys))
                if len(channel) > 0:
                    break
            if len(channel) > 0:
                return channel[0]
            else:
                return

        self.traces = obs.Stream()
        with h5.File(self.where, mode="r") as f:
            if hasattr(self, "hdf5_gid"):
                f = f[self.hdf5_gid]
            for sta in stations:
                if sta not in f["waveforms"]:
                    # station not available
                    continue
                for cp in components:
                    channel = find_channel(f["waveforms"][sta].keys(), cp)
                    if channel is None:
                        continue
                    tr = obs.Trace()
                    tr.data = f["waveforms"][sta][channel][()]
                    tr.stats.station = sta
                    tr.stats.channel = channel
                    tr.stats.sampling_rate = self.sampling_rate
                    ph = self.aux_data[f"phase_on_comp{cp}"].upper()
                    mv = self.moveouts_win.loc[sta, f"moveouts_{ph}"]
                    tr.stats.starttime = self.origin_time + mv
                    self.traces += tr
        self.set_availability(stations=stations)

    def write(
        self,
        db_filename,
        db_path=cfg.OUTPUT_PATH,
        save_waveforms=True,
        gid=None,
        overwrite=False,
    ):
        self.where = os.path.join(db_path, db_filename)
        if overwrite and os.path.isfile(self.where):
            os.remove(self.where)
        Event.write(
            self, db_filename, db_path=db_path, save_waveforms=save_waveforms, gid=gid
        )

    # ---------------------------------------------
    #  methods to investigate the detected events
    def read_catalog(
        self,
        filename=None,
        db_path=None,
        gid=None,
        extra_attributes=[],
        fill_value=np.nan,
        return_events=False,
        check_summary_file=True,
    ):
        """Build a `Catalog` instance.

        Parameters
        ------------
        filename: string, default to None
            Name of the detection file. If None, use the standard file
            and folder naming convention.
        db_path: string, default to None
            Name of the directory where the detection file is located. If None,
            use the standard file and folder naming convention.
        gid: string, int, or float, default to None
            If not None, this is the hdf5 group where to read the data.
        extra_attributes: list of strings, default to []
            Attributes to read in addition to the default 'longitude',
            'latitude', 'depth', and 'origin_time'.
        fill_value: string, int, or float, default to np.nan
            Default value if the target attribute does not exist.
        return_events: boolean, default to False
            If True, return a list of `dataset.Event` instances. Can only be
            True if `check_summary_file=False`.
        check_summary_file: boolean, default to True
            If True, check if the summary hdf5 file already exists and read from
            if it does; this uses the standard naming convention. If False,
            it builds the catalog from the detection output.
        """
        db_path_T, filename_T = os.path.split(self.where)
        if return_events and check_summary_file:
            print(
                "If `return_events` is True, `check_summary_file` has"
                " to be False. Change arguments."
            )
            return
        if check_summary_file:
            if filename is None:
                # try standard names
                if os.path.isfile(
                    os.path.join(db_path_T, f"summary_template{self.tid}.h5")
                ):
                    # found an existing summary file
                    filename = f"summary_template{self.tid}.h5"
                    build_from_scratch = False
                else:
                    # no existing summary file
                    filename = f"detections_{filename_T}"
                    build_from_scratch = True
            elif os.path.isfile(os.path.join(db_path_T, filename)):
                # use provided file
                build_from_scratch = False
            else:
                build_from_scratch = True
        else:
            if filename is None:
                # try standard name
                filename = f"detections_{filename_T}"
            build_from_scratch = True
        if build_from_scratch:
            if db_path is None:
                # guess from standard convention
                db_path = db_path_T[::-1].replace(
                    "template"[::-1], "matched_filter"[::-1], 1
                )[::-1]
            output = Catalog.read_from_detection_file(
                filename,
                db_path=db_path,
                gid=gid,
                extra_attributes=extra_attributes,
                fill_value=fill_value,
                return_events=return_events,
            )
            if not return_events:
                self.catalog = output
            else:
                self.catalog, events = output
            self.catalog.catalog["tid"] = [self.tid] * len(self.catalog.catalog)
            self.catalog.catalog["event_id"] = [
                f"{self.tid}.{i:d}" for i in range(len(self.catalog.catalog))
            ]
            self.catalog.catalog.set_index("event_id", inplace=True)
        else:
            catalog = {}
            with h5.File(os.path.join(db_path_T, filename), mode="r") as f:
                for key in f["catalog"].keys():
                    catalog[key] = f["catalog"][key][()]
                    if catalog[key].dtype.kind == "S":
                        catalog[key] = catalog[key].astype("U")
            extra_attributes = set(catalog.keys()).difference(
                {"longitude", "latitude", "depth", "origin_time"}
            )
            self.catalog = Catalog(
                catalog["longitude"],
                catalog["latitude"],
                catalog["depth"],
                catalog["origin_time"],
                **{key: catalog[key] for key in extra_attributes},
                event_ids=[f"{self.tid}.{i:d}" for i in range(len(catalog["depth"]))],
            )
        if return_events:
            return events

    def write_summary(self, attributes, filename=None, db_path=None, overwrite=True):
        """Write summary of template characteristics.

        hdf5 does not support storing datasets of strings. Therefore, You need
        to convert strings to bytes or this method will raise an error.

        Parameters
        -----------
        attributes: dictionary
            Dictionary with scalars, `numpy.ndarray`, dictionary, or
            `pandas.DataFrame`. The keys of the dictionary are used to name the
            dataset or group in the hdf5 file.
        filename: string, default to None
            Name of the detection file. If None, use the standard file
            and folder naming convention.
        db_path: string, default to None
            Name of the directory where the detection file is located. If None,
            use the standard file and folder naming convention.
        overwrite: boolean, default to True
            If True, overwrite existing datasets or groups.
        """
        if db_path is None:
            db_path, _ = os.path.split(self.where)
        if filename is None:
            filename = f"summary_template{self.tid}.h5"
        with h5.File(os.path.join(db_path, filename), mode="a") as f:
            for key in attributes.keys():
                if key in f:
                    if overwrite:
                        del f[key]
                    else:
                        continue
                if isinstance(attributes[key], (dict, pd.DataFrame)):
                    f.create_group(key)
                    for key2 in attributes[key].keys():
                        f[key].create_dataset(key2, data=attributes[key][key2])
                else:
                    f.create_dataset(key, data=attributes[key])

    # ---------------------------------------------
    # plotting methods
    def plot_detection(
        self,
        idx,
        filename=None,
        db_path=None,
        duration=60.0,
        phase_on_comp={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"},
        offset_ot=10.0,
        **kwargs,
    ):
        """Plot the `idx`-th detection made with this template.

        Parameters
        ------------
        filename: string, default to None
            Name of the detection file. If None, use the standard file
            and folder naming convention.
        db_path: string, default to None
            Name of the directory where the detection file is located. If None,
            use the standard file and folder naming convention.

        """
        if not hasattr(self, "traces"):
            print("Call `Template.read_waveforms` first.")
            return
        db_path_T, filename_T = os.path.split(self.where)
        if filename is None:
            # guess from standard convention
            filename = f"detections_{filename_T}"
        if db_path is None:
            # guess from standard convention
            db_path = db_path_T[::-1].replace(
                "template"[::-1], "matched_filter"[::-1], 1
            )[::-1]
        with h5.File(os.path.join(db_path, filename), mode="r") as f:
            keys = list(f.keys())
            event = Event.read_from_file(hdf5_file=f[keys[idx]])
        event.stations = self.stations.copy()
        event.read_waveforms(
            duration,
            offset_ot=offset_ot,
            phase_on_comp=phase_on_comp,
            time_shifted=False,
            **kwargs,
        )
        fig = event.plot(**kwargs)
        axes = fig.get_axes()
        cc, n_channels = 0.0, 0
        for s, sta in enumerate(event.stations):
            for c, cp in enumerate(event.components):
                for cp_alias in event.component_aliases[cp]:
                    tr = self.traces.select(station=sta, component=cp_alias)
                    if len(tr) > 0:
                        # succesfully retrieved data
                        break
                if len(tr) == 0:
                    continue
                else:
                    tr = tr[0]
                ph = phase_on_comp[cp_alias]
                starttime = (
                    event.origin_time
                    + self.moveouts_win.loc[sta, f"moveouts_{ph.upper()}"]
                )
                endtime = starttime + tr.stats.npts * tr.stats.delta
                time = utils.time_range(starttime, endtime, tr.stats.delta, unit="ms")
                try:
                    event_tr = event.traces.select(station=sta, component=cp_alias)[0]
                    idx1 = utils.sec_to_samp(
                        self.moveouts_win.loc[sta, f"moveouts_{ph.upper()}"]
                        + offset_ot,
                        sr=event_tr.stats.sampling_rate,
                    )
                    idx2 = idx1 + self.n_samples
                    max_amp = np.abs(event_tr.data[idx1:idx2]).max() * kwargs.get(
                        "gain", 1.0e6
                    )
                    cc_ = np.sum(
                        event_tr.data[idx1:idx2] * tr.data[: self.n_samples]
                    ) / np.sqrt(
                        np.sum(event_tr.data[idx1:idx2] ** 2)
                        * np.sum(tr.data[: self.n_samples] ** 2)
                    )
                    if not np.isnan(cc_):
                        cc += cc_
                        n_channels += 1
                except IndexError:
                    # trace not found
                    max_amp = 1.0
                except ValueError:
                    print(sta, cp, idx1, idx2)
                    max_amp = 1.0
                axes[s * len(event.components) + c].plot(
                    time[: self.n_samples],
                    utils.max_norm(tr.data[: self.n_samples]) * max_amp,
                    lw=0.75,
                    color="C3",
                )
        cc /= float(n_channels)
        fig.suptitle(fig._suptitle.get_text() + f" CC={cc:.3f}")
        return fig

    def plot_recurrence_times(
        self, ax=None, annotate_axes=True, figsize=(20, 10), **kwargs
    ):
        """Plot recurrence times vs detection times.

        Parameters
        -----------
        ax: `plt.Axes`, default to None
            If not None, use this `plt.Axes` instance to plot the data.
        """
        import matplotlib.pyplot as plt

        kwargs.setdefault("marker", "v")
        kwargs.setdefault("color", "k")
        kwargs.setdefault("ls", "")
        if ax is not None:
            fig = ax.get_figure()
        else:
            fig = plt.figure(f"recurrence_times_tp{self.tid}", figsize=figsize)
            ax = fig.add_subplot(111)
        if not hasattr(self, "catalog"):
            print("Call `read_catalog` first.")
            return
        rt = (
            self.catalog.origin_time.values[1:] - self.catalog.origin_time.values[:-1]
        ) / 1.0e9  # in sec
        ax.plot(self.catalog.origin_time[1:], rt, **kwargs)
        if annotate_axes:
            ax.set_xlabel("Detection Time")
            ax.set_ylabel("Recurrence Time (s)")
            ax.semilogy()
        return fig


class Family(ABC):
    """An abstract class for several subclasses."""

    def __init__(self):
        self._events = []
        self._update_attributes = []
        self.network = None

    # properties
    @property
    def components(self):
        if not hasattr(self, "network"):
            print("Call `self.set_network(network)` first.")
            return
        return self.network.components

    @property
    def stations(self):
        if not hasattr(self, "network"):
            print("Call `self.set_network(network)` first.")
            return
        return self.network.stations

    @property
    def moveouts_arr(self):
        if not hasattr(self, "_moveouts_arr"):
            self.get_moveouts_arr()
        return self._moveouts_arr

    @property
    def waveforms_arr(self):
        if not hasattr(self, "_waveforms_arr"):
            self.get_waveforms_arr()
        return self._waveforms_arr

    @property
    def _n_events(self):
        return len(self._events)

    def get_moveouts_arr(self):
        _moveouts_arr = np.zeros(
            (self._n_events, self.network.n_stations, self.network.n_components),
            dtype=np.int32,
        )
        for t in range(self._n_events):
            ev_stations = self._events[t].stations
            sta_indexes = self.network.station_indexes.loc[ev_stations]
            _moveouts_arr[t, sta_indexes, :] = self._events[t].moveouts_arr
        self._moveouts_arr = _moveouts_arr

    def get_waveforms_arr(self):
        if "read_waveforms" not in self._update_attributes:
            self.read_waveforms()
        # check the templates' duration
        n_samples = [ev.n_samples for ev in self._events]
        if min(n_samples) != max(n_samples):
            print(
                "Templates have different durations, we cannot return"
                " the template data in a single array."
            )
            return
        self._waveforms_arr = np.stack(
            [
                ev.get_np_array(stations=self.stations, components=self.components)
                for ev in self._events
            ],
            axis=0,
        )
        self._remember("get_waveforms_arr")

    def normalize(self, method="rms"):
        """Normalize the template waveforms.

        Parameters
        ------------
        method: string, default to 'rms'
            Either 'rms' (default) or 'max'.
        """
        if method == "rms":
            norm = np.std(self.waveforms_arr, axis=-1, keepdims=True)
        elif method == "max":
            norm = np.max(np.abs(self.waveforms_arr), axis=-1, keepdims=True)
        norm[norm == 0.0] = 1.0
        self._waveforms_arr /= norm
        self._remember("normalize")

    @abstractmethod
    def read_waveforms(self):
        pass

    def set_network(self, network):
        """Update `self.network` to the new desired `network`.

        Parameters
        -----------
        network: `dataset.Network` instance
            The `Network` instance used to query consistent data accross all
            templates.
        """
        self.network = network
        print("Updating the instance accordingly...")
        for action in self._update_attributes:
            func = getattr(self, action)
            func()

    def set_source_receiver_dist(self):
        """Compute the source-receiver distances for template."""
        for ev in self._events:
            ev.set_source_receiver_dist(self.network)
        self._remember("set_source_receiver_dist")

    def _remember(self, action):
        """Append `action` to the list of processes to remember.

        Parameters
        -----------
        action: string
            Name of the class method that was called once and that has to be
            repeated every time `self.network` is updated.
        """
        if action not in self._update_attributes:
            self._update_attributes.append(action)


class EventGroup(Family):
    """A class for a group of events.

    Each event is represented by a `dataset.Event` instance.
    """

    def __init__(self, events, network):
        """Initialize the EventGroup with a list of `dataset.Event` instances.

        Parameters
        ----------
        events: (n_events,) list of `dataset.Event` instances
            The list of events constituting the group.
        network: `dataset.Network` instance
            The `Network` instance used to query consistent data accross all
            events.
        """
        self.events = events
        self.network = network
        self._update_attributes = []

    # properties
    @property
    def _events(self):
        # alias to use the parent class' methods
        return self.events

    @property
    def n_events(self):
        return len(self.events)

    # methods
    def read_waveforms(self, duration, tag, time_shifted=False, **kwargs):
        """Call `dataset.Event.read_waveform` with each event."""
        self.time_shifted = time_shifted
        for ev in self.events:
            ev.read_waveforms(duration, tag, time_shifted=time_shifted, **kwargs)
        self._remember("read_waveforms")

    def SVDWF_stack(
        self,
        freqmin,
        freqmax,
        sampling_rate,
        expl_var=0.4,
        max_singular_values=5,
        wiener_filter_colsize=None,
    ):
        filtered_data = np.zeros_like(self.waveforms_arr)
        for s in range(len(self.stations)):
            for c in range(len(self.components)):
                filtered_data[:, s, c, :] = utils.SVDWF(
                    self.waveforms_arr[:, s, c, :],
                    max_singular_values=max_singular_values,
                    expl_var=expl_var,
                    freqmin=freqmin,
                    freqmax=freqmax,
                    sampling_rate=sampling_rate,
                    wiener_filter_colsize=wiener_filter_colsize,
                )
                if np.sum(filtered_data[:, s, c, :]) == 0:
                    print(
                        "Problem with station {} ({:d}), component {} ({:d})".format(
                            self.stations[s], s, self.components[c], c
                        )
                    )
        stacked_waveforms = np.mean(filtered_data, axis=0)
        norm = np.max(stacked_waveforms, axis=-1)[..., np.newaxis]
        norm[norm == 0.0] = 1.0
        stacked_waveforms /= norm
        self.filtered_data = filtered_data
        # create a stream with a fake origin time to track future
        # changes in reference time better
        stacked_traces = obs.Stream()
        reference_time = udt(datetime.datetime.now())
        for s, sta in enumerate(self.stations):
            for c, cp in enumerate(self.components):
                tr = obs.Trace()
                tr.stats.station = sta
                tr.stats.component = cp
                tr.stats.sampling_rate = sampling_rate
                tr.data = stacked_waveforms[s, c, :]
                if self.time_shifted:
                    ph = self.events[0].aux_data[f"phase_on_comp{cp.upper()}"]
                    tr.stats.starttime = (
                        reference_time
                        + self.events[0].moveouts.loc[sta, f"moveouts_{ph}"]
                    )
                else:
                    tr.stats.starttime = reference_time
                stacked_traces += tr
        self.stack = Stack(
            stacked_traces,
            self.events[0].moveouts.values,
            self.stations,
            self.events[0].phases,
            components=self.components,
            sampling_rate=sampling_rate,
            filtered_data=filtered_data,
        )
        # fetch auxiliary data
        select = lambda str: str.startswith("phase_on_comp")
        aux_data_to_keep = list(filter(select, self.events[0].aux_data.keys()))
        aux_data = {
            key: self.events[0].aux_data[key]
            for key in aux_data_to_keep
            if key in self.events[0].aux_data
        }
        self.stack.set_aux_data(aux_data)


class TemplateGroup(Family):
    """A class for a group of templates.

    Each template is represented by a `dataset.Template` instance.
    """

    def __init__(self, templates, network, source_receiver_dist=True):
        """Initialize the TemplateGroup instance with a list of
        `dataset.Template` instances.

        Parameters
        ------------
        templates: (n_templates,) list of `dataset.Template` instances
            The list of templates constituting the group.
        network: `dataset.Network` instance
            The `Network` instance used to query consistent data accross all
            templates.
        source_receiver_dist: boolean, default to True
            If True, compute the source-receiver distances on all templates.
        """
        self.templates = templates
        # self._events = self.templates # alias to use the base class methods
        self.network = network
        # self.n_templates = len(self.templates)
        self.tids = np.int32([tp.tid for tp in self.templates])
        # convenient map between template id and the template index in
        # the self.templates list
        self.tindexes = pd.Series(
            index=self.tids, data=np.arange(self.n_templates), name="tid_to_tindex"
        )
        # keep track of the attributes that need updating
        # when self.network changes
        self._update_attributes = []
        # compute source-receiver distances if requested
        if source_receiver_dist:
            self.set_source_receiver_dist()

    @classmethod
    def read_from_files(cls, filenames, network, gids=None, **kwargs):
        """Initialize the TemplateGroup instance given a list of filenames.

        Parameters
        -----------
        filenames: (n_templates,) list of strings
            List of full file paths from which we instanciate the list of
            `dataset.Template` objects.
        network: `dataset.Network` instance
            The `Network` instance used to query consistent data accross all
            templates.
        gids: (n_templates,) list of strings, default to None
            If not None, this should be a list of group ids where the template
            data are stored in their hdf5 files.

        Returns
        --------
        template_group: TemplateGroup instance
            The initialized TemplateGroup instance.
        """
        templates = []
        for i, fn in enumerate(filenames):
            if gids is not None:
                gid = gids[i]
            else:
                gid = None
            db_path, db_filename = os.path.split(fn)
            templates.append(
                Template.read_from_file(db_filename, db_path=db_path, gid=gid)
            )
        return cls(templates, network)

    # properties
    @property
    def _events(self):
        # alias to use the parent class' methods
        return self.templates

    @property
    def n_templates(self):
        return len(self.templates)

    @property
    def dir_errors(self):
        if not hasattr(self, "_dir_errors"):
            self.compute_dir_errors()
        return self._dir_errors

    @property
    def ellipsoid_dist(self):
        if not hasattr(self, "_ellipsoid_dist"):
            self.compute_ellipsoid_dist()
        return self._ellipsoid_dist

    @property
    def intertemplate_cc(self):
        if not hasattr(self, "_intertemplate_cc"):
            self.compute_intertemplate_cc()
        return self._intertemplate_cc

    @property
    def intertemplate_dist(self):
        if not hasattr(self, "_intertemplate_dist"):
            self.compute_intertemplate_dist()
        return self._intertemplate_dist

    @property
    def network_to_template_map(self):
        if not hasattr(self, "_network_to_template_map"):
            self.set_network_to_template_map()
        return self._network_to_template_map

    # methods
    def box(self, lon_min, lon_max, lat_min, lat_max, inplace=False):
        """Keep templates inside the requested geographic bounds.

        Parameters
        -----------
        lon_min: scalar float
            Minimum longitude, in decimal degrees.
        lon_max: scalar float
            Maximum longitude, in decimal degrees.
        lat_min: scalar float
            Minimum latitude, in decimal degrees.
        lat_max: scalar float
            Maximum latitude, in decimal degrees.
        """
        templates_inside = []
        for template in self.templates:
            if (
                (template.longitude >= lon_min)
                & (template.longitude <= lon_max)
                & (template.latitude >= lat_min)
                & (template.latitude <= lat_max)
            ):
                templates_inside.append(template)
        if inplace:
            self.templates = templates_inside
            self.tids = np.int32([tp.tid for tp in self.templates])
            self.tindexes = pd.Series(
                index=self.tids, data=np.arange(self.n_templates), name="tid_to_tindex"
            )
            if hasattr(self, "_intertemplate_dist"):
                self._intertemplate_dist = self._intertemplate_dist.loc[
                    self.tids, self.tids
                ]
            if hasattr(self, "_dir_errors"):
                self._dir_errors = self._dir_errors.loc[self.tids, self.tids]
            if hasattr(self, "_ellipsoid_dist"):
                self._ellipsoid_dist = self._ellipsoid_dist.loc[self.tids, self.tids]
            if hasattr(self, "_intertemplate_cc"):
                self._intertemplate_cc = self._intertemplate_cc.loc[
                    self.tids, self.tids
                ]
            if hasattr(self, "_waveforms_arr"):
                self.get_waveforms_arr()
        else:
            new_template_group = TemplateGroup(templates_inside, self.network)
            new_tids = new_template_group.tids
            if hasattr(self, "_intertemplate_dist"):
                new_template_group._intertemplate_dist = self._intertemplate_dist.loc[
                    new_tids, new_tids
                ]
            if hasattr(self, "_dir_errors"):
                new_template_group._dir_errors = self._dir_errors.loc[
                    new_tids, new_tids
                ]
            if hasattr(self, "_ellipsoid_dist"):
                new_template_group._ellipsoid_dist = self._ellipsoid_dist.loc[
                    new_tids, new_tids
                ]
            if hasattr(self, "_intertemplate_cc"):
                new_template_group._intertemplate_cc = self._intertemplate_cc.loc[
                    new_tids, new_tids
                ]
            return new_template_group

    def compute_intertemplate_dist(self):
        """Compute the template-pairwise distances, in km."""
        longitudes = np.float32([tp.longitude for tp in self.templates])
        latitudes = np.float32([tp.latitude for tp in self.templates])
        depths = np.float32([tp.depth for tp in self.templates])
        _intertemplate_dist = utils.compute_distances(
            longitudes, latitudes, depths, longitudes, latitudes, depths
        )
        self._intertemplate_dist = pd.DataFrame(
            index=self.tids, columns=self.tids, data=_intertemplate_dist
        )

    def compute_dir_errors(self):
        """Compute length of uncertainty ellipsoid in inter-template direction.

        New Attributes
        --------------
        _dir_errors: (n_templates, n_templates) pandas.DataFrame
            The length, in kilometers, of the uncertainty ellipsoid in the
            inter-template direction.
            Example: self.directional_errors.loc[tid1, tid2] is the width of
            template tid1's uncertainty ellipsoid in the direction of
            template tid2.
        """
        from cartopy import crs

        # X: west, Y: south, Z: downward
        s_68_3df = 3.52
        s_90_3df = 6.251
        print("Computing the inter-template directional errors...")
        longitudes = np.float32([tp.longitude for tp in self.templates])
        latitudes = np.float32([tp.latitude for tp in self.templates])
        depths = np.float32([tp.depth for tp in self.templates])
        # ----------------------------------------------
        #      Define the projection used to
        #      work in a cartesian space
        # ----------------------------------------------
        data_coords = crs.PlateCarree()
        projection = crs.Mercator(
            central_longitude=np.mean(longitudes),
            min_latitude=latitudes.min(),
            max_latitude=latitudes.max(),
        )
        XY = projection.transform_points(data_coords, longitudes, latitudes)
        cartesian_coords = np.stack([XY[:, 0], XY[:, 1], depths], axis=1)
        # compute the directional errors
        _dir_errors = np.zeros((self.n_templates, self.n_templates), dtype=np.float32)
        for t in range(self.n_templates):
            unit_direction = cartesian_coords - cartesian_coords[t, :]
            unit_direction /= np.sqrt(np.sum(unit_direction**2, axis=1))[
                :, np.newaxis
            ]
            # this operation produced NaNs for i=t
            unit_direction[np.isnan(unit_direction)] = 0.0
            if hasattr(self.templates[t], "cov_mat"):
                # compute the length of the covariance ellipsoid
                # in the direction that links the two earthquakes
                cov_dir = np.abs(
                    np.sum(
                        self.templates[t].cov_mat.dot(unit_direction.T)
                        * unit_direction.T,
                        axis=0,
                    )
                )
                # covariance is unit of [distance**2], therefore we need the sqrt:
                _dir_errors[t, :] = np.sqrt(s_68_3df * cov_dir)
            else:
                # use default large error
                _dir_errors[t, :] = 15.0
        self._dir_errors = pd.DataFrame(
            index=self.tids, columns=self.tids, data=_dir_errors
        )

    def compute_ellipsoid_dist(self):
        """Compute separation between unc. ellipsoids in inter-template dir.

        Can be negative if the uncertainty ellipsoids overlap.
        """
        self._ellipsoid_dist = (
            self.intertemplate_dist - self.dir_errors - self.dir_errors.T
        )

    def compute_intertemplate_cc(
        self,
        distance_threshold=5.0,
        n_stations=10,
        max_lag=10,
        save_cc=False,
        compute_from_scratch=False,
        device="cpu",
        progress=False,
    ):
        """Compute the pairwise template CCs.

        Parameters
        -----------
        distance_threshold: float, default to 5
            The distance threshold, in kilometers, between two
            uncertainty ellipsoids under which similarity is computed.
        n_stations: integer, default to 10
            The number of stations closest to each template used in
            the computation of the average CC.
        max_lag: integer, default to 10
            Maximum lag, in samples, allowed when searching for the
            maximum CC on each channel. This is to account for small
            discrepancies in windowing that could occur for two templates
            highly similar but associated to slightly different locations.
        save_cc: boolean, default to False
            If True, save the inter-template CCs in the same folder as
            `self.templates[0]` and with filename 'intertp_cc.h5'.
        compute_from_scratch: boolean, default to False
            If True, force to compute the inter-template CCs from scratch.
            Useful if user knows the computation is faster than reading a
            potentially large file.
        device: string, default to 'cpu'
            Either 'cpu' or 'gpu'.
        progress: boolean, default to False
            If True, print progress bar with `tqdm`.
        """
        import fast_matched_filter as fmf  # clearly need some optimization

        disable = np.bitwise_not(progress)

        # try reading the inter-template CC from db
        db_path, db_filename = os.path.split(self.templates[0].where)
        cc_fn = os.path.join(db_path, "intertp_cc.h5")
        if os.path.isfile(cc_fn):
            _intertemplate_cc = self._read_intertp_cc(cc_fn)
            if (
                len(np.intersect1d(self.tids, np.int32(_intertemplate_cc.index)))
                == self.n_templates
            ):
                # all current templates are contained in intertp_cc
                self._intertemplate_cc = _intertemplate_cc.loc[self.tids, self.tids]
                print(f"Read inter-template CCs from {cc_fn}.")
            else:
                compute_from_scratch = True
        else:
            compute_from_scratch = True
        if compute_from_scratch:
            # compute from scratch
            self.n_closest_stations(n_stations)
            print("Computing the similarity matrix...")
            # format arrays for FMF
            data_arr = self.waveforms_arr.copy()
            template_arr = self.waveforms_arr[..., max_lag:-max_lag]
            moveouts_arr = np.zeros(self.waveforms_arr.shape[:-1], dtype=np.int32)
            intertp_cc = np.zeros(
                (self.n_templates, self.n_templates), dtype=np.float32
            )
            n_stations, n_components = moveouts_arr.shape[1:]
            # use FMF on one template at a time against all others
            for t, template in tqdm(
                enumerate(self.templates), desc="Inter-tp CC", disable=disable
            ):
                # print(f'--- {t} / {self.n_templates} ---')
                weights = np.zeros(template_arr.shape[:-1], dtype=np.float32)
                weights[:, self.network_to_template_map[t, ...]] = 1.0
                weights /= np.sum(weights, axis=(1, 2), keepdims=True)
                above_thrs = self.ellipsoid_dist[self.tids[t]] > distance_threshold
                weights[above_thrs, ...] = 0.0
                keep = np.sum(weights != 0.0, axis=(1, 2)) > 0
                cc = fmf.matched_filter(
                    template_arr[keep, ...],
                    moveouts_arr[keep, ...],
                    weights[keep, ...],
                    data_arr[t, ...],
                    1,
                    arch=device,
                    network_sum=False,
                    check_zeros=False,
                )
                intertp_cc[t, keep] = np.sum(
                    weights[keep, ...] * np.max(cc, axis=1), axis=(-1, -2)
                )
            # make the CC matrix symmetric by averaging the lower
            # and upper triangles
            _intertemplate_cc = (intertp_cc + intertp_cc.T) / 2.0
            self._intertemplate_cc = pd.DataFrame(
                index=self.tids, columns=self.tids, data=_intertemplate_cc
            )
        if compute_from_scratch and save_cc:
            print(f"Saving inter-tp CC to {cc_fn}")
            self._save_intertp_cc(self._intertemplate_cc, cc_fn)

    @staticmethod
    def _save_intertp_cc(intertp_cc, fullpath):
        """Save inter-template correlation coefficients.

        Parameters
        -----------
        intertp_cc: `pd.DataFrame`
            The inter-template CC computed by `compute_intertemplate_cc`.
        fullpath: string
            Full path to output file.
        """
        with h5.File(fullpath, mode="w") as f:
            f.create_dataset("tids", data=np.int32(intertp_cc.columns))
            f.create_dataset("intertp_cc", data=np.float32(intertp_cc.values))

    @staticmethod
    def _read_intertp_cc(fullpath):
        """Read inter-template correlation coefficients from file.

        Parameters
        -----------
        fullpath: string
            Full path to output file.

        Returns
        --------
        intertp_cc: `pd.DataFrame`
            The inter-template CC in a `pd.DataFrame`.
        """
        with h5.File(fullpath, mode="r") as f:
            tids = f["tids"][()]
            intertp_cc = f["intertp_cc"][()]
        return pd.DataFrame(index=tids, columns=tids, data=intertp_cc)

    def read_waveforms(self, progress=False):
        """
        Parameters
        ----------
        progress: boolean, default to False
            If True, print progress bar with `tqdm`.
        """
        disable = np.bitwise_not(progress)
        for tp in tqdm(self.templates, desc="Reading waveforms", disable=disable):
            tp.read_waveforms(stations=self.stations, components=self.components)
        self._remember("read_waveforms")

    def set_network_to_template_map(self):
        """Compute the map between network arrays and template data.

        Template data are broadcasted to fit the dimensions of the network
        arrays. This method computes the `network_to_template_map` that tells
        which stations and channels are used on each template. For example:
        `network_to_template_map[t, s, c] = False` means that station s and
        channel c are not used on template t.
        """
        _network_to_template_map = np.zeros(
            (self.n_templates, self.network.n_stations, self.network.n_components),
            dtype=np.bool,
        )
        # 1) find the non-zero channels
        _network_to_template_map = ~(np.sum(self.waveforms_arr, axis=-1) == 0.0)
        # 2) only keep the stations that were selected on each template
        selected_stations = np.zeros(_network_to_template_map.shape, dtype=bool)
        for t, tp in enumerate(self.templates):
            valid_sta = self.network.station_indexes.loc[tp.stations]
            selected_stations[t, valid_sta, :] = True
        _network_to_template_map = _network_to_template_map & selected_stations
        self._network_to_template_map = _network_to_template_map

    def n_best_SNR_stations(self, n, available_stations=None):
        """Adjust `self.stations` on each template to the `n` best SNR stations.


        Find the `n` best stations and modify `self.stations` accordingly.
        The instance's properties will also change accordingly.

        Parameters
        ----------------
        n: scalar int
            The `n` closest stations.
        available_stations: list of strings, default to None
            The list of stations from which we search the closest stations.
            If some stations are known to lack data, the user
            may choose to not include these in the closest stations.
        """
        for tp in self.templates:
            tp.n_best_SNR_stations(n, available_stations=available_stations)
        if hasattr(self, "_network_to_template_map"):
            del self._network_to_template_map

    def n_closest_stations(self, n, available_stations=None):
        """Adjust `self.stations` on each template to the `n` closest stations.


        Find the `n` closest stations and modify `self.stations` accordingly.
        The instance's properties will also change accordingly.

        Parameters
        ----------------
        n: scalar int
            The `n` closest stations to fetch.
        available_stations: list of strings, default to None
            The list of stations from which we search the closest stations.
            If some stations are known to lack data, the user
            may choose to not include these in the closest stations.
        """
        for tp in self.templates:
            tp.n_closest_stations(n, available_stations=available_stations)
        if hasattr(self, "_network_to_template_map"):
            del self._network_to_template_map

    def read_catalog(
        self, extra_attributes=[], fill_value=np.nan, progress=False, **kwargs
    ):
        """Build a catalog from all templates' detections.

        Work only if folder and file names follow the standard convention.

        Parameters
        ------------
        extra_attributes: list of strings, default to []
            Attributes to read in addition to the default 'longitude',
            'latitude', 'depth', and 'origin_time'.
        fill_value: string, int, or float, default to np.nan
            Default value if the target attribute does not exist.
        progress: boolean, default to False
            If True, print progress bar with `tqdm`.
        """
        disable = np.bitwise_not(progress)
        for template in tqdm(self.templates, desc="Reading catalog", disable=disable):
            if not hasattr(template, "catalog"):
                template.read_catalog(
                    extra_attributes=extra_attributes, fill_value=fill_value, **kwargs
                )
        # concatenate all catalogs
        self.catalog = Catalog.concatenate(
            [template.catalog.catalog for template in self.templates],
            ignore_index=False,
        )

    def remove_multiples(
        self,
        n_closest_stations=10,
        dt_criterion=4.0,
        distance_criterion=1.0,
        speed_criterion=5.0,
        similarity_criterion=-1.0,
        progress=False,
        **kwargs,
    ):
        """Search for events detected by multiple templates.

        Parameters
        -----------
        n_closest_stations: integer, default to 10
            In case template similarity is taken into account,
            this is the number of stations closest to each template
            that are used in the calculation of the average cc.
        dt_criterion: float, default to 4
            Time interval, in seconds, under which two events are
            examined for redundancy.
        distance_criterion: float, default to 1
            Distance threshold, in kilometers, between two uncertainty
            ellipsoids under which two events are examined for redundancy.
        speed_criterion: float, default to 5
            Speed criterion, in km/s, below which the inter-event time and
            inter-event distance can be explained by errors in origin times and
            a reasonable P-wave speed.
        similarity_criterion: float, default to -1
            Template similarity threshold, in terms of average CC, over
            which two events are examined for redundancy. The default
            value of -1 is always verified, meaning that similarity is
            actually not taken into account.
        progress: boolean, default to False
            If True, print progress bar with `tqdm`.
        """
        disable = np.bitwise_not(progress)
        if not hasattr(self, "catalog"):
            self.read_catalog(extra_attributes=["cc"])
        self.catalog.catalog["origin_time_sec"] = (
            self.catalog.catalog["origin_time"]
            .values.astype("datetime64[ms]")
            .astype("float64")
            / 1000.0
        )
        self.catalog.catalog.sort_values("origin_time_sec", inplace=True)
        self.catalog.catalog["interevent_time_sec"] = np.hstack(
            (
                [0.0],
                self.catalog.catalog["origin_time_sec"].values[1:]
                - self.catalog.catalog["origin_time_sec"].values[:-1],
            )
        )
        # alias:
        catalog = self.catalog.catalog
        if similarity_criterion > -1.0:
            if not hasattr(self, "_intertemplate_cc"):
                self.compute_intertemplate_cc(
                    distance_threshold=distance_criterion,
                    n_stations=n_closest_stations,
                    max_lag=kwargs.get("max_lag", 10),
                    device=kwargs.get("device", "cpu"),
                )
        # -----------------------------------
        t1 = give_time()
        print("Searching for events detected by multiple templates")
        print(
            "All events occurring within {:.1f} sec, with uncertainty "
            "ellipsoids closer than {:.1f} km will and "
            "inter-template CC larger than {:.2f} be considered the same".format(
                dt_criterion, distance_criterion, similarity_criterion
            )
        )
        n_events = len(self.catalog.catalog)
        index_pool = np.arange(n_events)
        # dt_criterion = np.timedelta64(int(1000.0 * dt_criterion), "ms")
        unique_event = np.ones(n_events, dtype=np.bool)
        for n1 in tqdm(range(n_events), desc="Removing multiples", disable=disable):
            if not unique_event[n1]:
                continue
            tid1 = catalog["tid"].iloc[n1]
            # apply the time criterion
            # ---------- version 4 ------------
            n2 = n1 + 1
            if n2 < n_events:
                dt_n1n2 = catalog["interevent_time_sec"].iloc[n2]
            else:
                continue
            temporal_neighbors = [n1]
            while dt_n1n2 < dt_criterion:
                temporal_neighbors.append(n2)
                n2 += 1
                if n2 >= n_events:
                    break
                dt_n1n2 += catalog["interevent_time_sec"].iloc[n2]
            temporal_neighbors = np.array(temporal_neighbors).astype("int64")
            if len(temporal_neighbors) == 1:
                # did not find any temporal neighbors
                continue
            # remove events that were already flagged as non unique
            temporal_neighbors = temporal_neighbors[unique_event[temporal_neighbors]]
            candidates = temporal_neighbors
            if len(candidates) == 1:
                continue
            # get template ids of all events that passed the time criterion
            tids_candidates = np.array(
                [catalog["tid"].iloc[idx] for idx in candidates]
            ).astype("int64")
            # apply the spatial criterion to the distance between
            # uncertainty ellipsoids
            ellips_dist = self.ellipsoid_dist.loc[tid1, tids_candidates].values
            time_diff = (
                catalog["origin_time_sec"].iloc[candidates]
                - catalog["origin_time_sec"].iloc[n1]
            )
            # if the time difference were to be entirely due to errors in
            # origin times, what would be wave speed explaining the location
            # differences?
            # time_diff = 0 is the time_diff between n1 and n1
            time_diff[time_diff == 0.0] = 1.0
            speed_diff = ellips_dist / time_diff
            if similarity_criterion > -1.0:
                similarities = self.intertemplate_cc.loc[tid1, tids_candidates].values
                multiples = candidates[
                    np.where(
                        (
                            (ellips_dist < distance_criterion)
                            | (speed_diff < speed_criterion)
                        )
                        & (similarities >= similarity_criterion)
                    )[0]
                ]
            else:
                multiples = candidates[np.where(ellips_dist < distance_criterion)[0]]
            if len(multiples) <= 1:
                continue
            else:
                unique_event[multiples] = False
                # find best CC and keep it
                ccs = catalog["cc"].values[multiples]
                best_cc = multiples[ccs.argmax()]
                unique_event[best_cc] = True
        t2 = give_time()
        print(f"{t2-t1:.2f}s to flag the multiples")
        # -------------------------------------------
        catalog["unique_event"] = unique_event
        for tid in self.tids:
            tt = self.tindexes.loc[tid]
            cat_indexes = catalog.index[catalog["tid"] == tid]
            self.templates[tt].catalog.catalog["unique_event"] = np.zeros(
                len(self.templates[tt].catalog.catalog), dtype=bool
            )
            self.templates[tt].catalog.catalog.loc[
                cat_indexes, "unique_event"
            ] = catalog.loc[cat_indexes, "unique_event"].values

    # plotting routines
    def plot_detection(self, idx, **kwargs):
        """

        Parameters
        -----------
        idx: scalar int
            Event index in `self.catalog.catalog`.

        Returns
        ---------
        fig: `plt.Figure`
            The figure showing the detected event.
        """
        tid, evidx = self.catalog.catalog.index[idx].split(".")
        tt = self.tindexes.loc[int(tid)]
        print("Plotting:")
        print(self.catalog.catalog.iloc[idx])
        fig = self.templates[tt].plot_detection(int(evidx), **kwargs)
        return fig

    def plot_recurrence_times(self, figsize=(20, 10), progress=False, **kwargs):
        """Plot recurrence times vs detection times, template-wise.

        Parameters
        -----------
        figsize: tuple of floats, default to (20, 10)
            Size in inches of the figure (width, height).
        progress: boolean, default to False
            If True, print progress bar with `tqdm`.
        """
        import matplotlib.pyplot as plt

        disable = np.bitwise_not(progress)

        fig = plt.figure("recurrence_times", figsize=figsize)
        ax = fig.add_subplot(111)
        for template in tqdm(
            self.templates, desc="Plotting rec. times", disable=disable
        ):
            template.plot_recurrence_times(ax=ax, annotate_axes=False, **kwargs)
        ax.set_xlabel("Detection Time")
        ax.set_ylabel("Recurrence Time (s)")
        ax.semilogy()
        return fig


class Stack(Event):
    """A modification of the Event class for stacked events."""

    def __init__(
        self,
        stacked_traces,
        moveouts,
        stations,
        phases,
        latitude=None,
        longitude=None,
        depth=None,
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
        sampling_rate=None,
        components=["N", "E", "Z"],
        aux_data={},
        id=None,
        filtered_data=None,
    ):
        """Initialize an Event instance with basic attributes.

        Parameters
        -----------
        stacked_traces: `obspy.Stream`
            Traces with the stacked waveforms.
        moveouts: (n_stations, n_phases) float `numpy.ndarray`
            Moveouts, in seconds, for each station and each phase.
        stations: List of strings
            List of station names corresponding to `moveouts`.
        phases: List of strings
            List of phase names corresponding to `moveouts`.
        latitude: scalar float, default to None
            Event latitude.
        longitude: scalar float, default to None
            Event longitude.
        depth: scalar float, default to None
            Event depth.
        sampling_rate: scalar float, default to None
            Sampling rate (Hz) of the waveforms. It should be different from
            None only if you plan on reading preprocessed data with a fixed
            sampling rate.
        components: List of strings, default to ['N','E','Z']
            List of the components to use in reading and plotting methods.
        component_aliases: Dictionary, optional
            Each entry of the dictionary is a list of strings.
            `component_aliases[comp]` is the list of all aliases used for
            the same component 'comp'. For example, `component_aliases['N'] =
            ['N', '1']` means that both the 'N' and '1' channels will be mapped
            to the Event's 'N' channel.
        aux_data: dictionary, optional
            Dictionary with auxiliary data (see `dataset.Event`). Note that
            aux_data['phase_on_comp{cp}'] is necessary to call
            `self.read_waveforms`.
        id: string, default to None
            Identifying label.
        filtered_data: (n_events, n_stations, n_components, n_samples)
        `numpy.ndarray`, default to None
            The event waveforms filtered by the SVDWF technique.
        """
        self.stacked_traces = stacked_traces
        self.filtered_data = filtered_data
        self.origin_time = udt(datetime.datetime.now())
        self.date = self.origin_time  # for compatibility with Data class
        self.stations = np.asarray(stations).astype("U")
        self.components = np.asarray(components).astype("U")
        self.component_aliases = component_aliases
        self.phases = np.asarray(phases).astype("U")
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.where = ""
        self.sampling_rate = sampling_rate
        if moveouts.dtype in (np.int32, np.int64):
            print(
                "Integer data type detected for moveouts. Are you sure these"
                " are in seconds?"
            )
        # format moveouts in a Pandas data frame
        mv_table = {"stations": self.stations}
        for p, ph in enumerate(self.phases):
            mv_table[f"moveouts_{ph.upper()}"] = moveouts[:, p]
        self.moveouts = pd.DataFrame(mv_table)
        self.moveouts.set_index("stations", inplace=True)
        if id is None:
            self.id = self.origin_time.strftime("%Y%m%d_%H%M%S")
        else:
            self.id = id

    def read_waveforms(
        self,
        duration,
        phase_on_comp={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"},
        offset_phase={"P": 1.0, "S": 4.0},
        time_shifted=True,
        offset_ot=cfg.BUFFER_EXTRACTED_EVENTS_SEC,
    ):
        """Read waveform data.

        Parameters
        -----------
        duration: scalar float
            Duration, in seconds, of the extracted time windows.
        phase_on_comp: dictionary, optional
            Dictionary defining which seismic phase is extracted on each
            component. For example, phase_on_comp['N'] gives the phase that is
            extracted on the north component.
        offset_phase: dictionary, optional
            Dictionary defining when the time window starts with respect to the
            pick. A positive offset means the window starts before the pick. Not
            used if `time_shifted` is False.
        time_shifted: boolean, default to True
            If True, the moveouts are used to extract time windows from specific
            seismic phases. If False, windows are simply extracted with respect to
            the origin time.
        offset_ot: scalar float, default to `cfg.BUFFER_EXTRACTED_EVENTS_SEC`
            Only used if `time_shifted` is False. Time, in seconds, taken before
            `origin_time`.
        """
        self.traces = obs.Stream()
        self.n_samples = utils.sec_to_samp(duration, sr=self.sr)
        for s, sta in enumerate(self.stations):
            for c, cp in enumerate(self.components):
                ph = phase_on_comp[cp].upper()
                if time_shifted:
                    mv = (
                        offset_ot
                        + self.moveouts[f"moveouts_{ph}"].loc[sta]
                        - offset_phase[ph]
                    )
                else:
                    mv = offset_ot
                tr = (
                    self.stacked_traces.select(station=sta, component=cp)[0]
                    .slice(
                        starttime=self.origin_time + mv,
                        endtime=self.origin_time + mv + duration,
                    )
                    .copy()
                )
                tr = tr.trim(
                    starttime=self.origin_time + mv,
                    endtime=self.origin_time + mv + duration,
                    pad=True,
                    fill_value=0.0,
                )
                self.traces += tr
        aux_data = {
            f"offset_{ph.upper()}": offset_phase[ph] for ph in offset_phase.keys()
        }
        self.set_aux_data(aux_data)
        self.set_availability(stations=self.stations)

    def pick_PS_phases(
        self,
        duration,
        threshold_P=0.60,
        threshold_S=0.60,
        mini_batch_size=126,
        phase_on_comp={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"},
        n_threshold=1,
        err_threshold=100,
        central="mean",
        upsampling=1,
        downsampling=1,
        **kwargs,
    ):
        """Use PhaseNet (Zhu et al., 2019) to pick P and S waves.

        Note1: PhaseNet must be used with 3-comp data.
        Note2: Extra kwargs are passed to
        `phasenet.wrapper.automatic_detection`.

        Parameters
        -----------
        duration: scalar float
            Duration, in seconds, of the time window to process to search for P
            and S wave arrivals.
        threshold_P: scalar float, default to 0.60
            Threshold on PhaseNet's probabilities to trigger the identification
            of a P-wave arrival.
        threshold_S: scalar float, default to 0.60
            Threshold on PhaseNet's probabilities to trigger the identification
            of a S-wave arrival.
        mini_batch_size: scalar int, default to 126
            Number of traces processed in a single batch by PhaseNet. This
            shouldn't have to be tuned.
        phase_on_comp: dictionary, optional
            Dictionary defining which seismic phase is extracted on each
            component. For example, phase_on_comp['N'] gives the phase that is
            extracted on the north component.
        upsampling: scalar integer, default to 1
            Upsampling factor applied before calling PhaseNet.
        downsampling: scalar integer, default to 1
            Downsampling factor applied before calling PhaseNet.
        n_threshold: scalar int, optional
            Used if `self.filtered_data` is not None. Minimum number of
            successful picks to keep the phase. Default to 1.
        err_threshold: scalar int or float, optional
            Used if `self.filtered_data` is not None. Maximum error (in samples)
            on pick to keep the phase. Default to 100.
        central: string, optional
            Used if `self.filtered_data` is not None. Either 'mean' or 'mode'.
            The pick is either taken as the mean or the mode of the empirical
            distribution of picks.
        """
        from phasenet import wrapper as PN

        # read waveforms in "picking" mode
        self.read_waveforms(
            duration,
            offset_ot=0.0,
            phase_on_comp=phase_on_comp,
            time_shifted=False,
            **kwargs,
        )
        data_arr = self.get_np_array(self.stations, components=["N", "E", "Z"])
        if self.filtered_data is not None:
            data_arr = np.concatenate(
                [data_arr[np.newaxis, ...], self.filtered_data], axis=0
            )
            if upsampling > 1 or downsampling > 1:
                from scipy.signal import resample_poly
                data_arr = resample_poly(
                        data_arr,
                        upsampling,
                        downsampling,
                        axis=-1
                        )
                # momentarily update samping_rate
                sampling_rate0 = float(self.sampling_rate)
                self.sampling_rate = self.sr * upsampling / downsampling
            # call PhaseNet
            PhaseNet_probas, PhaseNet_picks = PN.automatic_picking(
                data_arr,
                self.stations,
                mini_batch_size=mini_batch_size,
                format="ram",
                threshold_P=threshold_P,
                threshold_S=threshold_S,
                **kwargs,
            )
            PhaseNet_picks = PN.get_all_picks(PhaseNet_picks)
            PhaseNet_picks = PN.fit_probability_density(PhaseNet_picks)
            PhaseNet_picks = PN.select_picks_family(
                PhaseNet_picks, n_threshold, err_threshold, central=central
            )
            # format picks in pandas DataFrame
            pandas_picks = {"stations": self.stations}
            for ph in ["P", "S"]:
                rel_picks_sec = np.zeros(len(self.stations), dtype=np.float32)
                err_picks = np.zeros(len(self.stations), dtype=np.float32)
                abs_picks = np.zeros(len(self.stations), dtype=object)
                for s, sta in enumerate(self.stations):
                    if sta in PhaseNet_picks[f"{ph}_picks"].keys():
                        rel_picks_sec[s] = PhaseNet_picks[f"{ph}_picks"][sta] / self.sr
                        err_picks[s] = PhaseNet_picks[f"{ph}_err"][sta] / self.sr
                        abs_picks[s] = (
                            self.traces.select(station=sta)[0].stats.starttime
                            + rel_picks_sec[s]
                        )
                pandas_picks[f"{ph}_picks_sec"] = rel_picks_sec
                pandas_picks[f"{ph}_err"] = err_picks
                pandas_picks[f"{ph}_abs_picks"] = abs_picks
            self.picks = pd.DataFrame(pandas_picks)
            self.picks.set_index("stations", inplace=True)
            self.picks.replace(0.0, np.nan, inplace=True)
            if upsampling > 1 or downsampling > 1:
                # reset the sampling rate to initial value
                self.sampling_rate = sampling_rate0
        else:
            super(Stack, self).pick_PS_phases(
                duration,
                "",
                threshold_P=threshold_P,
                threshold_S=threshold_S,
                read_waveforms=False,
            )


# -------------------------------------------------
# old classes that will disappear in the future

# class Stack(object):
#    """A class for stacked waveforms.
#
#    """
#
#    def __init__(self,
#                 stations,
#                 components,
#                 tid=None,
#                 sampling_rate=cfg.SAMPLING_RATE_HZ):
#
#        self.stations = stations
#        self.components = components
#        self.sampling_rate = sampling_rate
#        if isinstance(self.stations, str):
#            self.stations = [self.stations]
#        if isinstance(self.components, str):
#            self.components = [self.components]
#        if not isinstance(self.stations, list):
#            self.stations = list(self.stations)
#        if not isinstance(self.components, list):
#            self.components = list(self.components)
#        if tid is not None:
#            self.template_idx = tid
#
#    def add_data(self, waveforms):
#
#        self.waveforms = waveforms
#        self.traces = obs.Stream()
#        for s, sta in enumerate(self.stations):
#            for c, cp in enumerate(self.components):
#                tr = obs.Trace()
#                tr.data = self.waveforms[s, c, :]
#                tr.stats.station = sta
#                # not really a channel, but the component
#                tr.stats.channel = cp
#                tr.stats.sampling_rate = self.sampling_rate
#                self.traces += tr
#
#    def SVDWF_stack(self, detection_waveforms, freqmin, freqmax,
#                    expl_var=0.4, max_singular_values=5,
#                    wiener_filter_colsize=None):
#        filtered_data = np.zeros_like(detection_waveforms)
#        for s in range(len(self.stations)):
#            for c in range(len(self.components)):
#                filtered_data[:, s, c, :] = utils.SVDWF(
#                        detection_waveforms[:, s, c, :],
#                        max_singular_values=max_singular_values,
#                        expl_var=expl_var,
#                        freqmin=freqmin,
#                        freqmax=freqmax,
#                        sampling_rate=self.sampling_rate,
#                        wiener_filter_colsize=wiener_filter_colsize)
#                if np.sum(filtered_data[:, s, c, :]) == 0:
#                    print('Problem with station {} ({:d}), component {} ({:d})'.
#                            format(self.stations[s], s, self.components[c], c))
#        stacked_waveforms = np.mean(filtered_data, axis=0)
#        norm = np.max(stacked_waveforms, axis=-1)[..., np.newaxis]
#        norm[norm == 0.] = 1.
#        stacked_waveforms /= norm
#        self.add_data(stacked_waveforms)
#        self.data = filtered_data
#
#    def read_data(self,
#                  filename,
#                  db_path_S,
#                  db_path=cfg.INPUT_PATH):
#
#        with h5.File(os.path.join(db_path, db_path_S,
#            '{}meta.h5'.format(filename)), mode='r') as f:
#            file_stations = f['stations'][()].astype('U').tolist()
#            file_components = f['components'][()].astype('U').tolist()
#        with h5.File(os.path.join(db_path, db_path_S,
#            '{}wav.h5'.format(filename)), mode='r') as f:
#            file_waveforms = f['waveforms'][()]
#        station_map = []
#        for s in range(len(self.stations)):
#            station_map.append(file_stations.index(self.stations[s]))
#        component_map = []
#        for c in range(len(self.components)):
#            component_map.append(file_components.index(self.components[c]))
#        station_map = np.int32(station_map)
#        component_map = np.int32(component_map)
#        self.waveforms = file_waveforms[station_map, :, :][:, component_map, :]
#        self.traces = obs.Stream()
#        for s, sta in enumerate(self.stations):
#            for c, cp in enumerate(self.components):
#                tr = obs.Trace()
#                tr.data = self.waveforms[s, c, :]
#                tr.stats.station = sta
#                # not really a channel, but the component
#                tr.stats.channel = cp
#                tr.stats.sampling_rate = self.sampling_rate
#                self.traces += tr

# class FamilyCatalog(object):
#
#    def __init__(self, filename, db_path_M, db_path=cfg.INPUT_PATH):
#        self.filename = filename
#        self.db_path_M = db_path_M
#        self.db_path = db_path
#        self.full_filename = os.path.join(
#                db_path, db_path_M, filename)
#
#    def read_data(self, items_in=[], items_out=[]):
#        """Attach the requested attributes to the FamilyCatalog instance.
#
#        Parameters
#        -----------
#        items_in: list of strings, default to an empty list
#            List of items to read from the catalog file.
#            If an empty list is provided, then all items are read.
#        items_out: list of strings, default to an empty list
#            List of items to reject when reading from the catalog file.
#            If an empty list is provided, no items are discarded.
#        """
#        if not isinstance(items_in, list):
#            items_in = [items_in]
#        if not isinstance(items_out, list):
#            items_out = [items_out]
#        # attach 'template_idx' as a default item
#        items_in = list(set(items_in+['template_idx']))
#        with h5.File(self.full_filename, mode='r') as f:
#            for key in f.keys():
#                if key in items_out:
#                    continue
#                elif (key in items_in) or len(items_in) == 1:
#                    setattr(self, key, f[key][()])
#        # temporary:
#        if hasattr(self, 'location'):
#            self.latitude, self.longitude, self.depth =\
#                    self.location
#        # alias
#        self.tid = self.template_idx
#
#    def flatten_catalog(self, attributes=[], unique_event=False):
#        """Output a catalog with one row for each requested attribute.
#
#        Parameters
#        -----------
#        attributes: list of strings, default to an empty list
#            List of all the attributes, in addition to origin_times
#            and tids, that will be included in the flat catalog.
#        unique_event: boolean, default to False
#            If True, only returns the events flagged as unique.
#
#        Returns
#        ------------
#        flat_catalog: dictionary
#            Dictionary with one entry for each requested attribute.
#            Each entry contains an array of size n_events.
#        """
#        if not hasattr(self, 'origin_times'):
#            print('FamilyCatalog needs to have the origin_times attribute '
#                  'to return a flat catalog.')
#            return
#        n_events = len(self.origin_times)
#        flat_catalog = {}
#        flat_catalog['origin_times'] = np.asarray(self.origin_times)
#        flat_catalog['tids'] = np.ones(n_events, dtype=np.int32)*self.tid
#        for attr in attributes:
#            if not hasattr(self, attr):
#                print(f'FamilyCatalog does not have {attr}')
#                continue
#            attr_ = getattr(self, attr)
#            if not isinstance(attr_, list)\
#                    and not isinstance(attr_, np.ndarray):
#                flat_catalog[attr] = \
#                        np.ones(n_events, dtype=np.dtype(type(attr_)))*attr_
#            else:
#                #flat_catalog[attr] = np.asarray(attr_)[selection]
#                flat_catalog[attr] = np.asarray(attr_)
#                if flat_catalog[attr].shape[0] != n_events:
#                    # this condition should work for spotting
#                    # list-like attributes that are not of the
#                    # shape (n_events, n_attr)
#                    flat_catalog[attr] = np.repeat(flat_catalog[attr], n_events).\
#                            reshape((n_events,)+flat_catalog[attr].shape)
#                else:
#                    flat_catalog[attr] = flat_catalog[attr]
#        if unique_event:
#            selection = self.unique_event
#            for attr in flat_catalog.keys():
#                flat_catalog[attr] = flat_catalog[attr][selection]
#        return flat_catalog
#
#    def return_as_dic(self, attributes=[]):
#        catalog = {}
#        for attr in attributes:
#            if hasattr(self, attr):
#                catalog[attr] = getattr(self, attr)
#            else:
#                print(f'Template {self.tid} catalog has no attribute {attr}')
#        return catalog
#
# class FamilyGroupCatalog(object):
#
#    def __init__(self, families=None, filenames=None,
#                 db_path_M=None, db_path=cfg.INPUT_PATH):
#        """
#        Must either provide families (list of instances of Family),
#        or list of filenames.
#        """
#        if families is not None:
#            # index the single-template families by their tid
#            # in a dictionary
#            if isinstance(catalog.template_idx, np.ndarray):
#                self.families = {catalog.template_idx[0]: catalog
#                                 for catalog in families}
#            else:
#                self.families = {catalog.template_idx: catalog
#                                 for catalog in families}
#            self.tids = list(self.families.keys())
#        else:
#            self.db_path = db_path
#            self.db_path_M = db_path_M
#            self.filenames = filenames
#
#    def add_recurrence_times(self):
#        for tid in self.tids:
#            self.families[tid].recurrence_times =\
#                    np.hstack(([np.nan], np.diff(self.families[tid].origin_times)))
#
#    def read_data(self, items_in=[], items_out=[]):
#        """Attach the requested attributes to the Family instances.
#
#        Parameters
#        -----------
#        items_in: list of strings, default to an empty list
#            List of items to read from the catalog file.
#            If an empty list is provided, then all items are read.
#        items_out: list of strings, default to an empty list
#            List of items to reject when reading from the catalog file.
#            If an empty list is provided, no items are discarded.
#        """
#        if not isinstance(items_in, list):
#            items_in = [items_in]
#        if not isinstance(items_out, list):
#            items_out = [items_out]
#        # initialize the dictionary of Family instances
#        self.families = {}
#        for filename in self.filenames:
#            # initialize a Family instance
#            catalog = Family(filename, self.db_path_M, db_path=self.db_path)
#            # attach the requested attributes
#            catalog.read_data(items_in=items_in, items_out=items_out)
#            # fill the dictionary
#            if isinstance(catalog.template_idx, np.ndarray):
#                self.families[catalog.template_idx[0]] = catalog
#            else:
#                self.families[catalog.template_idx] = catalog
#        self.tids = list(self.families.keys())
#
#    def flatten_catalog(self, attributes=[], chronological_order=True,
#                        unique_event=False):
#        flat_catalogs = [self.families[tid].flatten_catalog(
#            attributes=attributes, unique_event=unique_event)
#            for tid in self.tids]
#        flat_agg_catalog = {}
#        for attr in flat_catalogs[0].keys():
#            flat_agg_catalog[attr] = \
#                    np.concatenate(
#                            [flat_cat[attr] for flat_cat in flat_catalogs],
#                             axis=0)
#        if chronological_order:
#            order = np.argsort(flat_agg_catalog['origin_times'])
#            for attr in flat_agg_catalog.keys():
#                flat_agg_catalog[attr] = flat_agg_catalog[attr][order]
#        return flat_agg_catalog
#
#    def remove_multiples(self, db_path_T, n_closest_stations=10,
#                         dt_criterion=3., distance_criterion=1.,
#                         similarity_criterion=-1., return_catalog=False):
#        """Search for events detected by multiple templates.
#
#        Parameters
#        -----------
#        db_path_T: string
#            Name of the directory where template files are stored.
#        n_closest_stations: integer, default to 10
#            In case template similarity is taken into account,
#            this is the number of stations closest to each template
#            that are used in the calculation of the average cc.
#        dt_criterion: float, default to 3
#            Time interval, in seconds, under which two events are
#            examined for redundancy.
#        distance_criterion: float, default to 1
#            Distance threshold, in kilometers, between two uncertainty
#            ellipsoids under which two events are examined for redundancy.
#        similarity_criterion: float, default to -1
#            Template similarity threshold, in terms of average CC, over
#            which two events are examined for redundancy. The default
#            value of -1 is always verified, meaning that similarity is
#            actually not taken into account.
#        return_catalog: boolean, default to False
#            If True, returns a flatten catalog.
#        """
#        self.db_path_T = db_path_T
#        catalog = self.flatten_catalog(
#                attributes=['latitude', 'longitude', 'depth',
#                            'correlation_coefficients'])
#        # define an alias for tids
#        catalog['template_ids'] = catalog['tids']
#        self.TpGroup = TemplateGroup(self.tids, db_path_T)
#        self.TpGroup.attach_ellipsoid_distances(substract_errors=True)
#        if similarity_criterion > -1.:
#            self.TpGroup.template_similarity(distance_threshold=distance_criterion,
#                                             n_stations=n_closest_stations)
#        # -----------------------------------
#        t1 = give_time()
#        print('Searching for events detected by multiple templates')
#        print('All events occurring within {:.1f} sec, with uncertainty '
#              'ellipsoids closer than {:.1f} km will and '
#              'inter-template CC larger than {:.2f} be considered the same'.
#              format(dt_criterion, distance_criterion, similarity_criterion))
#        n_events = len(catalog['origin_times'])
#        unique_event = np.ones(n_events, dtype=np.bool)
#        for n1 in range(n_events):
#            if not unique_event[n1]:
#                continue
#            tid1 = catalog['template_ids'][n1]
#            # apply the time criterion
#            dt_n1 = (catalog['origin_times'] - catalog['origin_times'][n1])
#            temporal_neighbors = (dt_n1 < dt_criterion) & (dt_n1 >= 0.)\
#                                & unique_event
#            # comment this line if you keep best CC
#            #temporal_neighbors[n1] = False
#            # get indices of where the above selection is True
#            candidates = np.where(temporal_neighbors)[0]
#            if len(candidates) == 0:
#                continue
#            # get template ids of all events that passed the time criterion
#            tids_candidates = np.int32([catalog['template_ids'][idx]
#                                       for idx in candidates])
#            # apply the spatial criterion to the distance between
#            # uncertainty ellipsoids
#            ellips_dist = self.TpGroup.ellipsoid_distances[tid1].\
#                    loc[tids_candidates].values
#            if similarity_criterion > -1.:
#                similarities = self.TpGroup.intertp_cc[tid1].\
#                        loc[tids_candidates].values
#                multiples = candidates[np.where(
#                    (ellips_dist < distance_criterion)\
#                   & (similarities >= similarity_criterion))[0]]
#            else:
#                multiples = candidates[np.where(
#                    ellips_dist < distance_criterion)[0]]
#            # comment this line if you keep best CC
#            #if len(multiples) == 0:
#            #    continue
#            # uncomment if you keep best CC
#            if len(multiples) == 1:
#                continue
#            else:
#                unique_event[multiples] = False
#                # find best CC and keep it
#                ccs = catalog['correlation_coefficients'][multiples]
#                best_cc = multiples[ccs.argmax()]
#                unique_event[best_cc] = True
#        t2 = give_time()
#        print('{:.2f}s to flag the multiples'.format(t2-t1))
#        # -------------------------------------------
#        catalog['unique_event'] = unique_event
#        for tid in self.tids:
#            selection = catalog['tids'] == tid
#            unique_event_t = catalog['unique_event'][selection]
#            self.families[tid].unique_event = unique_event_t
#        if return_catalog:
#            return catalog
#
# class FamilyEvents(object):
#
#    def __init__(self, tid, db_path_T, db_path_M, db_path=cfg.INPUT_PATH):
#        """
#        Initializes an FamilyEvents instance and attaches the
#        Template instance corresponding to this family.
#        """
#        self.tid = tid
#        self.db_path_T = db_path_T
#        self.db_path_M = db_path_M
#        self.db_path = db_path
#        self.template = Template(f'template{tid}', db_path_T, db_path=db_path)
#        self.template.read_waveforms()
#        self.sr = self.template.sampling_rate
#        self.stations = self.template.network_stations
#
#    def attach_catalog(self, items_in=[], items_out=[]):
#        """
#        Creates a Family instance and call Family.read_data()
#        """
#        filename = f'multiplets{self.tid}catalog.h5'
#        self.catalog = Family(filename, self.db_path_M, db_path=self.db_path)
#        self.catalog.read_data(items_in=items_in, items_out=items_out)
#
#    def check_template_reloc(self):
#        """
#        If templates were relocated, overwrite the catalog's location
#        with the template's relocated hypocenter
#        """
#        for attr in ['longitude', 'latitude', 'depth']:
#            if hasattr(self.template, 'relocated_'+attr):
#                setattr(self.catalog, attr, getattr(self.template, 'relocated_'+attr))
#
#    def find_closest_stations(self, n_stations, available_stations=None):
#        """
#        Here for consistency with FamilyGroupEvents and write the
#        cross_correlate method such that FamilyGroupEvents can inherit
#        from it.
#        """
#        self.template.n_closest_stations(
#                n_stations, available_stations=available_stations)
#        self.stations = self.template.stations
#        self.map_to_subnet = self.template.map_to_subnet
#
#    @property
#    def dtt_P(self, max_corr=1000.):
#        """ Travel-time corrections to individual events.
#
#        Parameters
#        -----------
#        max_corr: float, default to 1000.
#            Maximum correction time, in seconds, below which
#            a correction time is considered to be valid.
#        """
#        if hasattr(self, '_dtt_P'):
#            _dtt_P = self._dtt_P
#        else:
#            print('Call `get_tt_corrections` first!')
#            return None
#        if hasattr(self, 'event_ids'):
#            _dtt_P = _dtt_P[self.event_ids, :]
#        #if hasattr(self, 'map_to_subnet'):
#        #    _dtt_P = _dtt_P[:, self.map_to_subnet]
#        return _dtt_P
#
#    @property
#    def dtt_S(self, max_corr=1000.):
#        """ Travel-time corrections to individual events.
#
#        Parameters
#        -----------
#        max_corr: float, default to 1000.
#            Maximum correction time, in seconds, below which
#            a correction time is considered to be valid.
#        """
#        if hasattr(self, '_dtt_S'):
#            _dtt_S = self._dtt_S
#        else:
#            print('Call `get_tt_corrections` first!')
#            return None
#        if hasattr(self, 'event_ids'):
#            _dtt_S = _dtt_S[self.event_ids, :]
#        #if hasattr(self, 'map_to_subnet'):
#        #    _dtt_S = _dtt_S[:, self.map_to_subnet]
#        return _dtt_S
#
#    def get_tt_corrections(self, max_corr=1000.):
#        """ Read travel-time corrections to individual events.
#
#        Parameters
#        -----------
#        max_corr: float, default to 1000.
#            Maximum correction time, in seconds, below which
#            a correction time is considered to be valid.
#        """
#        cat_file = os.path.join(
#                            self.db_path, self.db_path_M,
#                            f'multiplets{self.tid}catalog.h5')
#        with h5.File(cat_file, mode='r') as f:
#            if 'dtt_P' in f:
#                self._dtt_P = f['dtt_P'][()]
#                self._dtt_P[self._dtt_P > max_corr] = 0.
#            else:
#                print(f'No P-wave travel-time corrections found in {cat_file}.')
#                self._dtt_P = np.zeros(
#                        (len(self.event_ids), len(self.template.network_stations)),
#                        dtype=np.float32)
#            if 'dtt_S' in f:
#                self._dtt_S = f['dtt_S'][()]
#                self._dtt_S[self._dtt_S > max_corr] = 0.
#            else:
#                print(f'No S-wave travel-time corrections found in {cat_file}.')
#                self._dtt_S = np.zeros(
#                        (len(self.event_ids), len(self.template.network_stations)),
#                        dtype=np.float32)
#
#    def read_data(self, **kwargs):
#        """Call 'fetch_detection_waveforms' from utils.
#
#        Read waveforms from the event waveforms that were
#        extracted at the time of detection.
#        """
#        # self.event_ids tell us in which order the events were read,
#        # which depends on the kwargs given to fetch_detection_waveforms
#        # force ordering to be chronological
#        kwargs['ordering'] = 'origin_times'
#        #kwargs['flip_order'] = True # why did I do that?? This seems totally unnecessary
#        if (kwargs.get('unique_event', False)\
#                and np.sum(self.catalog.unique_event) == 0):
#            self.detection_waveforms = []
#            self.event_ids, self.event_ids_str = [], []
#            self.n_events = 0
#        else:
#            self.detection_waveforms, _, self.event_ids = \
#                    utils.fetch_detection_waveforms(
#                            self.tid, self.db_path_T, self.db_path_M,
#                            return_event_ids=True, **kwargs)
#            self.n_events = self.detection_waveforms.shape[0]
#            self.event_ids_str = [f'{self.tid},{event_id}' for event_id in self.event_ids]
#
#    def trim_waveforms(self, duration, offset_start_S, offset_start_P,
#                       t0=cfg.BUFFER_EXTRACTED_EVENTS_SEC,
#                       S_window_time=4., P_window_time=1., correct_tt=False):
#        """Trim the waveforms using the P- and S-wave moveouts from the template.
#
#        Parameters
#        ------------
#        duration: scalar, float
#            Duration, in seconds, of the trimmed windows.
#        offset_start_S: scalar, float
#            Time, in seconds, taken BEFORE the S wave.
#        offset_start_P: scalar, float
#            Time, in seconds, taken BEFORE the P wave.
#        t0: float, optional
#            Time, in seconds, taken before the detection time
#            when the waveforms were extracted at the time
#            of detection. Default to the value written in
#            the parameter file.
#        S_window_time: scalar, float
#            Time between the beginning of the S-wave template
#            window and the predicted S-wave arrival time.
#        P_window_time: scalar, float
#            Time between the beginning of the P-wave template
#            widow and the predicted P-wave arrival time.
#        correct_tt: boolean, default to False
#            If True, use the individual phase picks -- if available -- to
#            adjust the template's travel times to individual events.
#        """
#        if not hasattr(self, 'detection_waveforms'):
#            print('Need to call read_data first.')
#            return
#        if self.n_events == 0:
#            print('No events were read, probably because only '
#                  'unique events were requested.')
#            return
#        # convert all times from seconds to samples
#        duration = utils.sec_to_samp(duration, sr=self.sr)
#        offset_start_S = utils.sec_to_samp(offset_start_S, sr=self.sr)
#        offset_start_P = utils.sec_to_samp(offset_start_P, sr=self.sr)
#        S_window_time = utils.sec_to_samp(S_window_time, sr=self.sr)
#        P_window_time = utils.sec_to_samp(P_window_time, sr=self.sr)
#        t0 = utils.sec_to_samp(t0, sr=self.sr)
#        new_shape = self.detection_waveforms.shape[:-1] + (duration,)
#        self.trimmed_waveforms = np.zeros(new_shape, dtype=np.float32)
#        _, n_stations, _, n_samples = self.detection_waveforms.shape
#        if correct_tt:
#            self.get_tt_corrections(max_corr=5.)
#        for s in range(n_stations):
#            if not correct_tt:
#                # P-wave window on vertical components
#                P_start = t0 + self.template.network_p_moveouts[s] + P_window_time - offset_start_P
#                P_end = P_start + duration
#                if P_start < n_samples:
#                    P_end = min(n_samples, P_end)
#                    self.trimmed_waveforms[:, s, 2, :P_end-P_start] = \
#                            self.detection_waveforms[:, s, 2, P_start:P_end]
#                # S-wave window on horizontal components
#                S_start = t0 + self.template.network_s_moveouts[s] + S_window_time - offset_start_S
#                S_end = S_start + duration
#                if S_start < n_samples:
#                    S_end = min(n_samples, S_end)
#                    self.trimmed_waveforms[:, s, :2, :S_end-S_start] = \
#                            self.detection_waveforms[:, s, :2, S_start:S_end]
#            else:
#                for n in range(self.n_events):
#                    # P-wave window on vertical components
#                    P_start = t0 + self.template.network_p_moveouts[s]\
#                            + P_window_time - offset_start_P\
#                            + utils.sec_to_samp(self.dtt_P[n, s], sr=self.sr)
#                    P_end = P_start + duration
#                    if P_start < n_samples:
#                        P_end = min(n_samples, P_end)
#                        self.trimmed_waveforms[n, s, 2, :P_end-P_start] = \
#                                self.detection_waveforms[n, s, 2, P_start:P_end]
#                    # S-wave window on horizontal components
#                    S_start = t0 + self.template.network_s_moveouts[s]\
#                            + S_window_time - offset_start_S\
#                            + utils.sec_to_samp(self.dtt_S[n, s], sr=self.sr)
#                    S_end = S_start + duration
#                    if S_start < n_samples:
#                        S_end = min(n_samples, S_end)
#                        self.trimmed_waveforms[n, s, :2, :S_end-S_start] = \
#                                self.detection_waveforms[n, s, :2, S_start:S_end]
#
#    def read_trimmed_waveforms(self, duration, offset_start, net, target_SR,
#                               tt_phases=['S', 'S', 'P'], norm_rms=True,
#                               buffer=2., unique_event=False, correct_tt=False,
#                               selection=None, **preprocess_kwargs):
#        """
#        Read waveforms from raw data and refilter/resample. Extra key-word
#        arguments will be passed to the preprocessing routine.
#
#        Parameters
#        ------------
#        duration: float
#            Duration, in seconds, of the extracted time windows.
#        offset_start: float
#            Time, in seconds, added to the requested time to define
#            the beginning of the time window. It should be negative if
#            the goal is to make the window start before the target phase.
#        net: `Network` object
#            `Network` instance.
#        tt_phases: list of strings, default to ['S', 'S', 'P']
#            Determine which phase is targetted on each component.
#        buffer: float, default to 2
#            Time, in seconds, taken at the beginning and end of the window.
#            It is used to make sure the preprocessing does not alter the
#            actual window.
#        unique_event: boolean, default to False
#            If True, only loads the unique detections.
#        correct_tt: boolean, default to False
#            If True, use the individual phase picks -- if available -- to
#            adjust the template's travel times to individual events.
#        selection: numpy array, default to None
#            Indexes of the events to use.
#        """
#        from . import event_extraction
#        if not hasattr(self, 'catalog'):
#            self.attach_catalog()
#        preprocess_kwargs['target_SR'] = target_SR
#        detection_waveforms  = []
#        # reshape travel times
#        station_indexes = np.int32([self.template.network_stations.tolist().index(sta)
#                for sta in net.stations])
#        # use travel times according to requested phase on each channel
#        phase_index = {'S': 1, 'P': 0}
#        tts = np.stack([self.template.network_travel_times\
#                [station_indexes, phase_index[tt_phases[c]]]
#            for c in range(len(tt_phases))], axis=1)
#        if selection is None:
#            if unique_event:
#                selection = self.catalog.unique_event
#            else:
#                selection = np.ones(len(self.catalog.origin_times), dtype=np.bool)
#        self.event_ids = np.arange(len(selection), dtype=np.int32)[selection]
#        if correct_tt:
#            self.get_tt_corrections(max_corr=5.)
#        print(f'Reading {np.sum(selection)} events...')
#        for n, evidx in enumerate(self.event_ids):
#            ot = self.catalog.origin_times[evidx]
#            if correct_tt:
#                tt_corrections = np.stack(
#                        [getattr(self, f'dtt_{ph}')[n, station_indexes] for ph in tt_phases], axis=1)
#            else:
#                tt_corrections = np.zeros_like(tts, dtype=np.float32)
#            event = event_extraction.extract_event_realigned(
#                    ot, net, tts+tt_corrections, duration=duration+2.*buffer,
#                    offset_start=offset_start-buffer, folder='raw')
#            event = event_extraction.preprocess_event(
#                    event, target_duration=duration+2.*buffer,
#                    **preprocess_kwargs)
#            if len(event) > 0:
#                detection_waveforms.append(utils.get_np_array(
#                    event, net.stations, components=net.components, verbose=False))
#            else:
#                detection_waveforms.append(np.zeros(
#                    len(net.stations), len(net.components),
#                    utils.sec_to_samp(target_duration, sr=target_SR),
#                    dtype=np.float32))
#        detection_waveforms = np.stack(detection_waveforms, axis=0)
#        if norm_rms:
#            # one normalization factor for each 3-comp seismogram
#            norm = np.std(detection_waveforms, axis=(2, 3))[..., np.newaxis, np.newaxis]
#            norm[norm == 0.] = 1.
#            detection_waveforms /= norm
#        # trim the waveforms
#        buffer = utils.sec_to_samp(buffer, sr=target_SR)
#        self.trimmed_waveforms = detection_waveforms[..., buffer:-buffer]
#        # update SR
#        self.sr = target_SR
#        # add metadata
#        self.n_events = self.trimmed_waveforms.shape[0]
#        self.event_ids_str = [f'{self.tid},{event_id}' for event_id in self.event_ids]
#
#    def read_trimmed_waveforms_raw(
#            self, duration, offset_start, net,
#            tt_phases=['S', 'S', 'P'],
#            buffer=2., unique_event=False, correct_tt=False,
#            selection=None, **preprocess_kwargs):
#        """
#        Read waveforms from raw data and remove instrument response if requested.
#        Extra key-word arguments will be passed to the preprocessing routine.
#        This should not be used to resample all traces to the same sampling
#        rate, instead use `read_trimmed_waveforms`.
#
#        Parameters
#        ------------
#        duration: float
#            Duration, in seconds, of the extracted time windows.
#        offset_start: float
#            Time, in seconds, added to the requested time to define
#            the beginning of the time window. It should be negative if
#            the goal is to make the window start before the target phase.
#        net: `Network` object
#            `Network` instance.
#        tt_phases: list of strings, default to ['S', 'S', 'P']
#            Determine which phase is targetted on each component.
#        buffer: float, default to 2
#            Time, in seconds, taken at the beginning and end of the window.
#            It is used to make sure the preprocessing does not alter the
#            actual window.
#        unique_event: boolean, default to False
#            If True, only loads the unique detections.
#        correct_tt: boolean, default to False
#            If True, use the individual phase picks -- if available -- to
#            adjust the template's travel times to individual events.
#        selection: numpy array, default to None
#            Indexes of the events to use.
#        """
#        from . import event_extraction
#        if not hasattr(self, 'catalog'):
#            self.attach_catalog()
#        detection_waveforms  = []
#        # reshape travel times
#        station_indexes = np.int32([self.template.network_stations.tolist().index(sta)
#                for sta in net.stations])
#        # use travel times according to requested phase on each channel
#        phase_index = {'S': 1, 'P': 0}
#        tts = np.stack([self.template.network_travel_times\
#                [station_indexes, phase_index[tt_phases[c]]]
#            for c in range(len(tt_phases))], axis=1)
#        if selection is None:
#            if unique_event:
#                selection = self.catalog.unique_event
#            else:
#                selection = np.ones(len(self.catalog.origin_times), dtype=np.bool)
#        self.event_ids = np.arange(len(selection), dtype=np.int32)[selection]
#        if correct_tt:
#            self.get_tt_corrections(max_corr=5.)
#        events = []
#        for n, evidx in enumerate(self.event_ids):
#            ot = self.catalog.origin_times[evidx]
#            if correct_tt:
#                tt_corrections = np.stack(
#                        [getattr(self, f'dtt_{ph}')[n, station_indexes] for ph in tt_phases], axis=1)
#            else:
#                tt_corrections = np.zeros_like(tts, dtype=np.float32)
#            event = event_extraction.extract_event_realigned(
#                    ot, net, tts+tt_corrections, duration=duration+2.*buffer,
#                    offset_start=offset_start-buffer, folder='raw',
#                    attach_response=True)
#            event = event_extraction.preprocess_event(
#                    event, target_duration=duration+2.*buffer,
#                    **preprocess_kwargs)
#            # now that the preprocessing is done, remove the sides
#            for tr in event:
#                tr.trim(starttime=tr.stats.starttime+buffer,
#                        endtime=tr.stats.endtime-buffer)
#            events.append(event)
#        self.trimmed_events = events
#        # add metadata
#        self.n_events = len(events)
#        self.event_ids_str = [f'{self.tid},{event_id}' for event_id in self.event_ids]
#        if self.n_events != np.sum(selection):
#            print('Number of extracted events does not match '
#                  f'expected number ({self.n_events} vs {np.sum(selection)}).')
#
#
#    def cross_correlate(self, n_stations=40, max_lag=10,
#                        paired=None, device='cpu', available_stations=None):
#        """
#        Parameters
#        -----------
#        n_stations: integer, default to 40
#            The number of stations closest to each template used in
#            the computation of the average CC.
#        max_lag: integer, default to 10
#            Maximum lag, in samples, allowed when searching for the
#            maximum CC on each channel. This is to account for small
#            discrepancies in windowing that could occur for two templates
#            highly similar but associated to slightly different locations.
#        paired: (n_events, n_events) boolean array, default to None
#            If not None, this array is used to determine for which
#            events the CC should be computed. This is mostly useful
#            when cross correlating large data sets, with potentially
#            many redundant events.
#        """
#        import fast_matched_filter as fmf
#        if not hasattr(self, 'trimmed_waveforms'):
#            print('The FamilyEvents instance needs the trimmed_waveforms '
#                  'attribute, see trim_waveforms.')
#            return
#        self.max_lag = max_lag
#        self.find_closest_stations(n_stations, available_stations=available_stations)
#        print('Finding the best inter-event CCs...')
#        # format arrays for FMF
#        slice_ = np.index_exp[:, self.map_to_subnet, :, :]
#        data_arr = self.trimmed_waveforms.copy()[slice_]
#        norm_data = np.std(data_arr, axis=-1)[..., np.newaxis]
#        norm_data[norm_data == 0.] = 1.
#        data_arr /= norm_data
#        template_arr = data_arr[..., max_lag:-max_lag]
#        n_stations, n_components = data_arr.shape[1:-1]
#        # initialize ouputs
#        if paired is None:
#            paired = np.ones((self.n_events, self.n_events), dtype=np.bool)
#            self.paired = paired
#        output_shape = (np.sum(paired), n_stations)
#        CCs_S = np.zeros(output_shape, dtype=np.float32)
#        lags_S = np.zeros(output_shape, dtype=np.int32)
#        CCs_P = np.zeros(output_shape, dtype=np.float32)
#        lags_P = np.zeros(output_shape, dtype=np.int32)
#        # re-arrange input arrays to pass pieces of array
#        data_arr_P = np.ascontiguousarray(data_arr[:, :, 2:3, :])
#        template_arr_P = np.ascontiguousarray(template_arr[:, :, 2:3, :])
#        moveouts_arr_P = np.zeros((self.n_events, 1, 1), dtype=np.int32)
#        weights_arr_P = np.ones((self.n_events, 1, 1), dtype=np.float32)
#        data_arr_S = np.ascontiguousarray(data_arr[:, :, :2, :])
#        template_arr_S = np.ascontiguousarray(template_arr[:, :, :2, :])
#        moveouts_arr_S = np.zeros((self.n_events, 1, 2), dtype=np.int32)
#        weights_arr_S = 0.5*np.ones((self.n_events, 1, 2), dtype=np.float32)
#        # free some space
#        del data_arr, template_arr
#        counter = 0
#        for n in range(self.n_events):
#            print(f'------ {n+1}/{self.n_events} -------')
#            selection = paired[n, :]
#            counter_inc = np.sum(selection)
#            for s in range(n_stations):
#                # use trick to keep station and component dim
#                slice_ = np.index_exp[selection, s:s+1, :, :]
#                cc_S = fmf.matched_filter(
#                        template_arr_S[slice_], moveouts_arr_S[selection, ...],
#                        weights_arr_S[selection, ...], data_arr_S[(n,)+slice_[1:]],
#                        1, check_zeros=False, arch=device)
#                cc_P = fmf.matched_filter(
#                        template_arr_P[slice_], moveouts_arr_P[selection, ...],
#                        weights_arr_P[selection, ...], data_arr_P[(n,)+slice_[1:]],
#                        1, check_zeros=False, arch=device)
#                # get best CC and its lag
#                CCs_S[counter:counter+counter_inc, s] = np.max(cc_S, axis=-1)
#                lags_S[counter:counter+counter_inc, s] = np.argmax(cc_S, axis=-1) - max_lag
#                CCs_P[counter:counter+counter_inc, s] = np.max(cc_P, axis=-1)
#                lags_P[counter:counter+counter_inc, s] = np.argmax(cc_P, axis=-1) - max_lag
#                # N.B: lags[n1, n2] is the ev1-ev2 time
#            counter += counter_inc
#        self.CCs_S = CCs_S
#        self.lags_S = lags_S
#        self.CCs_P = CCs_P
#        self.lags_P = lags_P
#        self.paired = paired
#        self.max_lag = max_lag
#
#    def plot_alignment(self, pair_id, s,
#                       components = ['N', 'E', 'Z']):
#        """
#        Check visually what the max correlation alignment is worth.
#        This also demonstrates that we use the following convention:
#        Argmax(CC(pair[i, j])) == tt_j - tt_i
#        i.e. tt_2 - tt_1 in GrowClust
#        """
#        import matplotlib.pyplot as plt
#        if not hasattr(self, 'CCs_S'):
#            print('Need to run self.cross_correlate first!')
#            return
#        pairs = np.column_stack(np.where(self.paired))
#        evid1, evid2 = pairs[pair_id]
#        ss = self.map_to_subnet[s]
#        fig, axes = plt.subplots(
#                num=f'ev{evid1}_ev{evid2}_station_{s}',
#                nrows=3, ncols=1, figsize=(18, 9))
#        time = np.arange(self.trimmed_waveforms.shape[-1], dtype=np.float32)
#        for c in range(len(components)):
#            phase = 'S' if c < 2 else 'P'
#            axes[c].plot(
#                    time, utils.max_norm(self.trimmed_waveforms[evid1, ss, c, :]),
#                    color='k', label=f'Ev. {evid1}: {components[c]} cp. - {phase} wave')
#            if phase == 'S':
#                mv = self.max_lag + self.lags_S[pair_id, s]
#                CC = self.CCs_S[pair_id, s]
#            else:
#                mv = self.max_lag + self.lags_P[pair_id, s]
#                CC = self.CCs_P[pair_id, s]
#            axes[c].plot(
#                    time[:-2*self.max_lag] + mv,
#                    utils.max_norm(self.trimmed_waveforms[evid2, ss, c, self.max_lag:-self.max_lag]),
#                    color='C3', label=f'Ev. {evid2}: {components[c]} cp. - {phase} wave'
#                    f'\nCC={CC:.2f}\nLag: {mv-self.max_lag}sp')
#            axes[c].axvline(self.max_lag, color='k')
#            axes[c].legend(loc='upper left')
#            axes[c].set_ylabel('Normalized Amp.')
#            axes[c].set_xlabel('Time (samples)')
#        plt.subplots_adjust(hspace=0.3)
#        return fig
#
#
#    # -------------------------------------------
#    #       GrowClust related methods
#    # -------------------------------------------
#    def read_GrowClust_output(self, filename, path):
#        print('Reading GrowClust output from {}'.
#                format(os.path.join(path, filename)))
#        ot_, lon_, lat_, dep_, err_h_, err_v_, err_t_ = \
#                [], [], [], [], [], [], []
#        with open(os.path.join(path, filename), 'r') as f:
#            for line in f.readlines():
#                line = line.split()
#                year, month, day, hour, minu, sec = line[:6]
#                # correct date if necessary
#                if int(day) == 0:
#                    date_ = udt(f'{year}-{month}-01')
#                    date_ -= datetime.timedelta(days=1)
#                    year, month, day = date_.year, date_.month, date_.day
#                # correct seconds if necessary
#                sec = float(sec)
#                if sec == 60.:
#                    sec -= 0.001
#                ot_.append(udt(f'{year}-{month}-{day}T{hour}:{minu}:{sec}').timestamp)
#                event_id = int(line[6])
#                latitude, longitude, depth = list(map(float, line[7:10]))
#                lon_.append(longitude)
#                lat_.append(latitude)
#                dep_.append(depth)
#                mag = float(line[10])
#                q_id, cl_id, cluster_pop = list(map(int, line[11:14]))
#                n_pairs, n_P_dt, n_S_dt = list(map(int, line[14:17]))
#                rms_P, rms_S = list(map(float, line[17:19]))
#                err_h, err_v, err_t = list(map(float, line[19:22])) # errors in km and sec
#                err_h_.append(err_h)
#                err_v_.append(err_v)
#                err_t_.append(err_t)
#                latitude_init, longitude_init, depth_init =\
#                        list(map(float, line[22:25]))
#        cat = {}
#        cat['origin_time'] = ot_
#        cat['longitude'] = lon_
#        cat['latitude'] = lat_
#        cat['depth'] = dep_
#        cat['error_hor'] = err_h_
#        cat['error_ver'] = err_v_
#        cat['error_t'] = err_t_
#        self.relocated_catalog =\
#                pd.DataFrame(data=cat)
#
#    def write_GrowClust_stationlist(self, filename, path,
#                                    network_filename='all_stations.in'):
#        """
#        This routine assumes that cross_correlate was called
#        shortly before and that self.template still has the same
#        set of stations as the ones used for the inter-event CCs.
#        """
#        net = Network(network_filename)
#        net.read()
#        subnet = net.subset(
#                self.stations, net.components, method='keep')
#        with open(os.path.join(path, filename), 'w') as f:
#            for s in range(len(subnet.stations)):
#                f.write('{:<5}\t{:.6f}\t{:.6f}\t{:.3f}\n'.
#                        format(subnet.stations[s], subnet.latitude[s],
#                               subnet.longitude[s], -1000.*subnet.depth[s]))
#
#    def write_GrowClust_eventlist(self, filename, path):
#        from obspy.core import UTCDateTime as udt
#        if not hasattr(self, 'catalog'):
#            self.attach_catalog()
#        with open(os.path.join(path, filename), 'w') as f:
#            for n in range(self.n_events):
#                nn = self.event_ids[n]
#                ot = udt(self.catalog.origin_times[nn])
#                if hasattr(self.catalog, relocated_catalog):
#                    # start from the relocated catalog
#                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
#                            format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
#                                   ot.second, self.catalog.relocated_latitude[nn],
#                                   self.catalog.relocated_longitude[nn],
#                                   self.catalog.relocated_depth[nn],
#                                   self.catalog.magnitudes[nn], self.event_ids[n]))
#                else:
#                    # all events are given the template location
#                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
#                            format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
#                                   ot.second, self.template.latitude,
#                                   self.template.longitude, self.template.depth,
#                                   self.catalog.magnitudes[nn], self.event_ids[n]))
#
#    def write_GrowClust_CC(self, filename, path, CC_threshold=0.):
#        if not hasattr(self, 'CCs_S'):
#            print('Need to compute the inter-event CCs first.')
#            return
#        with open(os.path.join(path, filename), 'w') as f:
#            n1, n2 = np.meshgrid(np.arange(self.n_events, dtype=np.int32),
#                                 np.arange(self.n_events, dtype=np.int32),
#                                 indexing='ij')
#            n1  = n1.flatten()[self.paired.flatten()]
#            n2  = n2.flatten()[self.paired.flatten()]
#            for n in range(len(n1)):
#                f.write('#\t{}\t{}\t0.0\n'.
#                        format(self.event_ids[n1[n]], self.event_ids[n2[n]]))
#                for s in range(len(self.stations)):
#                    # CCs that are zero are pairs that have to be skipped
#                    if self.CCs_S[n, s] > CC_threshold:
#                        f.write('  {:>5} {} {:.4f} S\n'.
#                                format(self.stations[s],
#                                       self.lags_S[n, s]/self.sr,
#                                       self.CCs_S[n, s]))
#                    if self.CCs_P[n, s] > CC_threshold:
#                        f.write('  {:>5} {} {:.4f} P\n'.
#                                format(self.stations[s],
#                                       self.lags_P[n, s]/self.sr,
#                                       self.CCs_P[n, s]))
#
# class FamilyGroupEvents(FamilyEvents):
#
#    def __init__(self, tids, db_path_T, db_path_M, db_path=cfg.INPUT_PATH):
#        self.tids = tids
#        self.n_templates = len(tids)
#        self.db_path_T = db_path_T
#        self.db_path_M = db_path_M
#        self.db_path = db_path
#        self.families = {tid: FamilyEvents(tid, db_path_T, db_path_M, db_path=db_path)
#                         for tid in tids}
#        self.sr = self.families[self.tids[0]].sr
#
#    def check_template_reloc(self):
#        """
#        If templates were relocated, overwrite the catalog's location
#        with the template's relocated hypocenter
#        """
#        if not hasattr(self, 'aggcat'):
#            print('This FamilyGroupEvents instance has no FamilyGroupCatalog. '
#                  'Do nothing.')
#        else:
#            for tid in self.tids:
#                template = self.families[tid].template
#                for attr in ['longitude', 'latitude', 'depth']:
#                    if hasattr(template, 'relocated_'+attr):
#                        setattr(self.aggcat.catalogs[tid], attr, getattr(template, 'relocated_'+attr))
#
#    def attach_catalog(self, dt_criterion=3., distance_criterion=5.,
#                       similarity_criterion=0.33, n_closest_stations=10,
#                       items_in=[]):
#        """
#        Creates an AggregatedCatalogs instance and
#        call AggregatedCatalogs.read_data(), as well as
#        AggregatedCatalogs.remove_multiples(). This is crucial to note
#        that remove_multiples returns a catalog that is ordered in time.
#        When reading data from the single-template families, these are
#        not ordered in time after concatenation.
#        """
#        # force 'origin_times' and 'magnitudes' to be among the items
#        items_in = items_in + ['origin_times', 'magnitudes',
#                    'correlation_coefficients', 'unique_event']
#        filenames = [f'multiplets{tid}catalog.h5' for tid in self.tids]
#        self.aggcat = AggregatedCatalogs(
#                filenames=filenames, db_path_M=self.db_path_M, db_path=self.db_path)
#        self.aggcat.read_data(items_in=items_in + ['location'])
#        self.check_template_reloc()
#        self.catalog = self.aggcat.remove_multiples(
#                self.db_path_T, dt_criterion=dt_criterion,
#                distance_criterion=distance_criterion,
#                similarity_criterion=similarity_criterion,
#                n_closest_stations=n_closest_stations, return_catalog=True)
#        # load the rest of the requested attributes
#        rest = [item for item in items_in if item not in self.catalog.keys()]
#        self.catalog.update(self.aggcat.flatten_catalog(
#            attributes=rest, chronological_order=True))
#
#    def find_closest_stations(self, n_stations, available_stations=None):
#        # overriden method from parent class
#        stations = []
#        for tid in self.tids:
#            self.families[tid].template.n_closest_stations(
#                    n_stations, available_stations=available_stations)
#            stations.extend(self.families[tid].template.stations)
#        stations, counts = np.unique(stations, return_counts=True)
#        sorted_ind = np.argsort(counts)[::-1]
#        self.stations = stations[sorted_ind[:n_stations]]
#        network_stations = self.families[self.tids[0]].template.network_stations
#        self.map_to_subnet = np.int32([np.where(network_stations == sta)[0]
#                                       for sta in self.stations]).squeeze()
#        # update all subfamilies
#        for tid in self.tids:
#            self.families[tid].stations = self.stations
#            self.families[tid].map_to_subnet = self.map_to_subnet
#
#    def pair_events(self, random_pairing_frac=0., random_max=2):
#        if not hasattr(self, 'catalog'):
#            # call attach_catalog() for calling remove_multiples()
#            # !!! the catalog that is returned like that takes all
#            # single-template families and order them in time, so
#            # it is ESSENTIAL to make sure that the data are ordered
#            # the same way
#            self.attach_catalog()
#        # all valid events are the unique events and also
#        # the events with highest CC from each family, to make
#        # sure all families will end up being paired by at least one event
#        valid_events = self.catalog['unique_event'].copy()
#        highest_CC_events_idx = {}
#        for tid in self.tids:
#            selection = self.catalog['tids'] == tid
#            highest_CC_events_idx[tid] = np.where(
#                self.catalog['correlation_coefficients']
#                == self.catalog['correlation_coefficients'][selection].max())[0][0]
#            valid_events[highest_CC_events_idx[tid]] = True
#        self.paired = np.zeros((self.n_events, self.n_events), dtype=np.bool)
#        for n in range(self.n_events):
#            if valid_events[n]:
#                # if this is a valid event, pair it with
#                # all other valid events
#                self.paired[n, valid_events] = True
#            # in all cases, pair the events with all other
#            # events of the same family
#            tid = self.catalog['tids'][n]
#            self.paired[n, self.catalog['tids'] == tid] = True
#            # OR --------------------
#            # link non-unique events only to their best CC event
#            #self.paired[n, highest_CC_events_idx[tid]] = True
#            # -------------------
#            # and add a few randomly selected connections
#            unpaired = np.where(~self.paired[n, :])[0]
#            n_random = min(random_max, int(random_pairing_frac*len(unpaired)))
#            if n_random > 0:
#                if len(unpaired) > 0:
#                    random_choice = np.random.choice(
#                            unpaired, size=n_random, replace=False)
#                    self.paired[n, random_choice] = True
#        np.fill_diagonal(self.paired, False)
#
#    def read_data(self, **kwargs):
#        # overriden method from parent class
#        for tid in self.tids:
#            if hasattr(self, 'aggcat'):
#                # use the already loaded, and potentially edited, catalog
#                kwargs['catalog'] = self.aggcat.families[tid]
#            self.families[tid].read_data(**kwargs)
#        self.event_ids_str = \
#                np.asarray([event_id for tid in self.tids
#                           for event_id in self.families[tid].event_ids_str])
#        self.n_events = len(self.event_ids_str)
#
#    def read_trimmed_waveforms(self, *args, **kwargs):
#        """
#        See FamilyEvents.read_trimmed_waveforms
#        """
#        # overriden method from parent class
#        for tid in self.tids:
#            self.families[tid].read_trimmed_waveforms(
#                    *args, **kwargs)
#        # agglomerate all waveforms into one array
#        self.trimmed_waveforms = np.concatenate(
#                [self.families[tid].trimmed_waveforms for tid in self.tids],
#                axis=0)
#        # add metadata
#        self.event_ids_str = \
#                np.asarray([event_id for tid in self.tids
#                           for event_id in self.families[tid].event_ids_str])
#        self.sr = self.families[self.tids[0]].sr
#        # reorder in chronological order
#        OT = []
#        for tid in self.tids:
#            #self.families[tid].attach_catalog(items_in=['origin_times'])
#            OT.extend(self.families[tid].catalog
#                        .origin_times[self.families[tid].event_ids])
#        OT = np.float64(OT) # and now these are the correct origin times
#        sorted_ind = np.argsort(OT)
#        self.trimmed_waveforms = self.trimmed_waveforms[sorted_ind, ...]
#        self.event_ids_str = self.event_ids_str[sorted_ind]
#        self.n_events = len(self.event_ids_str)
#        self.event_ids = np.arange(self.n_events)
#
#    def read_trimmed_waveforms_raw(self, *args, **kwargs):
#        """
#        See FamilyEvents.read_trimmed_waveforms_raw
#        """
#        # overriden method from parent class
#        for tid in self.tids:
#            self.families[tid].read_trimmed_waveforms_raw(
#                    *args, **kwargs)
#        # agglomerate all trimmed events into ine list
#        self.trimmed_events = sum(
#                [self.families[tid].trimmed_events for tid in self.tids], [])
#        # add metadata
#        self.event_ids_str = \
#                np.asarray([event_id for tid in self.tids
#                           for event_id in self.families[tid].event_ids_str])
#        # reorder in chronological order
#        OT = []
#        for tid in self.tids:
#            #self.families[tid].attach_catalog(items_in=['origin_times'])
#            OT.extend(self.families[tid].catalog
#                        .origin_times[self.families[tid].event_ids])
#        OT = np.float64(OT) # and now these are the correct origin times
#        sorted_ind = np.argsort(OT)
#        self.trimmed_events = [self.trimmed_events[i] for i in sorted_ind]
#        self.event_ids_str = self.event_ids_str[sorted_ind]
#        self.n_events = len(self.event_ids_str)
#        self.event_ids = np.arange(self.n_events)
#
#    def trim_waveforms(self, *args, **kwargs):
#        """
#        See FamilyEvents.trim_waveforms
#        """
#        # overriden method from parent class
#        for tid in self.tids:
#           self.families[tid].trim_waveforms(
#                   *args, **kwargs)
#        # agglomerate all waveforms into one array
#        self.trimmed_waveforms = np.concatenate(
#                [self.families[tid].trimmed_waveforms for tid in self.tids
#                    if self.families[tid].n_events > 0], axis=0)
#        # reorder in chronological order
#        #self.attach_catalog()
#        # no this is wrong!!! attach_catalog() call remove_multiples()
#        # which already merges single-template families and order them
#        # in time.... so to get the actual origin times that correspond
#        # to the data that were loaded, we need to read from the single-
#        # template catalog!!!
#        #sorted_ind = np.argsort(self.catalog['origin_times'])
#        OT = []
#        for tid in self.tids:
#            self.families[tid].attach_catalog(items_in=['origin_times'])
#            OT.extend(self.families[tid].catalog
#                        .origin_times[self.families[tid].event_ids])
#        OT = np.float64(OT) # and now these are the correct origin times
#        sorted_ind = np.argsort(OT)
#        self.trimmed_waveforms = self.trimmed_waveforms[sorted_ind, ...]
#        self.event_ids_str = self.event_ids_str[sorted_ind]
#        #for attr in self.catalog.keys():
#        #    self.catalog[attr] = self.catalog[attr][sorted_ind]
#        self.event_ids = np.arange(self.n_events)
#
#    # -------------------------------------------
#    #       GrowClust related methods
#    # -------------------------------------------
#    def read_GrowClust_output(self, filename_out, filename_evid,
#                              path_out, path_evid):
#        print('Reading GrowClust output from {}'.
#                format(os.path.join(path_out, filename_out)))
#        print('Reading event ids from {}'.
#                format(os.path.join(path_evid, filename_evid)))
#        event_ids_map = pd.read_csv(
#                os.path.join(path_evid, filename_evid), index_col=0)
#        ot_, lon_, lat_, dep_, err_h_, err_v_, err_t_, evids_ = \
#                [], [], [], [], [], [], [], []
#        with open(os.path.join(path_out, filename_out), 'r') as f:
#            for line in f.readlines():
#                line = line.split()
#                year, month, day, hour, minu, sec = line[:6]
#                # correct date if necessary
#                if int(day) == 0:
#                    date_ = udt(f'{year}-{month}-01')
#                    date_ -= datetime.timedelta(days=1)
#                    year, month, day = date_.year, date_.month, date_.day
#                # correct seconds if necessary
#                sec = float(sec)
#                if sec == 60.:
#                    sec -= 0.001
#                ot_.append(udt(f'{year}-{month}-{day}T{hour}:{minu}:{sec}').timestamp)
#                event_id = int(line[6])
#                evids_.append(event_id)
#                latitude, longitude, depth = list(map(float, line[7:10]))
#                lon_.append(longitude)
#                lat_.append(latitude)
#                dep_.append(depth)
#                mag = float(line[10])
#                q_id, cl_id, cluster_pop = list(map(int, line[11:14]))
#                n_pairs, n_P_dt, n_S_dt = list(map(int, line[14:17]))
#                rms_P, rms_S = list(map(float, line[17:19]))
#                err_h, err_v, err_t = list(map(float, line[19:22])) # errors in km and sec
#                err_h_.append(err_h)
#                err_v_.append(err_v)
#                err_t_.append(err_t)
#                latitude_init, longitude_init, depth_init =\
#                        list(map(float, line[22:25]))
#        # convert all lists to np arrays
#        ot_ = np.float64(ot_)
#        lon_, lat_, dep_ = np.float32(lon_), np.float32(lat_), np.float32(dep_)
#        err_h_, err_v_, err_t_ = \
#                np.float32(err_h_), np.float32(err_v_), np.float32(err_t_)
#        evids_ = np.int32(evids_)
#        # distribute results over corresponding templates
#        for tid in self.tids:
#            selection = event_ids_map.index == tid
#            event_ids_tid = event_ids_map['event_ids'][selection]
#            cat = {}
#            cat['origin_time'] = ot_[selection]
#            cat['longitude'] = lon_[selection]
#            cat['latitude'] = lat_[selection]
#            cat['depth'] = dep_[selection]
#            cat['error_hor'] = err_h_[selection]
#            cat['error_ver'] = err_v_[selection]
#            cat['error_t'] = err_t_[selection]
#            cat['event_ids'] = event_ids_tid
#            self.families[tid].relocated_catalog =\
#                    pd.DataFrame(data=cat)
#        # attach flattened version for convenience
#        cat = {}
#        cat['origin_time'] = ot_
#        cat['longitude'] = lon_
#        cat['latitude'] = lat_
#        cat['depth'] = dep_
#        cat['error_hor'] = err_h_
#        cat['error_ver'] = err_v_
#        cat['error_t'] = err_t_
#        cat['event_ids'] = event_ids_map['event_ids']
#        cat['tids'] = event_ids_map.index
#        self.relocated_catalog =\
#                pd.DataFrame(data=cat)
#        self.n_events = len(self.relocated_catalog)
#        self.event_ids = np.arange(self.n_events, dtype=np.int32)
#
#    def write_GrowClust_eventlist(self, filename, path, fresh_start=True):
#        """
#        Note: The different with its single family counterpart is that
#        here the catalog is already in the same order as self.event_ids
#        (cf. trim_waveforms)
#        """
#        from obspy.core import UTCDateTime as udt
#        if not hasattr(self, 'catalog'):
#            self.attach_catalog()
#        if hasattr(self, 'relocated_catalog') and not fresh_start:
#            print('Give locations from relocated catalog')
#        with open(os.path.join(path, filename), 'w') as f:
#            for n in range(self.n_events):
#                ot = udt(self.catalog['origin_times'][n])
#                if hasattr(self, 'relocated_catalog') and not fresh_start:
#                    # start from the relocated catalog
#                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
#                            format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
#                                   ot.second, self.relocated_catalog['latitude'].iloc[n],
#                                   self.relocated_catalog['longitude'].iloc[n],
#                                   self.relocated_catalog['depth'].iloc[n],
#                                   self.catalog['magnitudes'][n], self.event_ids[n]))
#                else:
#                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
#                            format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
#                                   ot.second, self.catalog['latitude'][n],
#                                   self.catalog['longitude'][n], self.catalog['depth'][n],
#                                   self.catalog['magnitudes'][n], self.event_ids[n]))
#
#    def write_GrowClust_eventids(self, filename, path):
#        """
#        For each flat integer event id used by GrowClust, we write
#        the corresponding template id and event id so that we can
#        link the relocated events to their template/detection.
#        """
#        with open(os.path.join(path, filename), 'w') as f:
#            # write column names
#            f.write('tids,event_ids\n')
#            for n in range(self.n_events):
#                f.write(f'{self.event_ids_str[n]}\n')
#                #sep = self.event_ids_str[n].find('.')
#                #f.write('{:d},{:d}\n'.format(
#                #    int(self.event_ids_str[n][:sep]),
#                #    int(self.event_ids_str[n][sep+1:])))
#
### --------------------------------------------------
#
#
#
##class TemplateGroup(object):
##    """A TemplateGroup class to handle groups of templates.
##
##
##    """
##
##    def __init__(self, tids, db_path_T, db_path=cfg.INPUT_PATH):
##        """Read the templates' data and metadata.
##
##        Parameters
##        -----------
##        tids: list or nd.array
##            List of template ids in the group.
##        db_path_T: string
##            Name of the folder with template files.
##        db_path: string, default to cfg.INPUT_PATH
##            Name of the folder with output files.
##        """
##        self.templates = []
##        self.tids = tids
##        self.n_templates = len(tids)
##        self.db_path_T = db_path_T
##        self.db_path = db_path
##        self.tids_map = {}
##        for t, tid in enumerate(tids):
##            self.templates.append(
##                    Template(f'template{tid}', db_path_T, db_path=db_path))
##            self.tids_map[tid] = t
##
##    def attach_intertp_distances(self):
##        """Compute distance between all template pairs.
##
##        """
##        print('Computing the inter-template distances...')
##        self.intertp_distances = intertp_distances_(
##                templates=self.templates, return_as_pd=True)
##
##    def attach_directional_errors(self):
##        """Compute the length of the uncertainty ellipsoid
##        inter-template direction.
##
##        New Attributes
##        ----------
##        directional_errors: (n_templates, n_templates) pandas DataFrame
##            The length, in kilometers, of the uncertainty ellipsoid in the
##            inter-template direction.
##            Example: self.directional_errors.loc[tid1, tid2] is the width of
##            template tid1's uncertainty ellipsoid in the direction of
##            template tid2.
##        """
##        print('Computing the inter-template directional errors...')
##        from cartopy import crs
##        # ----------------------------------------------
##        #      Define the projection used to
##        #      work in a cartesian space
##        # ----------------------------------------------
##        data_coords = crs.PlateCarree()
##        longitudes = np.float32([self.templates[i].longitude for i in range(self.n_templates)])
##        latitudes = np.float32([self.templates[i].latitude for i in range(self.n_templates)])
##        depths = np.float32([self.templates[i].depth for i in range(self.n_templates)])
##        projection = crs.Mercator(central_longitude=np.mean(longitudes),
##                                  min_latitude=latitudes.min(),
##                                  max_latitude=latitudes.max())
##        XY = projection.transform_points(data_coords, longitudes, latitudes)
##        cartesian_coords = np.stack([XY[:, 0], XY[:, 1], depths], axis=1)
##        # compute the directional errors
##        dir_errors = np.zeros((self.n_templates, self.n_templates), dtype=np.float32)
##        for t in range(self.n_templates):
##            unit_direction = cartesian_coords - cartesian_coords[t, :]
##            unit_direction /= np.sqrt(np.sum(unit_direction**2, axis=1))[:, np.newaxis]
##            # this operation produced NaNs for i=t
##            unit_direction[np.isnan(unit_direction)] = 0.
##            # compute the length of the covariance ellipsoid
##            # in the direction that links the two earthquakes
##            cov_dir = np.abs(np.sum(
##                self.templates[t].cov_mat.dot(unit_direction.T)*unit_direction.T, axis=0))
##            # covariance is unit of [distance**2], therefore we need the sqrt:
##            dir_errors[t, :] = np.sqrt(cov_dir)
##        # format it in a pandas DataFrame
##        dir_errors = pd.DataFrame(columns=[tid for tid in self.tids],
##                                  index=[tid for tid in self.tids],
##                                  data=dir_errors)
##        self.directional_errors = dir_errors
##
##
##    def attach_ellipsoid_distances(self, substract_errors=True):
##        """
##        Combine inter-template distances and directional errors
##        to compute the minimum inter-uncertainty ellipsoid distances.
##        This quantity can be negative if the ellipsoids overlap.
##        """
##        import pandas as pd
##        if not hasattr(self, 'intertp_distances'):
##            self.attach_intertp_distances()
##        if not hasattr(self, 'directional_errors'):
##            self.attach_directional_errors()
##        if substract_errors:
##            ellipsoid_distances = self.intertp_distances.values\
##                                - self.directional_errors.values\
##                                - self.directional_errors.values.T
##        else:
##            ellipsoid_distances = self.intertp_distances.values
##        self.ellipsoid_distances = pd.DataFrame(
##                columns=[tid for tid in self.tids],
##                index=[tid for tid in self.tids],
##                data=ellipsoid_distances)
##
##    def read_stacks(self, **SVDWF_kwargs):
##        self.db_path_M = SVDWF_kwargs.get('db_path_M', 'none')
##        self.stacks = []
##        for t, tid in enumerate(self.tids):
##            stack = utils.SVDWF_multiplets(
##                    tid, db_path=self.db_path,
##                    db_path_T=self.db_path_T,
##                    **SVDWF_kwargs)
##            self.stacks.append(stack)
##
##    def read_waveforms(self):
##        for template in self.templates:
##            template.read_waveforms()
##
##    def plot_cc(self, cmap='inferno'):
##        if not hasattr(self, 'intertp_cc'):
##            print('Should call self.template_similarity first')
##            return
##        import matplotlib.pyplot as plt
##        fig = plt.figure('template_similarity', figsize=(18, 9))
##        ax = fig.add_subplot(111)
##        tids1_g, tids2_g = np.meshgrid(self.tids, self.tids, indexing='ij')
##        pc = ax.pcolormesh(tids1_g, tids2_g, self.intertp_cc.values, cmap=cmap)
##        ax.set_xlabel('Template id')
##        ax.set_ylabel('Template id')
##        fig.colorbar(pc, label='Correlation Coefficient')
##        return fig
##
##    def template_similarity(self, distance_threshold=5.,
##                            n_stations=10, max_lag=10,
##                            device='cpu'):
##        """
##        Parameters
##        -----------
##        distance_threshold: float, default to 5
##            The distance threshold, in kilometers, between two
##            uncertainty ellipsoids under which similarity is computed.
##        n_stations: integer, default to 10
##            The number of stations closest to each template used in
##            the computation of the average CC.
##        max_lag: integer, default to 10
##            Maximum lag, in samples, allowed when searching for the
##            maximum CC on each channel. This is to account for small
##            discrepancies in windowing that could occur for two templates
##            highly similar but associated to slightly different locations.
##        """
##        import pandas as pd
##        import fast_matched_filter as fmf
##        if not hasattr(self, 'ellipsoid_distances'):
##            self.attach_ellipsoid_distances()
##        for template in self.templates:
##            template.read_waveforms()
##            template.n_closest_stations(n_stations)
##        print('Computing the similarity matrix...')
##        # format arrays for FMF
##        tp_array = np.stack([tp.network_waveforms for tp in self.templates],
##                             axis=0)
##        data = tp_array.copy()
##        tp_array = tp_array[..., max_lag:-max_lag]
##        moveouts = np.zeros(tp_array.shape[:-1], dtype=np.int32)
##        intertp_cc = np.zeros((self.n_templates, self.n_templates),
##                              dtype=np.float32)
##        n_stations, n_components = moveouts.shape[1:]
##        # use FMF on one template at a time against all others
##        for t in range(self.n_templates):
##            #print(f'--- {t} / {self.n_templates} ---')
##            template = self.templates[t]
##            weights = np.zeros(tp_array.shape[:-1], dtype=np.float32)
##            weights[:, template.map_to_subnet, :] = 1.
##            weights /= np.sum(weights, axis=(1, 2))[:, np.newaxis, np.newaxis]
##            above_thrs = self.ellipsoid_distances[self.tids[t]] > distance_threshold
##            weights[above_thrs, ...] = 0.
##            below_thrs = self.ellipsoid_distances[self.tids[t]] < distance_threshold
##            for s in range(n_stations):
##                for c in range(n_components):
##                    # use trick to keep station and component dim
##                    slice_ = np.index_exp[:, s:s+1, c:c+1, :]
##                    data_ = data[(t,)+slice_[1:]]
##                    # discard all templates that have weights equal to zero
##                    keep = (weights[:, s, c] != 0.)\
##                          &  (np.sum(tp_array[slice_], axis=-1).squeeze() != 0.)
##                    if (np.sum(keep) == 0) or (np.sum(data[(t,)+slice_[1:]]) == 0):
##                        # occurs if this station is not among the
##                        # n_stations closest stations
##                        # or if no data were available
##                        continue
##                    cc = fmf.matched_filter(
##                            tp_array[slice_][keep, ...], moveouts[slice_[:-1]][keep, ...],
##                            weights[slice_[:-1]][keep, ...], data[(t,)+slice_[1:]],
##                            1, arch=device)
##                    # add best contribution from this channel to
##                    # the average inter-template CC
##                    intertp_cc[t, keep] += np.max(cc, axis=-1)
##        # make the CC matrix symmetric by averaging the lower
##        # and upper triangles
##        intertp_cc = (intertp_cc + intertp_cc.T)/2.
##        self.intertp_cc = pd.DataFrame(
##                columns=[tid for tid in self.tids],
##                index=[tid for tid in self.tids],
##                data=intertp_cc)
##
##    # -------------------------------------------
##    #       GrowClust related methods
##    # -------------------------------------------
##    def cross_correlate(self, duration, offset_start_S, offset_start_P,
##                        max_lag=20, n_stations=30):
##        """
##        Create an FamilyEvents instance to access its methods.
##         --- Should be rewritten properly ---
##        """
##        family = FamilyEvents(
##                self.tids[0], self.db_path_T, self.db_path_M, db_path=self.db_path)
##        family.detection_waveforms = \
##                np.float32([np.mean(stack.data, axis=0) for stack in self.stacks])
##        family.n_events = len(self.tids)
##        # trim waveforms
##        family.trim_waveforms(duration, offset_start_S, offset_start_P)
##        # cross-correlate trimmed waveforms
##        family.cross_correlate(
##                n_stations=n_stations, max_lag=max_lag, device='precise')
##        self.CCs_stations = family.stations
##        new_shape = (self.n_templates, self.n_templates, -1)
##        self.CCs_P = family.CCs_P.reshape(new_shape)
##        self.CCs_S = family.CCs_S.reshape(new_shape)
##        self.lags_P = family.lags_P.reshape(new_shape)
##        self.lags_S = family.lags_S.reshape(new_shape)
##        self.max_lag = max_lag
##        del family
##
##    def read_GrowClust_output(self, filename, path, add_results_to_db=False):
##        print('Reading GrowClust output from {}'.
##                format(os.path.join(path, filename)))
##        ot_, lon_, lat_, dep_, err_h_, err_v_, err_t_, tids_ = \
##                [], [], [], [], [], [], [], []
##        with open(os.path.join(path, filename), 'r') as f:
##            for line in f.readlines():
##                line = line.split()
##                year, month, day, hour, minu, sec = line[:6]
##                # correct date if necessary
##                if int(day) == 0:
##                    date_ = udt(f'{year}-{month}-01')
##                    date_ -= datetime.timedelta(days=1)
##                    year, month, day = date_.year, date_.month, date_.day
##                # correct seconds if necessary
##                sec = float(sec)
##                if sec == 60.:
##                    sec -= 0.001
##                ot_.append(udt(f'{year}-{month}-{day}T{hour}:{minu}:{sec}').timestamp)
##                tid = int(line[6])
##                latitude, longitude, depth = list(map(float, line[7:10]))
##                lon_.append(longitude)
##                lat_.append(latitude)
##                dep_.append(depth)
##                mag = float(line[10])
##                q_id, cl_id, cluster_pop = list(map(int, line[11:14]))
##                n_pairs, n_P_dt, n_S_dt = list(map(int, line[14:17]))
##                rms_P, rms_S = list(map(float, line[17:19]))
##                err_h, err_v, err_t = list(map(float, line[19:22])) # errors in km and sec
##                err_h_.append(err_h)
##                err_v_.append(err_v)
##                err_t_.append(err_t)
##                latitude_init, longitude_init, depth_init =\
##                        list(map(float, line[22:25]))
##                tids_.append(tid)
##        for t, tid in enumerate(tids_):
##            tt = self.tids_map[tid]
##            self.templates[tt].relocated_latitude = lat_[t]
##            self.templates[tt].relocated_longitude = lon_[t]
##            self.templates[tt].relocated_depth = dep_[t]
##            self.templates[tt].reloc_err_h = err_h_[t]
##            self.templates[tt].reloc_err_v = err_v_[t]
##            if add_results_to_db:
##                keys = ['relocated_longitude', 'relocated_latitude',
##                        'relocated_depth', 'reloc_err_h', 'reloc_err_v']
##                with h5.File(os.path.join(
##                    self.db_path, self.db_path_T, f'template{tid}meta.h5'), 'a') as f:
##                    for key in keys:
##                        if key in f.keys():
##                            del f[key]
##                        f.create_dataset(key, data=getattr(self.templates[tt], key))
##
##    def write_GrowClust_stationlist(self, filename, path,
##                                    network_filename='all_stations.in'):
##        """
##        This routine assumes that cross_correlate was called
##        shortly before and that self.template still has the same
##        set of stations as the ones used for the inter-event CCs.
##        """
##        net = Network(network_filename)
##        net.read()
##        subnet = net.subset(
##                self.CCs_stations, net.components, method='keep')
##        with open(os.path.join(path, filename), 'w') as f:
##            for s in range(len(subnet.stations)):
##                f.write('{:<5}\t{:.6f}\t{:.6f}\t{:.3f}\n'.
##                        format(subnet.stations[s], subnet.latitude[s],
##                               subnet.longitude[s], -1000.*subnet.depth[s]))
##
##    def write_GrowClust_eventlist(self, filename, path):
##        from obspy.core import UTCDateTime as udt
##        # fake date
##        ot = udt('2000-01-01')
##        # fake mag
##        mag = 1.
##        with open(os.path.join(path, filename), 'w') as f:
##            for t, tid in enumerate(self.tids):
##                # all events are given the template location
##                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
##                        format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
##                               ot.second, self.templates[t].latitude,
##                               self.templates[t].longitude, self.templates[t].depth,
##                               mag, tid))
##
##    def write_GrowClust_CC(self, filename, path, CC_threshold=0.):
##        if not hasattr(self, 'CCs_S'):
##            print('Need to cross_correlate first.')
##            return
##        sr = self.templates[0].sampling_rate
##        with open(os.path.join(path, filename), 'w') as f:
##            for t1, tid1 in enumerate(self.tids):
##                for t2, tid2 in enumerate(self.tids):
##                    if t2 == t1:
##                        continue
##                    f.write('#\t{}\t{}\t0.0\n'.format(tid1, tid2))
##                    for s in range(len(self.CCs_stations)):
##                        # CCs that are zero are pairs that have to be skipped
##                        if self.CCs_S[t1, t2, s] > CC_threshold:
##                            f.write('  {:>5} {} {:.4f} S\n'.
##                                    format(self.CCs_stations[s],
##                                           self.lags_S[t1, t2, s]/sr,
##                                           self.CCs_S[t1, t2, s]))
##                        if self.CCs_P[t1, t2, s] > CC_threshold:
##                            f.write('  {:>5} {} {:.4f} P\n'.
##                                    format(self.CCs_stations[s],
##                                           self.lags_P[t1, t2, s]/sr,
##                                           self.CCs_P[t1, t2, s]))
