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
        """Compute the distance between all station pairs.

        Returns:
            DataFrame: A pandas DataFrame containing the interstation distances.
                       The rows and columns represent the station names, and the
                       values represent the distances in kilometers.
        """
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
                    d.squeeze() ** 2 + (self.depth[s] - self.depth) ** 2
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
        catalog["origin_time"] = pd.to_datetime(
            np.asarray(origin_times, dtype="datetime64[ms]")
        )
        catalog.update(kwargs)
        if event_ids is not None:
            catalog["event_id"] = event_ids
        self.catalog = pd.DataFrame(catalog)
        if event_ids is not None:
            self.catalog.set_index("event_id", inplace=True)
        self.catalog.sort_values("origin_time", inplace=True)

    @property
    def latitude(self):
        return self.catalog.latitude.values

    @property
    def longitude(self):
        return self.catalog.longitude.values

    @property
    def depth(self):
        return self.catalog.depth.values

    @property
    def origin_time(self):
        return self.catalog.origin_time.values

    @property
    def n_events(self):
        return len(self.catalog)

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
        """Build catalog from list of `Event` instances.
        
        Parameters
        ----------
        events : list
            List of `Event` instances.
        extra_attributes : list, optional
            Default attributes included in the catalog are: longitude,
            latitude, depth and origin time. Any extra attribute that
            should be returned in the catalog should be given here as
            a string in a list.
            Example: `extra_attributes=['hmax_inc', 'vmax_inc']`
        fill_value : any, optional
            Value that is returned in the catalog if an attribute
            from `extra_attributes` is not found. Default is `numpy.nan`.

        Returns
        -------
        catalog : `Catalog`
            `Catalog` instance.
        """
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
        
        Parameters
        ----------
        dataframe : `pandas.DataFrame`
            `pandas.DataFrame` from which the `Catalog` instance will
            be built.

        Returns
        -------
        catalog : `Catalog`
            `Catalog` instance.
        """
        catalog = cls(
            dataframe["longitude"],
            dataframe["latitude"],
            dataframe["depth"],
            dataframe["origin_time"],
            **dataframe.drop(columns=["longitude", "latitude", "depth", "origin_time"]),
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
        """Read all detected events and build catalog.

        Parameters
        ----------
        filename : string
            Name of the hdf5 file with the event data base.
        db_path : string, optional
            Name of the directory where `filename` is located.
            Default is `config.OUTPUT_PATH`.
        gid : string or float or int, optional
            Name of the group to read within the hdf5 file. If
            None (default), reads all the groups in the root
            directory of the hdf5 file.
        extra_attributes : list, optional
            Default attributes included in the catalog are: longitude,
            latitude, depth and origin time. Any extra attribute that
            should be returned in the catalog should be given here as
            a string in a list.
            Example: `extra_attributes=['hmax_inc', 'vmax_inc']`
        fill_value : any, optional
            Value that is returned in the catalog if an attribute
            from `extra_attributes` is not found. Default is `numpy.nan`.
        return_events : boolean, optional
            If True, returns the list of `Event` instances in addition to
            the `Catalog` instance. Default is False.

        Returns
        -------
        catalog : `Catalog`
            `Catalog` instance.
        events : list, optional
            List of `Event` instances of `return_events=True`.
        """
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
    def plot_time_statistics(self, UTC_local_corr=0., figsize=(16, 7), **kwargs):
        """Plot the histograms of time of the day and day of the week.

        Parameters
        ----------
        figsize : tuple of floats, optional
            Size, in inches, of the figure (width, height). Default is (16, 7).
        UTC_local_corr : float, optional
            Apply UTC to local time correction such that:
                `local_hour = UTC_hour + UTC_local_corr`

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created matplotlib Figure object.
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

        ((self.catalog["origin_time"].dt.hour + UTC_local_corr)%24)\
                .hist(bins=np.arange(25), ax=axes[1])
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
        plot_uncertainties=False,
        depth_colorbar=True,
        **kwargs,
    ):
        """Plot epicenters on map.

        Parameters
        ------------
        ax : matplotlib.pyplot.Axes, default to None
            If None, create a new `plt.Figure` and `plt.Axes` instances. If
            speficied by user, use the provided instance to plot.
        figsize : tuple of floats, default to (20, 10)
            Size, in inches, of the figure (width, height).
        depth_min : scalar float, default to 0
            Smallest depth, in km, in the depth colormap.
        depth_max : scalar float, default to 20
            Largest depth, in km, in the depth colormap.
        network : BPMF.dataset.Network, optional
            If provided, use information in `network` to plot the stations.
        plot_uncertainties : boolean, default to False
            If True, plot the location uncertainty ellipses.
        depth_colorbar : boolean, default to True
            If True, plot the depth colorbar on the left.

        Returns
        ----------
        fig : matplotlib.pyplot.Figure
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
            c=scalar_map.to_rgba(self.depth),
            label="Earthquakes",
            **scatter_kwargs,
            transform=data_coords,
        )
        if plot_uncertainties:
            columns = self.catalog.columns
            if (
                "hmax_unc" in columns
                and "hmin_unc" in columns
                and "az_hmax_unc" in columns
            ):
                for i in range(len(self.longitude)):
                    row = self.catalog.iloc[i]
                    longitude_ellipse, latitude_ellipse = plotting_utils.uncertainty_ellipse(
                        row.hmax_unc,
                        row.hmin_unc,
                        row.az_hmax_unc,
                        row.longitude,
                        row.latitude,
                    )
                    color = scalar_map.to_rgba(self.depth[i])
                    if color[:3] == (1., 1., 1.):
                        # white!
                        color = "dimgrey"
                    ax.plot(
                        longitude_ellipse,
                        latitude_ellipse,
                        color=color,
                        transform=data_coords,
                    )
            else:
                print(
                    "If you want to plot the uncertainty ellipses,"
                    " self.catalog needs the following columns: "
                    "hmax_unc, hmin_unc, az_hmax_unc"
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
        if depth_colorbar:
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
        fig : matplotlib.pyplot.Figure
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
        """Arguments go to `BPMF.utils.get_np_array`.
        """
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
        ----------
        trim_traces : bool, optional
            If True, call `trim_waveforms` to make sure all traces have the same
            start time.

        Notes
        -----
        Additional parameters can be provided as keyword arguments `**reader_kwargs`
        which are passed to the `data_reader` method for waveform reading.

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
        """
        Set the data availability.

        A station is available if at least one station has non-zero data. The
        availability is then accessed via the property `self.availability`.

        Parameters
        ----------
        stations : list of strings or numpy.ndarray
            Names of the stations on which we check availability. If None, use
            `self.stations`.
        components : list, optional
            List of component codes to consider for availability check,
            by default ["N", "E", "Z"]
        component_aliases : dict, optional
            Dictionary mapping component codes to their aliases,
            by default {"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]}

        Returns
        -------
        None
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
        """
        Trim waveforms.

        Adjusts the start and end times of waveforms to ensure all traces
        are synchronized.

        Parameters
        ----------
        starttime : str or datetime, optional
            Start time to use for trimming the waveforms. If None,
            `self.date` is used as the start time. (default None)
        endtime : str or datetime, optional
            End time to use for trimming the waveforms. If None,
            `self.date` + `self.duration` is used as the end time. (default None)

        Returns
        -------
        None
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
        origin_time : string
            Origin time, or detection time, of the event.
        moveouts : numpy.ndarray
            (n_stations, n_phases) float `numpy.ndarray`.
            Moveouts, in seconds, for each station and each phase.
        stations : list of strings
            List of station names corresponding to `moveouts`.
        phases : list of strings
            List of phase names corresponding to `moveouts`.
        data_filename : string
            Name of the data file.
        data_path : string
            Path to the data directory.
        latitude : float, optional
            Event latitude. Default is None.
        longitude : float, optional
            Event longitude. Default is None.
        depth : float, optional 
            Event depth. Default is None.
        sampling_rate : float, optionak
            Sampling rate (Hz) of the waveforms. It should be different from
            None (default) only if you plan to read preprocessed data with a fixed
            sampling rate.
        components : list of strings, optional
            List of the components to use in reading and plotting methods.
            Default is `['N','E','Z']`.
        component_aliases : dictionary, optional
            Each entry of the dictionary is a list of strings.
            `component_aliases[comp]` is the list of all aliases used for
            the same component 'comp'. For example, `component_aliases['N'] =
            ['N', '1']` means that both the 'N' and '1' channels will be mapped
            to the Event's 'N' channel.
        id : string, optional
            Identifying label. Default is None, in which case the id is taken
            to be YYYYMMDD_HHMMSS.
        data_reader : function, optional
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
            self.id = self.origin_time.strftime("%Y%m%d_%H%M%S.%f")
        else:
            self.id = id
        self.data_reader = data_reader

    @classmethod
    def read_from_file(
        cls,
        filename=None,
        db_path=cfg.INPUT_PATH,
        hdf5_file=None,
        gid=None,
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
            path_database = parent_file.filename
            if gid is not None:
                # go to specified group
                f = parent_file[gid]
            else:
                f = parent_file
            close_file = True  # remember to close file at the end
        else:
            f = hdf5_file
            if hasattr(f, "file"):
                path_database = f.file.filename
                gid = f.name
            else:
                path_database = f.filename
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
        event.path_database = path_database
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
        elif hasattr(self, "aux_data") and "hmin_unc" in self.aux_data:
            return self.aux_data["hmin_unc"]
        else:
            self.hor_ver_uncertainties()
            return self._hmin_unc

    @property
    def vmax_unc(self):
        if hasattr(self, "_vmax_unc"):
            return self._vmax_unc
        elif hasattr(self, "aux_data") and "vmax_unc" in self.aux_data:
            return self.aux_data["vmax_unc"]
        else:
            self.hor_ver_uncertainties()
            return self._vmax_unc

    @property
    def az_hmax_unc(self):
        if hasattr(self, "_az_hmax_unc"):
            return self._az_hmax_unc
        elif hasattr(self, "aux_data") and "az_hmax_unc" in self.aux_data:
            return self.aux_data["az_hmax_unc"]
        else:
            self.hor_ver_uncertainties()
            return self._az_hmax_unc

    @property
    def az_hmin_unc(self):
        if hasattr(self, "_az_hmin_unc"):
            return self._az_hmin_unc
        elif hasattr(self, "aux_data") and "az_hmin_unc" in self.aux_data:
            return self.aux_data["az_hmin_unc"]
        else:
            self.hor_ver_uncertainties()
            return self._az_hmin_unc

    @property
    def pl_vmax_unc(self):
        if hasattr(self, "_pl_vmax_unc"):
            return self._pl_vmax_unc
        elif hasattr(self, "aux_data") and "pl_vmax_unc" in self.aux_data:
            return self.aux_data["pl_vmax_unc"]
        else:
            self.hor_ver_uncertainties()
            return self._pl_vmax_unc

    @property
    def location(self):
        return [self.longitude, self.latitude, self.depth]

    @property
    def source_receiver_dist(self):
        if hasattr(self, "_source_receiver_dist"):
            return self._source_receiver_dist[self.stations]
        else:
            print(
                "You need to set source_receiver_dist before."
                " Call self.set_source_receiver_dist(network)"
            )
            return

    @property
    def source_receiver_epicentral_dist(self):
        if hasattr(self, "_source_receiver_epicentral_dist"):
            return self._source_receiver_epicentral_dist[self.stations]
        else:
            print(
                "You need to set source_receiver_epicentral_dist before."
                " Call self.set_source_receiver_dist(network)"
            )
            return


    @property
    def sr(self):
        return self.sampling_rate

    def get_np_array(self, stations, components=None, priority="HH", verbose=True):
        """Arguments are passed to `BPMF.utils.get_np_array`.
        """
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

    def get_peak_amplitudes(self, stations, components):
        """Get peak waveform amplitudes.

        The peak waveform amplitudes are typically used to compute
        amplitude-based local magnitudes.

        Parameters
        ----------
        stations : list of strings
            Names of the stations to include in the output array. Define the order
            of the station axis.
        components : list of strings, default to ['N','E','Z']
            Names of the components to include in the output array. Define the order
            of the component axis.

        Returns
        -------
        peak_amplitudes : numpy.ndarray
            (num_stations, num_components) numpy.ndarray with the peak
            waveform amplitude on each channel.
        """
        waveforms = self.get_np_array(stations, components=components)
        peak_amplitudes = np.max(
                np.abs(waveforms - np.mean(waveforms, axis=-1, keepdims=True)), axis=-1
                )
        return peak_amplitudes


    def hor_ver_uncertainties(self, mode="intersection"):
        """
        Compute the horizontal and vertical uncertainties on location.

        Returns the errors as given by the 68% confidence ellipsoid.

        Parameters
        ----------
        mode : str, optional
            Specifies the mode for calculating the horizontal uncertainties.
            - If 'intersection', the horizontal uncertainties are the lengths
            of the semi-axes of the ellipse defined by the intersection between
            the confidence ellipsoid and the horizontal plane.
            - If 'projection', the horizontal uncertainties are the maximum and minimum
            spans of the confidence ellipsoid in the horizontal directions.
            (default 'intersection')

        Returns
        -------
        None

        Raises
        ------
        None

        New Attributes
        --------------
        _hmax_unc : float
            The maximum horizontal uncertainty in kilometers.
        _hmin_unc : float
            The minimum horizontal uncertainty in kilometers.
        _vmax_unc : float
            The maximum vertical uncertainty in kilometers.
        _az_hmax_unc : float
            The azimuth (angle from north) of the maximum horizontal uncertainty in degrees.
        _az_hmin_unc : float
            The azimuth (angle from north) of the minimum horizontal uncertainty in degrees.

        Notes
        -----
        - The sum of _hmax_unc and _vmax_unc does not necessarily equal the
        maximum length of the uncertainty ellipsoid's semi-axis; the latter
        represents the longest semi-axis of the ellipsoid.
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
        self._pl_vmax_unc = np.rad2deg(np.arccos(v[2, vertical_unc.argmax()]))
        self._pl_vmax_unc = min(self._pl_vmax_unc, 180. - self._pl_vmax_unc)
        self._az_hmax_unc = az_hmax
        self._az_hmin_unc = az_hmin

    def n_closest_stations(self, n, available_stations=None):
        """
        Adjust `self.stations` to the `n` closest stations.

        Finds the `n` closest stations based on distance and modifies `self.stations` accordingly.
        The instance's properties will also change to reflect the updated station selection.

        Parameters
        ----------
        n : int
            The number of closest stations to fetch.
        available_stations : list of strings, optional
            The list of stations from which the closest stations are searched.
            If certain stations are known to lack data, they can be excluded
            from the closest stations selection. (default None)

        Returns
        -------
        None
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
        search_win_sec=2.0,
        ml_model=None,
        ml_model_name="original",
        keep_probability_time_series=False,
        **kwargs,
    ):
        """
        Use PhaseNet (Zhu et al., 2019) to pick P and S waves (Event class).

        Parameters
        ----------
        duration : float
            Duration of the time window, in seconds, to process and search
            for P and S wave arrivals.
        threshold_P : float, optional
            Threshold on PhaseNet's probabilities to trigger the
            identification of a P-wave arrival. (default 0.60)
        threshold_S : float, optional
            Threshold on PhaseNet's probabilities to trigger the
            identification of an S-wave arrival. (default 0.60)
        offset_ot : float, optional
            Offset in seconds to apply to the origin time.
            (default cfg.BUFFER_EXTRACTED_EVENTS_SEC)
        mini_batch_size : int, optional
            Number of traces processed in a single batch by PhaseNet. (default 126)
        phase_on_comp : dict, optional
            Dictionary defining the seismic phase extracted on each component.
            For example, `phase_on_comp['N']` specifies the phase extracted on
            the north component.
            (default {"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"})
        component_aliases : dict, optional
            Dictionary mapping each component to a list of strings representing
            aliases used for the same component. For example, `component_aliases['N'] = ['N', '1']`
            means that both the 'N' and '1' channels will be mapped to the
            Event's 'N' channel.
            (default {"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]})
        upsampling : int, optional
            Upsampling factor applied before calling PhaseNet. (default 1)
        downsampling : int, optional
            Downsampling factor applied before calling PhaseNet. (default 1)
        use_apriori_picks : bool, optional
            Flag indicating whether to use apriori picks for refining
            the P and S wave picks. (default False)
        search_win_sec : float, optional
            Search window size, in seconds, used for refining
            the P and S wave picks. (default 2.0)
        ml_model : object, optional
            Pre-trained PhaseNet model object. If not provided,
            the default model will be loaded. (default None)
        ml_model_name : str, optional
            Name of the pre-trained PhaseNet model to load if `ml_model`
            is not provided. (default "original")
        **kwargs : dict, optional
            Extra keyword arguments passed to `Event.read_waveforms`.

        Returns
        -------
        None

        Notes
        -----
        - PhaseNet must be used with 3-component data.
        - Results are stored in the object's attribute `self.picks`.
        """
        from torch import no_grad, from_numpy

        if ml_model is None:
            import seisbench.models as sbm
            ml_model = sbm.PhaseNet.from_pretrained(ml_model_name)
            ml_model.eval()

        ml_p_index = kwargs.get("ml_P_index", 1)
        ml_s_index = kwargs.get("ml_S_index", 2)

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

            data_arr = resample_poly(data_arr, upsampling, downsampling, axis=-1)
            # momentarily update samping_rate
            sampling_rate0 = float(self.sampling_rate)
            self.sampling_rate = self.sr * upsampling / downsampling
        data_arr_n = utils.normalize_batch(
                data_arr, 
                normalization_window_sample=kwargs.get(
                    "normalization_window_sample", data_arr.shape[-1]
                    )
                )
        closest_pow2 = int(np.log2(data_arr_n.shape[-1])) + 1
        diff = 2**closest_pow2 - data_arr_n.shape[-1]
        left = diff//2
        right = diff//2 + diff%2
        data_arr_n = np.pad(
                data_arr_n,
                ((0, 0), (0, 0), (left, right)),
                mode="reflect"
                )
        with no_grad():
            ml_probas = ml_model(
                    from_numpy(data_arr_n).float()
                    )
            ml_probas = ml_probas.detach().numpy()
        if keep_probability_time_series:
            self.probability_time_series = pd.DataFrame(
                    index=self.stations,
                    columns=["P", "S"]
                    )
            times = (
                    np.arange(data_arr_n.shape[-1] - left - right).astype("float64")
                    /
                    float(self.sampling_rate)
                    )
            if times[1] > 1.e-3:
                times = (1000 * times).astype("timedelta64[ms]")
            else:
                times = (1e9 * times).astype("timedelta64[ns]")
            self.probability_times = (
                    np.datetime64(self.traces[0].stats.starttime) + times
                    )
            for s, sta in enumerate(self.stations):
                self.probability_time_series.loc[
                        sta, "P"
                        ] = ml_probas[s, ml_p_index, left:-right]
                self.probability_time_series.loc[
                        sta, "S"
                        ] = ml_probas[s, ml_s_index, left:-right]
        # find picks and store in dictionaries
        picks = {}
        picks["P_picks"] = {}
        picks["P_proba"] = {}
        picks["S_picks"] = {}
        picks["S_proba"] = {}
        for s, sta in enumerate(self.stations):
            picks["P_proba"][sta], picks["P_picks"][sta] = utils.trigger_picks(
                    ml_probas[s, ml_p_index, left:-right], threshold_P, 
                    )
            picks["S_proba"][sta], picks["S_picks"][sta] = utils.trigger_picks(
                    ml_probas[s, ml_s_index, left:-right], threshold_S, 
                    )
        if use_apriori_picks and hasattr(self, "arrival_times"):
            columns = []
            if "P" in self.phases:
                columns.append("P")
            if "S" in self.phases:
                columns.append("S")
            prior_knowledge = pd.DataFrame(columns=columns)
            #for sta in self.stations:
            for sta in self.arrival_times.index:
                for ph in prior_knowledge.columns:
                    prior_knowledge.loc[sta, ph] = utils.sec_to_samp(
                        udt(self.arrival_times.loc[sta, f"{ph}_abs_arrival_times"])
                        - self.traces[0].stats.starttime,
                        sr=self.sampling_rate,
                    )
        else:
            prior_knowledge = None
        # only used if use_apriori_picks is True
        search_win_samp = utils.sec_to_samp(search_win_sec, sr=self.sampling_rate)
        # keep best P- and S-wave pick on each 3-comp seismogram
        picks = utils.get_picks(
            picks,
            prior_knowledge=prior_knowledge,
            search_win_samp=search_win_samp,
        )
        # format picks in pandas DataFrame
        pandas_picks = {"stations": self.stations}
        for ph in ["P", "S"]:
            rel_picks_sec = np.zeros(len(self.stations), dtype=np.float32)
            proba_picks = np.zeros(len(self.stations), dtype=np.float32)
            abs_picks = np.zeros(len(self.stations), dtype=object)
            for s, sta in enumerate(self.stations):
                if sta in picks[f"{ph}_picks"].keys():
                    rel_picks_sec[s] = picks[f"{ph}_picks"][sta][0] / self.sr
                    proba_picks[s] = picks[f"{ph}_proba"][sta][0]
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
        n_threads=1,
        **reader_kwargs,
    ):
        """
        Read waveform data (Event class).

        Parameters
        ----------
        duration : float
            Duration of the extracted time windows in seconds.
        phase_on_comp : dict, optional
            Dictionary defining the seismic phase extracted on each component.
            For example, `phase_on_comp['N']` specifies the phase extracted on the north component.
            (default {"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"})
        component_aliases : dict, optional
            Dictionary mapping each component to a list of strings
            representing aliases used for the same component.
            For example, `component_aliases['N'] = ['N', '1']` means that both the
            'N' and '1' channels will be mapped to the Event's 'N' channel.
            (default {"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]})
        offset_phase : dict, optional
            Dictionary defining when the time window starts with respect to the pick.
            A positive offset means the window starts before the pick.
            Not used if `time_shifted` is False. (default {"P": 1.0, "S": 4.0})
        time_shifted : bool, optional
            If True (default), the moveouts are used to extract time windows
            from specific seismic phases. If False, windows are simply extracted
            with respect to the origin time.
        offset_ot : float, optional
            Only used if `time_shifted` is False.
            Time in seconds taken before `origin_time`.
            (default cfg.BUFFER_EXTRACTED_EVENTS_SEC)
        data_reader : function, optional
            Function that takes a path and optional keyword arguments to read
            data from this path and returns an `obspy.Stream` instance.
            If None (default), this function uses `self.data_reader` and
            returns None if `self.data_reader=None`.
        n_threads : int, optional
            Number of threads used to parallelize reading. Default is 1 (sequential reading).
        **reader_kwargs : dict, optional
            Extra keyword arguments passed to the `data_reader` function.

        Returns
        -------
        None

        Raises
        ------
        None

        Notes
        -----
        - This function populates the `self.traces` attribute with the waveform data.
        - The `duration` and `component_aliases` are stored as attributes in the instance.
        - The `phase_on_comp` and `offset_phase` information is stored
        as auxiliary data in the instance.
        - If `reader_kwargs` contains the "attach_response" key set to True,
        traces without instrument response information are removed from `self.traces`.
        """
        from obspy import Stream
        from functools import partial
        if n_threads != 1:
            from concurrent.futures import ThreadPoolExecutor

        if data_reader is None:
            data_reader = self.data_reader
        if data_reader is None:
            print("You need to specify a data reader for the class instance.")
            return
        self.traces = Stream()
        self.duration = duration
        self.n_samples = utils.sec_to_samp(self.duration, sr=self.sr)
        reading_task_list = []
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
                    reading_task_list.append(
                            partial(
                                data_reader,
                                where=self.where,
                                station=sta,
                                channel=cp_alias,
                                starttime=pick,
                                endtime=pick + duration,
                                **reader_kwargs,
                                )
                            )
        if n_threads != 1:
            if n_threads in [0, None, "all"]:
                # n_threads = None means use all CPUs
                n_threads = None
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                traces_ = list(
                        executor.map(
                            lambda i: reading_task_list[i](),
                            range(len(reading_task_list))
                            )
                        )
            for tr in traces_:
                self.traces += tr
        else:
            for task in reading_task_list:
                self.traces += task()
        if reader_kwargs.get("attach_response", False):
            # remove traces for which we could not find the instrument response
            for tr in self.traces:
                if "response" not in tr.stats:
                    self.traces.remove(tr)
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
        """
        Wrapper function for earthquake relocation using multiple methods.

        This function serves as an interface for the earthquake relocation process
        using different relocation routines. The routine to be used is specified
        by the `routine` parameter. All keyword arguments are passed to the
        corresponding routine.

        Parameters
        ----------
        routine : str, optional
            Method used for the relocation. Available options are:
            - 'NLLoc': Calls the `relocate_NLLoc` method and requires the event
              object to have the `picks` attribute.
            - 'beam': Calls the `relocate_beam` method.
            Default is 'NLLoc'.

        Notes
        -----
        - The `relocate` function acts as a wrapper that allows for flexibility in
          choosing the relocation routine.
        - Depending on the specified `routine`, the function calls the corresponding
          relocation method with the provided arguments and keyword arguments.
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
        uncertainty_method="spatial",
        restricted_domain_side_km=100.,
        device="cpu",
        **kwargs,
    ):
        """
        Use a Beamformer instance for backprojection to relocate the event.

        Parameters
        ----------
        beamformer : BPMF.template_search.Beamformer
            Beamformer instance used for backprojection.
        duration : float, optional
            Duration, in seconds, of the extracted time windows. Default is 60.0 seconds.
        offset_ot : float, optional
            Time in seconds taken before the `origin_time`. Only used if `time_shifted` is False.
            Default is `cfg.BUFFER_EXTRACTED_EVENTS_SEC`.
        phase_on_comp : dict, optional
            Dictionary defining which seismic phase is extracted on each component.
            For example, `phase_on_comp['N']` gives the phase extracted on the north component.
            Default is `{"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"}`.
        component_aliases : dict, optional
            Dictionary where each entry is a list of strings. `component_aliases[comp]`
            is the list of all aliases used for the same component 'comp'. For example,
            `component_aliases['N'] = ['N', '1']` means that both the 'N' and '1' channels
            will be mapped to the Event's 'N' channel. Default is `{"N": ["N", "1"],
            "E": ["E", "2"], "Z": ["Z"]}`.
        waveform_features : numpy.ndarray, optional
            If not None, it must be a `(num_stations, num_channels, num_time_samples)`
            numpy.ndarray containing waveform features or characteristic functions
            that are backprojected onto the grid of theoretical seismic sources.
            Default is None.
        restricted_domain_side_km : float, optional
            The location uncertainties are computed on the full 3D beam at the time when
            the 4D beam achieves its maximum over the `duration` seconds. To avoid having
            grid-size-dependent uncertainties, it is useful to truncate the domain around
            the location of the maximum beam. This parameter controls the size of the
            truncated domain. Default is 100.0 km.
        device : str, optional
            Device to be used for computation. Must be either 'cpu' (default) or 'gpu'.
        **kwargs : dict, optional
            Additional keyword arguments.

        Notes
        -----
        - The function updates the `origin_time`, `longitude`, `latitude`, and `depth`
          attributes of the event based on the maximum focusing of the beamformer.
        - Location uncertainties are estimated based on the computed likelihood,
          a restricted domain, and the location coordinates.
        - The estimated uncertainties can be accessed through the `Event`'s
        properties `Event.hmax_unc`, `Event.hmin_unc`, `Event.vmax_unc`.
        """
        from .template_search import Beamformer, envelope

        if kwargs.get("read_waveforms", True) and waveform_features is None:
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
            norm[norm == 0.0] = 1.0
            data_arr /= norm
            waveform_features = envelope(data_arr)
        self.waveform_features = waveform_features
        # print(waveform_features)
        out_of_bounds = kwargs.get("out_of_bounds", "flexible")
        if uncertainty_method == "spatial":
            reduce = "none"
        elif uncertainty_method == "temporal":
            reduce = "max"
        beamformer.backproject(
                waveform_features, device=device, reduce=reduce, out_of_bounds=out_of_bounds
                )
        # find where the maximum focusing occurred
        if uncertainty_method == "spatial":
            src_idx, time_idx = np.unravel_index(
                beamformer.beam.argmax(), beamformer.beam.shape
            )
        elif uncertainty_method == "temporal":
            time_idx = beamformer.maxbeam.argmax()
            src_idx = beamformer.maxbeam_sources[time_idx]
        # update hypocenter
        self.origin_time = (
            self.traces[0].stats.starttime + time_idx / self.sampling_rate
        )
        self.longitude = beamformer.source_coordinates["longitude"].iloc[src_idx]
        self.latitude = beamformer.source_coordinates["latitude"].iloc[src_idx]
        self.depth = beamformer.source_coordinates["depth"].iloc[src_idx]
        # estimate location uncertainty
        if uncertainty_method == "spatial":
            # 1) define a restricted domain
            domain = beamformer._rectangular_domain(
                    self.longitude,
                    self.latitude,
                    side_km=restricted_domain_side_km,
                    )
            # 2) compute likelihood
            likelihood = beamformer._likelihood(beamformer.beam[:, time_idx])
            likelihood_domain = domain
        elif uncertainty_method == "temporal":
            # 1) likelihood is given by Gibbs distribution of maxbeam
            effective_kT = kwargs.get("effective_kT", 0.33)
            gibbs_cutoff = kwargs.get("gibbs_cutoff", 0.25)
            gibbs_weight = np.exp(
                    -(beamformer.maxbeam.max() - beamformer.maxbeam) / effective_kT
                    )
            domain = beamformer.maxbeam_sources[gibbs_weight > gibbs_cutoff]
            likelihood = gibbs_weight
            likelihood_domain = gibbs_weight > gibbs_cutoff
        beamformer.likelihood = likelihood
        # 3) compute uncertainty
        hunc, vunc = beamformer._compute_location_uncertainty(
                self.longitude,
                self.latitude,
                self.depth,
                likelihood[likelihood_domain],
                domain,
                )
        # 4) set attributes
        self._hmax_unc = hunc
        self._hmin_unc = hunc
        self._az_hmax_unc = 0.
        self._az_hmin_unc = 0.
        self._vmax_unc = vunc
        self.set_aux_data(
                {"hmax_unc": hunc, "hmin_unc": hunc, "az_hmax_unc": 0., "vmax_unc": vunc}
                )
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
            for p, ph in enumerate(beamformer.phases):
                self.arrival_times.loc[sta, f"{ph}_tt_sec"] = (
                    travel_times[s, p] / self.sampling_rate
                )
                self.arrival_times.loc[sta, f"{ph}_abs_arrival_times"] = (
                    self.origin_time + self.arrival_times.loc[sta, f"{ph}_tt_sec"]
                )
        for ph in beamformer.phases:
            self.arrival_times[f"{ph}_tt_sec"] = (
                    self.arrival_times[f"{ph}_tt_sec"].astype("float32")
                    )

    def relocate_NLLoc(
            self, stations=None, method="EDT", verbose=0, cleanup_out_dir=True, **kwargs
            ):
        """
        Relocate the event using NonLinLoc (NLLoc) based on the provided picks.

        Parameters
        ----------
        stations : list of str, optional
            Names of the stations to include in the relocation process. If None,
            all stations in `self.stations` are used. Default is None.
        method : str, optional
            Optimization algorithm used by NonLinLoc. Available options are:
            'GAU_ANALYTIC', 'EDT', 'EDT_OT', 'EDT_OT_WT_ML'. Refer to NonLinLoc's
            documentation for more details. Default is 'EDT'.
        verbose : int, optional
            Verbosity level of NonLinLoc. If greater than 0, NonLinLoc's outputs
            are printed to the standard output. Default is 0.
        cleanup_out_dir : bool, optional
            If True, NLLoc's output files are deleted after reading the relevant
            information. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments for `BPMF.NLLoc_utils.write_NLLoc_control`.

        Notes
        -----
        - The event's attributes, including the origin time and location, are updated.
        - The theoretical arrival times are attached to the event in `Event.arrival_times`.
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
        if cleanup_out_dir:
            for fn in glob.glob(os.path.join(output_dir, "*")):
                pathlib.Path(fn).unlink()
            if os.path.isdir(output_dir):
                # add this protection against unexpected
                # external change
                pathlib.Path(output_dir).rmdir()

    def remove_outlier_picks(self, max_diff_percent=25.0):
        """
        Remove picks that are too far from the predicted arrival times.

        Picks that have a difference exceeding the specified threshold between the
        picked arrival time and the predicted arrival time are considered outliers
        and are removed.

        Parameters
        ----------
        max_diff_percent : float, optional
            Maximum allowable difference, in percentage, between the picked and
            predicted arrival times. Picks with a difference greater than this
            threshold will be considered outliers and removed. Default is 25.0.
        """
        stations_outlier = []
        for sta in self.stations:
            for ph in self.phases:
                if pd.isna(self.picks.loc[sta, f"{ph}_abs_picks"]):
                    # no valid pick, pass
                    continue
                pick = pd.Timestamp(str(self.picks.loc[sta, f"{ph}_abs_picks"]))
                predicted = pd.Timestamp(
                    str(self.arrival_times.loc[sta, f"{ph}_abs_arrival_times"])
                )
                predicted_tt = self.arrival_times.loc[sta, f"{ph}_tt_sec"]
                # use a minimum value for predicted_tt of a few samples
                # to avoid issues arising when using self.set_arrival_times_to_moveouts
                # because, by definition of a moveout, the min moveout is 0
                predicted_tt = max(predicted_tt, 5/self.sampling_rate)
                diff_percent = (
                    100.0 * abs((pick - predicted).total_seconds()) / predicted_tt
                )
                if diff_percent > max_diff_percent:
                    stations_outlier.append(sta)
                    self.picks.loc[sta, f"{ph}_abs_picks"] = np.nan
                    self.picks.loc[sta, f"{ph}_picks_sec"] = np.nan
                    self.picks.loc[sta, f"{ph}_probas"] = np.nan

    def zero_out_clipped_waveforms(self, kurtosis_threshold=-1.0):
        """
        Find waveforms with anomalous statistic and zero them out.

        This function identifies waveforms with a kurtosis value below the specified
        threshold as anomalous and zeros out their data. The kurtosis of a waveform
        is a measure of its statistical distribution, with a value of 0 indicating a
        Gaussian distribution.

        Parameters
        ----------
        kurtosis_threshold : float, optional
            Threshold below which the kurtosis is considered anomalous. Waveforms
            with a kurtosis value lower than this threshold will have their data
            zeroed out. Default is -1.0.

        Notes
        -----
        - This is an oversimplified technique to find clipped waveforms.
        """
        from scipy.stats import kurtosis

        if not hasattr(self, "traces"):
            return
        for tr in self.traces:
            if kurtosis(tr.data) < kurtosis_threshold:
                tr.data = np.zeros(len(tr.data), dtype=tr.data.dtype)

    def remove_distant_stations(self, max_distance_km=50.0):
        """
        Remove picks on stations that are further than a given distance.

        This function removes picks on stations that are located at a distance
        greater than the specified maximum distance. The distance between the source
        and each station is computed using the `source_receiver_dist` attribute.
        Picks on stations beyond the maximum distance are set to NaN.

        Parameters
        ----------
        max_distance_km : float, optional
            Maximum distance, in kilometers, beyond which picks are considered to be
            on stations that are too distant. Picks on stations with a distance
            greater than this threshold will be set to NaN. Default is 50.0.

        Notes
        -----
        - The function checks if the `source_receiver_dist` attribute is present in
          the object. If it is not available, an informative message is printed and
          the function returns.
        - For each station in `stations`, the distance between the source and the
          station is retrieved from the `source_receiver_dist` attribute.
        - If the distance of a station is greater than the specified maximum distance
          (`max_distance_km`), the picks associated with that station are set to NaN.
        """
        if self.source_receiver_dist is None:
            print(
                "Call self.set_source_receiver_dist(network) before "
                "using self.remove_distant_stations."
            )
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
            columns=[f"{ph.upper()}_abs_arrival_times" for ph in self.phases]
        )
        for ph in self.phases:
            ph = ph.upper()
            field1 = f"{ph}_abs_arrival_times"
            field2 = f"{ph}_tt_sec"
            for sta in self.moveouts.index:
                self.arrival_times.loc[sta, field1] = (
                    self.origin_time + self.moveouts.loc[sta, f"moveouts_{ph}"]
                )
                self.arrival_times.loc[sta, field2] = (
                        self.moveouts.loc[sta, f"moveouts_{ph}"]
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
        """
        Set source-receiver distances using the provided `network`.

        This function calculates the hypocentral and epicentral distances between
        the event's source location and the stations in the given network. It
        stores the distances as attributes of the event object.

        Parameters
        ----------
        network : dataset.Network
            The `Network` instance containing the station coordinates to use in
            the source-receiver distance calculation.

        Notes
        -----
        - The function uses the `BPMF.utils.compute_distances` function to compute
          both hypocentral and epicentral distances between the event's source
          location and the stations in the network.
        - The computed distances can be accessed with the `Event`'s properties
        `Event.source_receiver_dist` and `Event.source_receiver_epicentral_dist`.
        """
        hypocentral_distances, epicentral_distances = utils.compute_distances(
            [self.longitude],
            [self.latitude],
            [self.depth],
            network.longitude.values,
            network.latitude.values,
            network.depth.values,
            return_epicentral_distances=True
        )
        self._source_receiver_dist = pd.Series(
            index=network.stations,
            data=hypocentral_distances.squeeze(),
            name="source-receiver hypocentral distance (km)",
        )
        self._source_receiver_epicentral_dist = pd.Series(
            index=network.stations,
            data=epicentral_distances.squeeze(),
            name="source-receiver epicentral distance (km)",
        )
        if not hasattr(self, "network_stations"):
            #self.network_stations = self.stations.copy() # why this line?
            self.network_stations = network.stations.values.astype("U")
        self._source_receiver_dist = self.source_receiver_dist.loc[
            self.network_stations
        ]
        self._source_receiver_epicentral_dist = self.source_receiver_epicentral_dist.loc[
            self.network_stations
        ]

    def trim_waveforms(self, starttime=None, endtime=None):
        """
        Trim waveforms to a specified time window.

        This function trims the waveforms in the event's `traces` attribute to the
        specified time window. It ensures that all traces have the same start time
        by adjusting the start time if necessary.

        Parameters
        ----------
        starttime : str or datetime, optional
            The start time of the desired time window. If None, the event's `date`
            attribute is used as the start time. Default is None.
        endtime : str or datetime, optional
            The end time of the desired time window. If None, the end time is set
            as `self.date` + `self.duration`, where `self.date` is the event's date
            attribute and `self.duration` is the event's duration attribute.
            Default is None.
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
            tr.trim(
                starttime=starttime, endtime=endtime, pad=True, fill_value=fill_value
            )

    def update_picks(self):
        """
        Update the picks with respect to the current origin time.

        This function updates the picks of each station with respect to the current
        origin time of the event. It computes the relative pick times by subtracting
        the origin time from the absolute pick times and update the
        `picks` attribute of the event.
        """
        if not hasattr(self, "picks"):
            print("Does not have a `picks` attribute.")
            return
        for station in self.picks.index:
            for ph in self.phases:
                if pd.isnull(self.picks.loc[station, f"{ph.upper()}_abs_picks"]):
                    continue
                self.picks.loc[station, f"{ph.upper()}_picks_sec"] = np.float32(udt(
                    self.picks.loc[station, f"{ph.upper()}_abs_picks"]
                ) - udt(self.origin_time))

    def update_travel_times(self):
        """
        Update the travel times with respect to the current origin time.

        This function updates the travel times of each station and phase with respect
        to the current origin time of the event. It adjusts the propagation times
        by subtracting the origin time from the absolute times and update the
        `arrival_times` attribute of the event.
        """
        if not hasattr(self, "arrival_times"):
            print("Does not have an `arrival_times` attribute.")
            return
        for station in self.arrival_times.index:
            for ph in self.phases:
                self.arrival_times.loc[station, f"{ph.upper()}_tt_sec"] = udt(
                    self.arrival_times.loc[station, f"{ph.upper()}_abs_arrival_times"]
                ) - udt(self.origin_time)

    def update_aux_data_database(self, overwrite=False):
        """
        Update the auxiliary data in the database with new elements from `self.aux_data`.

        This function adds new elements from the `self.aux_data` attribute to the database
        located at `path_database`. If the element already exists in the database
        and `overwrite` is False, it skips the element. If `overwrite` is True, it
        overwrites the existing data in the database with the new data.

        Parameters
        ----------
        overwrite : bool, optional
            If True, overwrite existing data in the database. If False (default),
            skip existing elements and do not modify the database.

        Notes
        -----
        - If any error occurs during the process, the function removes the lock file and
          raises the exception.

        Raises
        ------
        Exception
            If an error occurs while updating the database, the function raises the
            exception after removing the lock file.
        """
        if not hasattr(self, "path_database"):
            print("It looks like you have created this Event instance from scratch...")
            print("Call Event.write instead.")
            return
        lock_file = os.path.splitext(self.path_database)[0] + ".lock"
        while os.path.isfile(lock_file):
            # another process is already writing in this file
            # wait a bit a check again
            sleep(0.1 + np.random.random())
        # create empty lock file
        open(lock_file, "w").close()
        try:
            with h5.File(self.path_database, mode="a") as fdb:
                if hasattr(self, "hdf5_gid"):
                    fdb = fdb[self.hdf5_gid]
                for key in self.aux_data:
                    if key in fdb["aux_data"] and not overwrite:
                        # already exists
                        continue
                    elif key in fdb["aux_data"] and overwrite:
                        # overwrite it
                        del fdb["aux_data"]
                    fdb["aux_data"].create_dataset(
                            key, data=self.aux_data[key]
                            )
        except Exception as e:
            os.remove(lock_file)
            raise (e)
        # remove lock file
        os.remove(lock_file)

    def _write(
        self,
        db_filename,
        db_path=cfg.OUTPUT_PATH,
        save_waveforms=False,
        gid=None,
        hdf5_file=None,
    ):
        """See `Event.write`.
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

    def write(
        self,
        db_filename,
        db_path=cfg.OUTPUT_PATH,
        save_waveforms=False,
        gid=None,
        hdf5_file=None,
    ):
        """
        Write the event information to an HDF5 file.

        This function writes the event information, including waveform data if
        specified, to an HDF5 file. The event information is stored in a specific
        group within the file. The function uses the `h5py` module to interact
        with the HDF5 file.

        Parameters
        ----------
        db_filename : str
            Name of the HDF5 file storing the event information.
        db_path : str, optional
            Path to the directory where the HDF5 file is located. Defaults to
            `cfg.OUTPUT_PATH`.
        save_waveforms : bool, optional
            If True, save the waveform data associated with the event in the HDF5
            file. Defaults to False.
        gid : str, optional
            Name of the HDF5 group where the event will be stored. If `gid` is None,
            the event is stored directly at the root of the HDF5 file. Defaults to
            None.
        hdf5_file : h5py.File, optional
            An opened HDF5 file object pointing directly to the subfolder of
            interest. If provided, the `db_path` parameter is ignored. Defaults to
            None.

        Notes
        -----
        - The function uses the `read_write_waiting_list` function from the `utils`
          module to handle the writing operation in a thread-safe manner. The actual
          writing is performed by the `_write` method, which is called with the
          appropriate parameters.
        - The `read_write_waiting_list` function handles the synchronization and
          queueing of write operations to avoid conflicts when multiple processes
          try to write to the same file simultaneously. The function waits for a
          lock before executing the `func` partial function, which performs the
          actual writing operation. **BUT** the queueing is still not bullet-proof.
        """
        from functools import partial
        func = partial(
                self._write,
                db_path="",
                save_waveforms=save_waveforms,
                gid=gid,
                hdf5_file=hdf5_file
                )
        utils.read_write_waiting_list(
                func, os.path.join(db_path, db_filename)
                )


    # -----------------------------------------------------------
    #            plotting method(s)
    # -----------------------------------------------------------

    def plot(
        self,
        figsize=(20, 15),
        gain=1.0e6,
        stations=None,
        ylabel=r"Velocity ($\mu$m/s)",
        plot_picks=True,
        plot_predicted_arrivals=True,
        plot_probabilities=False,
        **kwargs,
    ):
        """
        Plot the waveforms of the Event instance.

        This function plots the waveforms associated with the Event instance. The
        waveforms are plotted as subplots, with each subplot representing a station
        and component combination. The start and end times of the waveforms are
        determined to set the x-axis limits of each subplot. The picks and theoretical
        arrival times associated with the Event are overlaid on the waveforms.

        Parameters
        ----------
        figsize : tuple, optional
            The figure size in inches, specified as a tuple (width, height). Defaults
            to (20, 15).
        gain : float, optional
            The gain factor applied to the waveform data before plotting. Defaults to
            1.0e6.
        stations : list of str, optional
            The list of station names for which to plot the waveforms. If None, all
            stations associated with the Event are plotted. Defaults to None.
        ylabel : str, optional
            The label for the y-axis. Defaults to r"Velocity ($\mu$m/s).
        **kwargs
            Additional keyword arguments that are passed to the matplotlib plot
            function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance produced by this method.
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
        pick_colors = {"P": "C0", "S": "C3"}
        predicted_arrival_colors = {"P": "C4", "S": "C1"}
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
                # channel-specific num_samples
                num_samples = min(len(time), len(tr.data))
                axes[s, c].plot(
                    time[: num_samples], tr.data[: num_samples] * gain, color="k"
                )
                for ph in ["P", "S"]:
                    # plot the P-/S-wave ML probabilities
                    if plot_probabilities:
                        axb = axes[s, c].twinx()
                        ylim = axes[s, c].get_ylim()
                        axb.set_ylim(-1.05 * abs(ylim[0])/abs(ylim[1]), 1.05)
                    if plot_probabilities and hasattr(self, "probability_time_series") and (
                            sta in self.probability_time_series[ph].dropna().index
                            ):
                        selection = (
                                (self.probability_times >= time[0])
                                &
                                (self.probability_times <= time[-1])
                                )
                        axb.plot(
                                self.probability_times[selection],
                                self.probability_time_series.loc[sta, ph][selection],
                                color=pick_colors[ph],
                                lw=0.75
                                )
                    # plot the picks
                    if plot_picks and hasattr(self, "picks") and (
                        sta in self.picks[f"{ph}_abs_picks"].dropna().index
                    ):
                        for ph_pick in np.atleast_1d(self.picks.loc[sta, f"{ph}_abs_picks"]):
                            axes[s, c].axvline(
                                    np.datetime64(ph_pick), color=pick_colors[ph], lw=1.00, ls="--"
                                    )
                    # plot the theoretical arrival times
                    if plot_predicted_arrivals and hasattr(self, "arrival_times") and (sta in self.arrival_times.index):
                        ph_pick = np.datetime64(
                            self.arrival_times.loc[sta, f"{ph}_abs_arrival_times"]
                        )
                        axes[s, c].axvline(ph_pick, color=predicted_arrival_colors[ph], lw=1.25)
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
    """
    A class for template events.

    The Template class is a subclass of the Event class and is specifically designed
    for template events. It inherits all the attributes and methods of the Event
    class and adds additional functionality specific to template events.

    Parameters
    ----------
    origin_time : str or datetime
        The origin time of the template event. Can be specified as a string in
        ISO 8601 format or as a datetime object.
    moveouts : pandas.DataFrame
        The moveout table containing the travel time information for the template
        event. The moveouts table should have the following columns: 'Phase', 'Station',
        'Distance', and 'TravelTime'.
    stations : list of str
        The list of station names associated with the template event.
    phases : list of str
        The list of phase names associated with the template event.
    template_filename : str
        The filename of the template waveform file.
    template_path : str
        The path to the directory containing the template waveform file.
    latitude : float or None, optional
        The latitude of the template event in decimal degrees. If None, latitude
        information is not provided. Defaults to None.
    longitude : float or None, optional
        The longitude of the template event in decimal degrees. If None, longitude
        information is not provided. Defaults to None.
    depth : float or None, optional
        The depth of the template event in kilometers. If None, depth information
        is not provided. Defaults to None.
    sampling_rate : float or None, optional
        The sampling rate of the template waveforms in Hz. If None, the sampling
        rate is not provided. Defaults to None.
    components : list of str, optional
        The list of component names associated with the template waveforms.
        Defaults to ['N', 'E', 'Z'].
    id : str or None, optional
        The ID of the template event. If None, no ID is assigned. Defaults to None.
    """

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
        """
        Instantiate a Template object from an Event object.

        This class method creates a Template object based on an existing Event object.
        It converts the Event object into a Template object by transferring the relevant
        attributes and data.

        Parameters
        ----------
        event : Event instance
            The Event instance to convert to a Template instance.
        attach_waveforms : boolean, optional
            Specifies whether to attach the waveform data to the Template instance.
            If True, the waveform data is attached. If False, the waveform data is not
            attached. Defaults to True.

        Returns
        -------
        template : Template instance
            The Template instance based on the provided Event instance.
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
            "tid",
            "cov_mat",
            "Mw",
            "Mw_err",
            "hmax_unc",
            "hmin_unc",
            "vmax_unc",
            "az_hmax_unc"
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
        """
        Initialize a Template instance from a file.

        This class method reads a file and initializes a Template instance based on the
        content of the file. It utilizes the Event class method `read_from_file` to
        read the file and convert it into an Event instance. The Event instance is then
        converted into a Template instance using the `init_from_event` class method.

        Parameters
        ----------
        filename : str
            The name of the file to read and initialize the Template instance.
        db_path : str, optional
            The path to the directory containing the file. Defaults to cfg.INPUT_PATH.
        gid : str, optional
            The name of the hdf5 group where the file is stored. Defaults to None.

        Returns
        -------
        template : Template instance
            The Template instance initialized from the specified file.
        """
        template = cls.init_from_event(
            Event.read_from_file(filename, db_path=db_path, gid=gid),
            attach_waveforms=False,
        )
        template.n_samples = template.aux_data["n_samples"]
        template.id = str(template.aux_data["tid"])
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
        """
        Compute the distance between the template and a given location.

        This function calculates the distance between the longitude, latitude, and depth
        of the template and the longitude, latitude, and depth of a target location using
        the `two_point_distance` function from the `utils` module.

        Parameters
        ----------
        longitude : float
            The longitude of the target location.
        latitude : float
            The latitude of the target location.
        depth : float
            The depth of the target location in kilometers.

        Returns
        -------
        distance : float
            The distance, in kilometers, between the template and the target location.
        """
        from .utils import two_point_distance

        return two_point_distance(
            self.longitude, self.latitude, self.depth, longitude, latitude, depth
        )

    def n_best_SNR_stations(self, n, available_stations=None):
        """
        Adjust `self.stations` to the `n` best SNR stations.

        This function finds the `n` best stations based on signal-to-noise ratio (SNR)
        and modifies the `self.stations` attribute accordingly. The instance's properties
        will also change to reflect the new selection of stations.

        Parameters
        ----------
        n : int
            The number of best SNR stations to select.
        available_stations : list of str, default None
            The list of stations from which to search for the closest stations. If
            provided, only stations in this list will be considered. This can be used
            to exclude stations that are known to lack data.
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
        """
        Read the waveforms time series.

        This function reads the waveforms from the stored data file and initializes
        the `self.traces` attribute with the waveform data.

        Parameters
        ----------
        stations : list of str, default None
            The list of station names for which to read the waveforms. If None, all
            stations in `self.stations` will be read.
        components : list of str, default None
            The list of component names for which to read the waveforms. If None,
            all components in `self.components` will be read.

        Notes
        -----
        - The waveform data is stored in the `self.traces` attribute as an `obspy.Stream`
          object.
        - The `starttime` of each trace is set based on the origin time of the event
          and the moveout values stored in the `moveouts_win` attribute.
        - The `set_availability` method is called to update the availability of the
          waveforms for the selected stations.
        """
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
        """
        Write the Template instance to an HDF5 file.

        This function writes the Template instance to an HDF5 file specified by
        `db_filename` and `db_path`. It uses the `Event.write` method to handle the
        majority of the writing process.

        Parameters
        ----------
        db_filename : str
            Name of the HDF5 file to store the Template instance.
        db_path : str, optional
            Path to the directory where the HDF5 file will be located. Defaults to
            `cfg.OUTPUT_PATH`.
        save_waveforms : bool, optional
            Flag indicating whether to save the waveforms in the HDF5 file. Defaults
            to True.
        gid : str, optional
            Name of the HDF5 group under which the Template instance will be stored.
            If None, the Template instance will be stored directly in the root of
            the HDF5 file. Defaults to None.
        overwrite : bool, optional
            Flag indicating whether to overwrite an existing HDF5 file with the same
            name. If True, the existing file will be removed before writing. If False,
            the write operation will be skipped. Defaults to False.

        Notes
        -----
        - The `write` function sets the `where` attribute of the Template instance
          to the full path of the HDF5 file.
        - If `overwrite` is True and an existing file with the same name already
          exists, it will be removed before writing the Template instance.
        - The majority of the writing process is handled by the `Event.write` method,
          which is called with appropriate arguments.
        """
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
        """
        Build a Catalog instance from detection data.

        This function builds a Catalog instance by reading detection data from a file
        specified by `filename` and `db_path`, or by checking for an existing summary
        file. It supports reading additional attributes specified by `extra_attributes`
        and provides options to control the behavior of filling missing values with
        `fill_value` and returning a list of Event instances with `return_events`.

        Parameters
        ----------
        filename : str, optional
            Name of the detection file. If None, the standard file and folder naming
            convention will be used. Defaults to None.
        db_path : str, optional
            Name of the directory where the detection file is located. If None, the
            standard file and folder naming convention will be used. Defaults to None.
        gid : str, int, or float, optional
            If not None, this is the HDF5 group where the data will be read from.
            Defaults to None.
        extra_attributes : list of str, optional
            Additional attributes to read in addition to the default attributes
            ('longitude', 'latitude', 'depth', and 'origin_time'). Defaults to an
            empty list.
        fill_value : str, int, or float, optional
            Default value to fill missing target attributes. Defaults to np.nan.
        return_events : bool, optional
            If True, a list of Event instances will be returned. This can only be
            True if check_summary_file is set to False. Defaults to False.
        check_summary_file : bool, optional
            If True, it checks if the summary HDF5 file already exists and reads from
            it using the standard naming convention. If False, it builds the catalog
            from the detection output. Defaults to True.

        Notes
        -----
        - If `return_events` is set to True, `check_summary_file` must be set to False.
        - The `read_catalog` function first checks if the summary file exists.
        - The resulting Catalog instance is assigned to `self.catalog`, and additional
          attributes such as 'tid' and 'event_id' are set accordingly.
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
        """
        Write the summary of template characteristics to an HDF5 file.

        This function writes the summary of template characteristics, e.g. the
        characteristics of the detected events, specified in the `attributes`
        dictionary to an HDF5 file specified by `filename` and `db_path`.
        It supports overwriting existing datasets or groups with `overwrite` option.

        Parameters
        ----------
        attributes : dict
            Dictionary containing scalars, numpy arrays, dictionaries, or pandas dataframes.
            The keys of the dictionary are used to name the dataset or group in the HDF5 file.
        filename : str, optional
            Name of the detection file. If None, the standard file and folder naming
            convention will be used. Defaults to None.
        db_path : str, optional
            Name of the directory where the detection file is located. If None, the
            standard file and folder naming convention will be used. Defaults to None.
        overwrite : bool, optional
            If True, existing datasets or groups will be overwritten. If False, they
            will be skipped. Defaults to True.
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
        """
        Plot the `idx`-th detection made with this template.

        Parameters
        ----------
        idx : int
            Index of the detection to plot.
        filename : str, optional
            Name of the detection file. If None, use the standard file and folder naming
            convention. Defaults to None.
        db_path : str, optional
            Name of the directory where the detection file is located. If None, use the
            standard file and folder naming convention. Defaults to None.
        duration : float, optional
            Duration of the waveforms to plot, in seconds. Defaults to 60.0.
        phase_on_comp : dict, optional
            Dictionary specifying the phase associated with each component. The keys
            are component codes, and the values are phase codes. Defaults to
            {"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"}.
        offset_ot : float, optional
            Offset in seconds to apply to the origin time of the event when retrieving
            the waveforms. Defaults to 10.0.
        **kwargs:
            Additional keyword arguments to be passed to the `Event.plot` method.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure instance produced by this method.
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
        if kwargs.get("stations", None) is None:
            stations = event.stations
        else:
            stations = kwargs.get("stations", None)
        #stations = event.stations
        axes = fig.get_axes()
        cc, n_channels = 0.0, 0
        for s, sta in enumerate(stations):
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
        self, ax=None, annotate_axes=True, unique=False, figsize=(20, 10), **kwargs
    ):
        """
        Plot recurrence times vs detection times.

        Parameters
        ----------
        ax : plt.Axes, optional
            If not None, use this plt.Axes instance to plot the data.
        annotate_axes : bool, optional
            Whether to annotate the axes with labels. Defaults to True.
        figsize : tuple, optional
            Figure size (width, height) in inches. Defaults to (20, 10).
        **kwargs:
            Additional keyword arguments to be passed to the plt.plot function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure instance produced by this method.
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
            self.catalog.origin_time[1:] - self.catalog.origin_time[:-1]
        ).astype("timedelta64[ns]").astype("float64") / 1.0e9  # in sec
        if unique and "unique_event" in self.catalog.catalog:
            unique_event = self.catalog.catalog["unique_event"].values
            time = self.catalog.origin_time[1:][unique_event[1:]]
            rt = rt[unique_event[1:]]
        else:
            time = self.catalog.origin_time[1:]
        ax.plot(time, rt, **kwargs)
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
    """
    A class for a group of events.

    Each event is represented by a `dataset.Event` instance.

    Attributes
    ----------
    events : list
        List of `dataset.Event` instances constituting the group.
    network : `dataset.Network`
        The `Network` instance used to query consistent data across all events.
    """

    def __init__(self, events, network):
        """
        Initialize the EventGroup with a list of `dataset.Event` instances.

        Parameters
        ----------
        events : list of `dataset.Event` instances
            List of `dataset.Event` instances constituting the group.
        network : `dataset.Network` instance
            The `Network` instance used to query consistent data across all events.
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
    def read_waveforms(self, duration, time_shifted=False, progress=False, **kwargs):
        """
        Call `dataset.Event.read_waveform` with each event.

        Parameters
        ----------
        duration : float
            Duration of the waveform to read, in seconds.
        time_shifted : bool, default False
            Whether to apply time shifting to the waveforms.
        progress : bool, default False
            Whether to display progress information during waveform reading.

        Other Parameters
        ----------------
        **kwargs : dict
            Additional keyword arguments to pass to `dataset.Event.read_waveforms`.

        Returns
        -------
        None

        """
        self.time_shifted = time_shifted
        disable = np.bitwise_not(progress)
        for ev in tqdm(self.events, desc="Reading event waveforms", disable=disable):
            ev.read_waveforms(duration, time_shifted=time_shifted, **kwargs)
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
        """
        Apply Singular Value Decomposition Waveform Filtering (SVDWF) and stack the waveforms.

        Parameters
        ----------
        freqmin : float
            Minimum frequency in Hz for the bandpass filter.
        freqmax : float
            Maximum frequency in Hz for the bandpass filter.
        sampling_rate : float
            Sampling rate in Hz of the waveforms.
        expl_var : float, default 0.4
            Explained variance ratio threshold for retaining singular values.
        max_singular_values : int, default 5
            Maximum number of singular values to retain during SVDWF.
        wiener_filter_colsize : int, default None
            Size of the column blocks used for the Wiener filter.

        Returns
        -------
        None

        Notes
        -----
        See: Moreau, L., Stehly, L., Boué, P., Lu, Y., Larose, E., & Campillo, M. (2017).
        Improving ambient noise correlation functions with an SVD-based Wiener filter.
        Geophysical Journal International, 211(1), 418-426.
        """
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
    """
    A class for a group of templates.

    Each template is represented by a `dataset.Template` instance.
    """

    def __init__(self, templates, network, source_receiver_dist=True):
        """
        Initialize the TemplateGroup instance with a list of `dataset.Template` instances.

        Parameters
        ----------
        templates : list of `dataset.Template` instances
            The list of templates constituting the group.
        network : `dataset.Network` instance
            The `Network` instance used to query consistent data across all templates.
        source_receiver_dist : bool, optional
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
        """
        Initialize the TemplateGroup instance given a list of filenames.

        Parameters
        ----------
        filenames : list of str
            List of full file paths from which to instantiate the list
            of `dataset.Template` objects.
        network : `dataset.Network` instance
            The `Network` instance used to query consistent data across all templates.
        gids : list of str, optional
            If provided, this should be a list of group IDs where the
            template data is stored in their HDF5 files.
        
        Returns
        -------
        template_group : TemplateGroup instance
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
        """
        Keep templates inside the requested geographic bounds.

        Parameters
        ----------
        lon_min : float
            Minimum longitude in decimal degrees.
        lon_max : float
            Maximum longitude in decimal degrees.
        lat_min : float
            Minimum latitude in decimal degrees.
        lat_max : float
            Maximum latitude in decimal degrees.
        inplace : bool, default False
            If True, perform the operation in-place by modifying the existing TemplateGroup.
            If False, create and return a new TemplateGroup with the filtered templates.

        Returns
        -------
        template_group : TemplateGroup instance
            The TemplateGroup instance containing templates within the specified geographic bounds.
            This is returned only when inplace=False.
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
        """
        Compute the template-pairwise distances in kilometers.

        This method calculates the distances between all pairs of templates in the TemplateGroup.
        The distances are computed based on the longitude, latitude, and depth of each template.

        The computed distances are then read at the `self.intertemplate_dist` property
        of the TemplateGroup instance.

        Returns
        -------
        None
            The computed distances can be accessed at the `self.intertemplate_dist`
            property of the TemplateGroup.
        """
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
        """
        Compute the length of the uncertainty ellipsoid in the inter-template direction.

        This method calculates the length of the uncertainty ellipsoid in the
        inter-template direction for each pair of templates in the TemplateGroup.
        The inter-template direction is defined as the direction linking two templates.
        The length of the uncertainty ellipsoid is computed based on the covariance
        matrix of each template.

        The computed errors can be read at the `self.dir_errors` property of
        the TemplateGroup. It is a pandas DataFrame with dimensions
        (n_templates, n_templates), where each entry represents the length of
        the uncertainty ellipsoid in kilometers.

        Example: `self.dir_errors.loc[tid1, tid2]` is the width of template `tid1`'s
        uncertainty ellipsoid in the direction of template `tid2`.

        Returns
        -------
        None
            The computed errors can be read at the `self.dir_errors` property
            of the TemplateGroup instance.
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
        """
        Compute the separation between uncertainty ellipsoids in the inter-template direction.

        This method calculates the separation between the uncertainty
        ellipsoids in the inter-template direction for each pair of templates
        in the TemplateGroup instance. The separation is computed as the difference
        between the inter-template distances and the directional errors. It can
        be negative if the uncertainty ellipsoids overlap.

        The computed separations can be read at the `self.ellipsoid_dist`
        property of the TemplateGroup instance. It is a pandas DataFrame with
        dimensions (n_templates, n_templates), where each entry represents
        the separation between the uncertainty ellipsoids in kilometers.

        Returns
        -------
        None
            The computed separations can be read at the `self.ellipsoid_dist`
            property of the TemplateGroup instance.
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
        output_filename="intertp_cc.h5"
    ):
        """
        Compute the pairwise template cross-correlations (CCs).

        This method computes the pairwise cross-correlations (CCs) between
        templates in the TemplateGroup.

        The CCs measure the similarity between the waveforms of different templates.
        CCs are computed only for template pairs that are separated by less than
        `distance_threhold`. The number of closest stations to include and the maximum
        lag for searching the maximum CC are configurable parameters.

        Parameters
        ----------
        distance_threshold : float, default to 5.0
            The distance threshold, in kilometers, between two uncertainty
            ellipsoids under which the CC is computed.
        n_stations : int, default to 10
            The number of closest stations to each template that are used in
            the computation of the average CC.
        max_lag : int, default to 10
            The maximum lag, in samples, allowed when searching for the maximum
            CC on each channel. This parameter accounts for small discrepancies
            in windowing that could occur for two templates highly similar but
            associated with slightly different locations.
        save_cc : bool, default to False
            If True, save the inter-template CCs in the same folder as the first
            template (`self.templates[0]`) with filename 'output_filename'.
        compute_from_scratch : bool, default to False
            If True, force the computation of the inter-template CCs from scratch.
            This is useful when the user knows that the computation is faster than
            reading a potentially large file.
        device : str, default to 'cpu'
            The device to use for the computation. Can be either 'cpu' or 'gpu'.
        progress : bool, default to False
            If True, print a progress bar using `tqdm`.
        output_filename : str, default to 'intertp_cc.h5'
            The filename to use when saving the inter-template CCs.

        Returns
        -------
        None
            The computed inter-template CCs can be read at the `self.intertemplate_cc`
            property of the TemplateGroup instance.
        """
        import fast_matched_filter as fmf  # clearly need some optimization

        disable = np.bitwise_not(progress)

        # try reading the inter-template CC from db
        db_path, db_filename = os.path.split(self.templates[0].where)
        cc_fn = os.path.join(db_path, output_filename)
        if not compute_from_scratch and os.path.isfile(cc_fn):
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
            #self.n_closest_stations(n_stations)
            print("Computing the similarity matrix...")
            # format arrays for FMF
            data_arr = self.waveforms_arr.copy()
            template_arr = self.waveforms_arr[..., max_lag:-max_lag]
            moveouts_arr = np.zeros(self.waveforms_arr.shape[:-1], dtype=np.int32)
            intertp_cc = np.zeros(
                (self.n_templates, self.n_templates), dtype=np.float32
            )
            n_network_stations, n_components = moveouts_arr.shape[1:]
            # use FMF on one template at a time against all others
            for t, template in tqdm(
                enumerate(self.templates), desc="Inter-tp CC", disable=disable
            ):
                weights = np.zeros(template_arr.shape[:-1], dtype=np.float32)
                # select the `n_stations` closest stations
                # apply similar approach than Event.n_closest_stations
                station_pool = template.network_stations[template.availability]
                closest_stations = (
                        template.source_receiver_dist\
                                .loc[station_pool].sort_values().index[:n_stations]
                        )
                # make sure we return a n_stations-vector
                if len(closest_stations) < n_stations:
                    missing = n_stations - len(closest_stations)
                    closest_stations = np.hstack(
                        (
                            closest_stations,
                            template.source_receiver_dist.drop(closest_stations, axis="rows")
                            .sort_values()
                            .index[:missing],
                        )
                    )
                for s, sta in enumerate(template.network_stations):
                    if sta in closest_stations:
                        weights[:, s, :] = np.int32(
                                template.availability_per_cha.loc[sta].values
                                )
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
        """
        Save the inter-template correlation coefficients to a file.

        This method saves the inter-template correlation coefficients computed
        by the `compute_intertemplate_cc` method to a specified file.

        Parameters
        ----------
        intertp_cc : pd.DataFrame
            The inter-template correlation coefficients computed by the
            `compute_intertemplate_cc` method.
        fullpath : str
            The full path to the output file where the correlation coefficients
            will be saved.

        Returns
        -------
        None
        """
        with h5.File(fullpath, mode="w") as f:
            f.create_dataset("tids", data=np.int32(intertp_cc.columns))
            f.create_dataset("intertp_cc", data=np.float32(intertp_cc.values))

    @staticmethod
    def _read_intertp_cc(fullpath):
        """Read inter-template correlation coefficients from file.

        Parameters
        -----------
        fullpath : string
            Full path to output file.

        Returns
        --------
        intertp_cc : pd.DataFrame
            The inter-template CC in a pd.DataFrame.
        """
        with h5.File(fullpath, mode="r") as f:
            tids = f["tids"][()]
            intertp_cc = f["intertp_cc"][()]
        return pd.DataFrame(index=tids, columns=tids, data=intertp_cc)

    def read_waveforms(self, n_threads=1, progress=False):
        """
        Read waveforms for all templates in the TemplateGroup.

        This method calls `template.read_waveforms` for each `Template`
        instance in `self.templates`.

        Parameters
        ----------
        n_threads : int, optional
            The number of threads to use for parallel execution. If set to 1
            (default), the waveforms will be read sequentially. If set to a
            value greater than 1, the waveforms will be read in parallel using
            multiple threads. If set to 0, None, or "all", the method will use all
            available CPUs for parallel execution.
        progress : bool, optional
            If True, a progress bar will be displayed during the waveform reading process.

        Returns
        -------
        None
        """
        disable = np.bitwise_not(progress)
        if n_threads != 1:
            from concurrent.futures import ThreadPoolExecutor
            # cannot use tqdm with parallel execution
            disable = True
            if n_threads in [0, None, "all"]:
                # n_threads = None means use all CPUs
                n_threads = None
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                executor.map(
                        lambda tp: tp.read_waveforms(
                            stations=self.stations, components=self.components
                            ),
                        self.templates
                        )
        else:
            for tp in tqdm(self.templates, desc="Reading waveforms", disable=disable):
                tp.read_waveforms(stations=self.stations, components=self.components)
        self._remember("read_waveforms")

    def set_network_to_template_map(self):
        """
        Compute the map between network arrays and template data.

        Template data are broadcasted to fit the dimensions of the network
        arrays. This method computes the `network_to_template_map` that tells
        which stations and channels are used on each template.
        For example:
        `network_to_template_map[t, s, c] = False`
        means that station s and channel c are not used on template t.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _network_to_template_map = np.zeros(
            (self.n_templates, self.network.n_stations, self.network.n_components),
            dtype=bool,
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
        """
        Adjust `self.stations` on each template to the `n` best SNR stations.

        This method calls `template.n_best_SNR_stations` for each Template
        instance in `self.templates`. For each template, it finds the `n` best
        SNR stations and modify `template.stations` accordingly.

        Parameters
        ----------
        n : int
            The `n` best SNR stations.
        available_stations : list of str, optional
            The list of stations from which we search the closest stations.
            If some stations are known to lack data, the user
            may choose to not include these in the best SNR stations.
            Defaults to None.

        Returns
        -------
        None
        """
        for tp in self.templates:
            tp.n_best_SNR_stations(n, available_stations=available_stations)
        if hasattr(self, "_network_to_template_map"):
            del self._network_to_template_map

    def n_closest_stations(self, n, available_stations=None):
        """
        Adjust `self.stations` on each template to the `n` closest stations.

        This method calls `template.n_closest_stations` for each Template
        instance in `self.templates`. For each template, it finds the `n`
        closest stations and modify `template.stations` accordingly.

        Parameters
        ----------
        n : int
            The `n` closest stations.
        available_stations : list of str, optional
            The list of stations from which we search the closest stations.
            If some stations are known to lack data, the user
            may choose to not include these in the closest stations.
            Defaults to None.

        Returns
        -------
        None

        """
        for tp in self.templates:
            tp.n_closest_stations(n, available_stations=available_stations)
        if hasattr(self, "_network_to_template_map"):
            del self._network_to_template_map

    def read_catalog(
        self, extra_attributes=[], fill_value=np.nan, progress=False, **kwargs
    ):
        """
        Build a catalog from all templates' detections.

        Parameters
        ----------
        extra_attributes : list of str, optional
            Additional attributes to read in addition to the default attributes of
            'longitude', 'latitude', 'depth', and 'origin_time'.
        fill_value : str, int, or float, optional
            Default value to use if the target attribute does not exist in a template's
            detections.
        progress : bool, optional
            If True, display a progress bar during the operation.
        **kwargs
            Additional keyword arguments to be passed to the `read_catalog`
            method of each template.

        Returns
        -------
        None

        Notes
        -----
        This method reads the detection information from each template in the TemplateGroup
        and builds a catalog that contains the combined detections from all templates.

        The resulting catalog will include the default attributes of 'longitude', 'latitude',
        'depth', and 'origin_time', as well as any additional attributes specified in
        the `extra_attributes` parameter.

        If a template does not have a specific attribute in its detections, it will be assigned
        the `fill_value` for that attribute.

        The `progress` parameter controls whether a progress bar is displayed during the operation.
        Setting it to True provides visual feedback on the progress of reading the catalogs.
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
        """
        Search for events detected by multiple templates and flag them.

        Parameters
        ----------
        n_closest_stations : int, optional
            The number of stations closest to each template used in the calculation
            of the average cross-correlation (cc).
        dt_criterion : float, optional
            The time interval, in seconds, under which two events are examined for redundancy.
        distance_criterion : float, optional
            The distance threshold, in kilometers, between two uncertainty ellipsoids
            under which two events are examined for redundancy.
        speed_criterion : float, optional
            The speed criterion, in km/s, below which the inter-event time and inter-event
            distance can be explained by errors in origin times and a reasonable P-wave speed.
        similarity_criterion : float, optional
            The template similarity threshold, in terms of average cc, over which two events
            are examined for redundancy. The default value of -1 means that similarity is not
            taken into account.
        progress : bool, optional
            If True, print progress bar with `tqdm`.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None

        Notes
        -----
        This method creates a new entry in each Template instance's catalog where
        unique events are flagged:
        `self.templates[i].catalog.catalog['unique_events']`
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
        unique_event = np.ones(n_events, dtype=bool)
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
                            #| (speed_diff < speed_criterion)
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
        Plot the idx-th event in `self.catalog.catalog`.

        This method identifies which template detected the idx-th event
        and calls `template.plot_detection` with the appropriate arguments.

        Parameters
        -----------
        idx : int
            Event index in `self.catalog.catalog`.

        Returns
        ---------
        fig : matplotlib.pyplot.Figure 
            The figure showing the detected event.
        """
        tid, evidx = self.catalog.catalog.index[idx].split(".")
        tt = self.tindexes.loc[int(tid)]
        print("Plotting:")
        print(self.catalog.catalog.iloc[idx])
        fig = self.templates[tt].plot_detection(int(evidx), **kwargs)
        return fig

    def plot_recurrence_times(self, figsize=(15, 7), progress=False, **kwargs):
        """
        Plot recurrence times vs detection times, template-wise.

        This method calls `template.plot_recurrence_times` for every
        Template instance in `self.templates`. Thus, the recurrence time
        is defined template-wise.

        Parameters
        -----------
        figsize : tuple of floats, optional 
            Size in inches of the figure (width, height).
            Defaults to (15, 7).
        progress : boolean, optional
            If True, print progress bar with `tqdm`. Defaults to False.
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
    """
    A modification of the Event class for stacked events."""

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
        """
        Initialize an Event instance with basic attributes.

        Parameters
        ----------
        stacked_traces : obspy.Stream
            Traces with the stacked waveforms.
        moveouts : numpy.ndarray
            Moveouts, in seconds, for each station and each phase.
            Shape: (n_stations, n_phases)
        stations : list of str
            List of station names corresponding to `moveouts`.
        phases : list of str
            List of phase names corresponding to `moveouts`.
        latitude : float, optional
            Event latitude.
        longitude : float, optional
            Event longitude.
        depth : float, optional
            Event depth.
        sampling_rate : float, optional
            Sampling rate (Hz) of the waveforms. It should be different from None
            only if you plan on reading preprocessed data with a fixed sampling rate.
        components : list of str, optional
            List of the components to use in reading and plotting methods.
            Default: ['N', 'E', 'Z']
        component_aliases : dict, optional
            Each entry of the dictionary is a list of strings.
            `component_aliases[comp]` is the list of all aliases used for the same
            component 'comp'. For example, `component_aliases['N'] = ['N', '1']`
            means that both the 'N' and '1' channels will be mapped to the Event's
            'N' channel.
        aux_data : dict, optional
            Dictionary with auxiliary data. Note that aux_data['phase_on_comp{cp}']
            is necessary to call `self.read_waveforms`.
        id : str, optional
            Identifying label.
        filtered_data : numpy.ndarray, optional
            The event waveforms filtered by the SVDWF technique.
            Shape: (n_events, n_stations, n_components, n_samples)

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
        """
        Read waveform data.

        Parameters
        ----------
        duration : float
            Duration, in seconds, of the extracted time windows.
        phase_on_comp : dict, optional
            Dictionary defining which seismic phase is extracted on each component.
            For example, `phase_on_comp['N']` gives the phase that is extracted on the
            north component.
        offset_phase : dict, optional
            Dictionary defining when the time window starts with respect to the pick.
            A positive offset means the window starts before the pick. Not used if
            `time_shifted` is False.
        time_shifted : bool, optional
            If True, the moveouts are used to extract time windows from specific seismic
            phases. If False, windows are simply extracted with respect to the origin time.
        offset_ot : float, optional
            Only used if `time_shifted` is False. Time, in seconds, taken before `origin_time`.

        Returns
        -------
        None

        Notes
        -----
        The waveforms are read from `self.stacked_waveforms` and are formatted as
        obspy.Stream instances that populate the `self.traces` attribute.
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

    def pick_PS_phases_family_mode(
        self,
        duration,
        threshold_P=0.60,
        threshold_S=0.60,
        mini_batch_size=126,
        phase_on_comp={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "P"},
        upsampling=1,
        downsampling=1,
        ml_model_name="original",
        ml_model=None,
        **kwargs,
    ):
        """
        Use PhaseNet (Zhu et al., 2019) to pick P and S waves.

        This method picks P- and S-wave arrivals on every 3-component seismogram
        in `self.filtered_data`. Thus, potentially many picks can be found on every
        station and all of them returned in `self.picks`.

        Parameters
        ----------
        duration : float
            Duration, in seconds, of the time window to process to search
            for P and S wave arrivals.
        threshold_P : float, optional
            Threshold on PhaseNet's probabilities to trigger the identification
            of a P-wave arrival. Default: 0.60
        threshold_S : float, optional
            Threshold on PhaseNet's probabilities to trigger the identification
            of an S-wave arrival. Default: 0.60
        mini_batch_size : int, optional
            Number of traces processed in a single batch by PhaseNet.
            This shouldn't have to be tuned. Default: 126
        phase_on_comp : dict, optional
            Dictionary defining which seismic phase is extracted on each component.
            For example, `phase_on_comp['N']` gives the phase that is extracted on
            the north component.
        upsampling : int, optional
            Upsampling factor applied before calling PhaseNet. Default: 1
        downsampling : int, optional
            Downsampling factor applied before calling PhaseNet. Default: 1

        Returns
        -------
        None

        Notes
        -----
        - PhaseNet must be used with 3-comp data.
        - If `self.filtered_data` does not exist, `self.pick_PS_phases` is used
        on the stacked traces.
        """

        from torch import no_grad, from_numpy

        if ml_model is None:
            import seisbench.models as sbm
            ml_model = sbm.PhaseNet.from_pretrained(ml_model_name)
            ml_model.eval()

        ml_p_index = kwargs.get("ml_P_index", 1)
        ml_s_index = kwargs.get("ml_S_index", 2)

        if kwargs.get("read_waveforms", True):
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

                data_arr = resample_poly(data_arr, upsampling, downsampling, axis=-1)
                # momentarily update samping_rate
                sampling_rate0 = float(self.sampling_rate)
                self.sampling_rate = self.sr * upsampling / downsampling
            num_events, num_stations = data_arr.shape[:2]
            num_traces = num_events * num_stations
            data_arr_n = utils.normalize_batch(
                    data_arr.reshape(num_traces, data_arr.shape[2], data_arr.shape[3])
                    )
            closest_pow2 = int(np.log2(data_arr_n.shape[-1])) + 1
            diff = 2**closest_pow2 - data_arr_n.shape[-1]
            left = diff//2
            right = diff//2 + diff%2
            data_arr_n = np.pad(
                    data_arr_n,
                    ((0, 0), (0, 0), (left, right)),
                    mode="reflect"
                    )
            with no_grad():
                ml_probas = ml_model(
                        from_numpy(data_arr_n).float()
                        )
                ml_probas = ml_probas.detach().numpy()
            # find picks and sotre in dictionaries
            picks = {}
            picks["P_picks"] = {}
            picks["P_proba"] = {}
            picks["S_picks"] = {}
            picks["S_proba"] = {}
            for s, sta in enumerate(self.stations):
                picks["P_proba"][sta], picks["P_picks"][sta] = [], []
                picks["S_proba"][sta], picks["S_picks"][sta] = [], []
                for n in range(num_events):
                    tr_idx = s * num_events + n
                    P_proba, P_pick = utils.trigger_picks(
                            ml_probas[tr_idx, ml_p_index, left:-right], threshold_P, 
                            )
                    picks["P_proba"][sta].append(P_proba)
                    picks["P_picks"][sta].append(P_pick)
                    S_proba, S_pick = utils.trigger_picks(
                            ml_probas[tr_idx, ml_s_index, left:-right], threshold_S, 
                            )
                    picks["S_proba"][sta].append(S_proba)
                    picks["S_picks"][sta].append(S_pick)
                picks["P_proba"][sta] = np.atleast_1d(
                        np.hstack(picks["P_proba"][sta])
                        )
                picks["S_proba"][sta] = np.atleast_1d(
                        np.hstack(picks["S_proba"][sta])
                        )
                picks["P_picks"][sta] = np.atleast_1d(
                        np.hstack(picks["P_picks"][sta])
                        )
                picks["S_picks"][sta] = np.atleast_1d(
                        np.hstack(picks["S_picks"][sta])
                        )

            # format picks in pandas DataFrame
            pandas_picks = {"stations": self.stations}
            for ph in ["P", "S"]:
                rel_picks_sec = np.zeros(len(self.stations), dtype=object)
                proba_picks = np.zeros(len(self.stations), dtype=object)
                abs_picks = np.zeros(len(self.stations), dtype=object)
                for s, sta in enumerate(self.stations):
                    if sta in picks[f"{ph}_picks"].keys():
                        rel_picks_sec[s] = np.float32(picks[f"{ph}_picks"][sta]) / self.sr
                        proba_picks[s] = np.float32(picks[f"{ph}_proba"][sta])
                        abs_picks[s] = np.asarray([
                            np.datetime64(
                                self.traces.select(station=sta)[0].stats.starttime
                                +
                                rel_pick,
                                "ms"
                                ) for rel_pick in rel_picks_sec[s]
                        ])
                pandas_picks[f"{ph}_picks_sec"] = rel_picks_sec
                pandas_picks[f"{ph}_probas"] = proba_picks
                pandas_picks[f"{ph}_abs_picks"] = abs_picks

            self.picks = pd.DataFrame(pandas_picks)
            self.picks.set_index("stations", inplace=True)
            #self.picks.replace(0.0, np.nan, inplace=True)
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
