import os

from .config import cfg
from . import dataset
from . import utils

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from obspy.core import UTCDateTime as udt

# -------------------------------------------------------
#       Many functions will disappear soon, replaced
#       by plotting methods of new classes introduced in dataset.py
# -------------------------------------------------------


def plot_template(
    idx,
    db_path_T="template_db_2/",
    db_path=cfg.INPUT_PATH,
    n_stations=10,
    stations=None,
    mv_view=True,
    show=True,
):
    # ---------------------------
    font = {"family": "sans-serif", "weight": "normal", "size": 14}
    plt.rc("font", **font)
    # ---------------------------
    template = dataset.Template(
        "template{:d}".format(idx), db_path_T, db_path=db_path, attach_waveforms=True
    )
    if hasattr(template, "loc_uncertainty"):
        uncertainty_label = r"($\Delta r$ = {:.2f} km)".format(
            template.location_uncertainty
        )
    elif hasattr(template, "cov_mat"):
        uncertainty_label = (
            r"($\Delta X$={:.2f}km, $\Delta Y$={:.2f}km, $\Delta Z$={:.2f}km)".format(
                np.sqrt(template.cov_mat[0, 0]),
                np.sqrt(template.cov_mat[1, 1]),
                np.sqrt(template.cov_mat[2, 2]),
            )
        )
    else:
        uncertainty_label = ""
    if stations is not None:
        template.subnetwork(stations)
        n_stations = len(stations)
    else:
        # select the n_stations closest stations
        template.n_closest_stations(n_stations)
    sta = list(template.stations)
    n_stations = min(n_stations, len(sta))
    # sta.sort()
    n_components = len(template.channels)
    plt.figure(
        "template_{:d}_from_{}".format(idx, db_path + db_path_T), figsize=(18, 9)
    )
    if mv_view:
        MVs = np.column_stack(
            [template.s_moveouts, template.s_moveouts, template.p_moveouts]
        )
        MVs -= MVs.min()
        time = (
            np.arange(template.traces[0].data.size + MVs.max()) / template.sampling_rate
        )
    else:
        time = np.arange(template.traces[0].data.size) / template.sampling_rate
    for s in range(n_stations):
        for c in range(n_components):
            ax = plt.subplot(n_stations, n_components, s * n_components + c + 1)
            lab = "{}.{}".format(sta[s], template.channels[c])
            if mv_view:
                id1 = MVs[s, c]
                id2 = id1 + template.traces[0].data.size
                plt.plot(
                    time[id1:id2],
                    template.traces.select(station=sta[s])[c].data,
                    label=lab,
                )
                if c < 2:
                    plt.axvline(time[int((id1 + id2) / 2)], lw=2, ls="--", color="k")
                else:
                    plt.axvline(time[id1] + 1.0, lw=2, ls="--", color="k")
            else:
                plt.plot(
                    time, template.traces.select(station=sta[s])[c].data, label=lab
                )
            plt.xlim((time[0], time[-1]))
            plt.yticks([])
            if c < 2:
                plt.legend(
                    loc="upper left", frameon=False, handlelength=0.1, borderpad=0.0
                )
            else:
                plt.legend(
                    loc="upper right", frameon=False, handlelength=0.1, borderpad=0.0
                )
            if s == n_stations - 1:
                plt.xlabel("Time (s)")
            else:
                plt.xticks([])
    plt.subplots_adjust(bottom=0.07, top=0.94, hspace=0.04, wspace=0.12)
    plt.suptitle(
        "Template {:d}, location: {:.2f}$^{{\mathrm{{o}}}}$E,"
        "{:.2f}$^{{\mathrm{{o}}}}$N,{:.2f}km {}".format(
            template.template_idx,
            template.longitude,
            template.latitude,
            template.depth,
            uncertainty_label,
        ),
        fontsize=16,
    )
    if show:
        plt.show()


def plot_detection_matrix(
    X, datetimes=None, stack=None, title=None, ax=None, show=True, **kwargs
):

    kwargs["time_min"] = kwargs.get("time_min", None)
    kwargs["time_max"] = kwargs.get("time_max", None)
    kwargs["text_offset"] = kwargs.get("text_offset", 0.1)
    kwargs["text_size"] = kwargs.get("text_size", plt.rcParams["font.size"])
    kwargs["datetime_format"] = kwargs.get("datetime_format", "%Y,%m,%d--%H:%M:%S")

    if datetimes is not None:
        # reorder X
        new_order = np.argsort(datetimes)
        X = X[new_order, :]
        datetimes = datetimes[new_order]
    n_detections = X.shape[0]
    if ax is None:
        fig = plt.figure("detection_matrix", figsize=(18, 9))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ax.set_title(title)
    time = np.linspace(0.0, X.shape[-1] / cfg.SAMPLING_RATE_HZ, X.shape[-1])
    time_min = kwargs["time_min"] if kwargs["time_min"] is not None else time.min()
    time_max = kwargs["time_max"] if kwargs["time_max"] is not None else time.max()
    time -= time_min
    if stack is not None:
        offset = 2.0
        ax.plot(time, stack, color="C3", label="SVDWF Stack")
    else:
        offset = 0.0
    for i in range(n_detections):
        label = "Individual Events" if i == 0 else ""
        ax.plot(time, utils.max_norm(X[i, :]) + offset, lw=0.75, color="k", label=label)
        if datetimes is not None:
            plt.text(
                0.50,
                offset + kwargs["text_offset"],
                udt(datetimes[i]).strftime(kwargs["datetime_format"]),
                bbox={"facecolor": "white", "alpha": 0.75},
                fontsize=kwargs["text_size"],
            )
        offset += 2.0
    ax.set_ylabel("Offset normalized amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0.0, time_max - time_min)
    ax.legend(loc="upper right")
    plt.subplots_adjust(top=0.96, bottom=0.06)
    if show:
        plt.show()
    return fig


def plot_catalog(
    tids=None,
    db_path_T=None,
    db_path_M=None,
    catalog=None,
    ax=None,
    remove_multiples=True,
    scat_kwargs={},
    cmap=None,
    db_path=cfg.INPUT_PATH,
):

    if cmap is None:
        try:
            import colorcet as cc

            cmap = cc.cm.bjy
        except Exception as e:
            print(e)
            cmap = "viridis"

    # ------------------------------------------------
    #        Scattering plot kwargs
    scat_kwargs["edgecolor"] = scat_kwargs.get("edgecolor", "k")
    scat_kwargs["linewidths"] = scat_kwargs.get("linewidths", 0.5)
    scat_kwargs["s"] = scat_kwargs.get("s", 10)
    scat_kwargs["zorder"] = scat_kwargs.get("zorder", 0)
    # ------------------------------------------------

    if catalog is None:
        # if catalog is None, tids, db_path_T and db_path_M
        # should be given
        # ------------------
        # compile detections from these templates in a single
        # earthquake catalog
        catalog_filenames = [f"multiplets{tid}catalog.h5" for tid in tids]
        AggCat = dataset.AggregatedCatalogs(
            filenames=catalog_filenames, db_path_M=db_path_M, db_path=db_path
        )
        AggCat.read_data(items_in=["origin_times", "location", "unique_events"])
        catalog = AggCat.flatten_catalog(
            attributes=["origin_times", "latitude", "longitude", "depth"],
            unique_events=True,
        )
    cNorm = Normalize(vmin=catalog["latitude"].min(), vmax=catalog["latitude"].max())
    scalar_map = ScalarMappable(norm=cNorm, cmap=cmap)
    scalar_map.set_array([])

    # plot catalog
    if ax is None:
        fig = plt.figure("earthquake_catalog", figsize=(18, 9))
        ax = fig.add_subplot(111)
    else:
        # use the user-provided axis
        fig = ax.get_figure()
    ax.set_title("{:d} events".format(len(catalog["origin_times"])))
    ax.set_xlabel("Calendar Time")
    ax.set_ylabel("Longitude")
    times = np.array(
        [str(udt(time)) for time in catalog["origin_times"]], dtype="datetime64"
    )
    ax.scatter(
        times,
        catalog["longitude"],
        color=scalar_map.to_rgba(catalog["latitude"]),
        rasterized=True,
        **scat_kwargs,
    )
    ax.set_xlim(times.min(), times.max())

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="2%", pad=0.08)
    plt.colorbar(scalar_map, cax, orientation="vertical", label="Latitude")
    return fig


# ---------------------------------------------------------------
#                Utils for maps
# ---------------------------------------------------------------


def initialize_map(
    map_longitudes,
    map_latitudes,
    map_axis=None,
    seismic_stations=None,
    text_size=14,
    markersize=10,
    topography_file=None,
    path_topo="",
    faults=None,
    right_labels=False,
    left_labels=True,
    bottom_labels=True,
    top_labels=False,
    **kwargs,
):
    """Initialize map instance with Cartopy."""
    import cartopy as ctp

    kwargs["topo_alpha"] = kwargs.get("topo_alpha", 0.30)
    kwargs["downsample_faults"] = kwargs.get("downsample_faults", True)
    kwargs["shaded_topo"] = kwargs.get("shaded_topo", True)
    kwargs["topo_cmap"] = kwargs.get("topo_cmap", "gray")
    kwargs["topo_cnorm"] = kwargs.get("topo_cnorm", None)
    kwargs["fault_zorder"] = kwargs.get("fault_zorder", 1.01)
    figsize = kwargs.get("figsize", (15, 15))

    map_corners = [
        map_longitudes[0],
        map_latitudes[0],
        map_longitudes[1],
        map_latitudes[1],
    ]

    data_coords = ctp.crs.PlateCarree()
    if map_axis is None:
        # projection = ctp.crs.PlateCarree()
        projection = ctp.crs.Mercator(
            central_longitude=sum(map_longitudes) / 2.0,
            min_latitude=map_latitudes[0],
            max_latitude=map_latitudes[1],
        )
        fig = plt.figure(kwargs.get("figname", "map"), figsize=figsize)
        map_axis = fig.add_subplot(111, projection=projection)
        map_axis.set_rasterization_zorder(1)
        map_axis.set_extent(
            [map_longitudes[0], map_longitudes[1], map_latitudes[0], map_latitudes[1]],
            crs=data_coords,
        )

    RES = "10m"

    if topography_file is not None:
        # -----------------------
        # get topography
        import netCDF4

        with netCDF4.Dataset(os.path.join(path_topo, topography_file), "r") as f:
            if "z" in f.variables:
                topo = f.variables["z"][:].data
            elif "Band1" in f.variables:
                topo = f.variables["Band1"][:].data
            if "lon" in f.variables:
                lon_topo = f.variables["lon"][:].data
            elif "x" in f.variables:
                lon_topo = f.variables["x"][:].data
            if "lat" in f.variables:
                lat_topo = f.variables["lat"][:].data
            elif "y" in f.variables:
                lat_topo = f.variables["y"][:].data
        # select relevant area
        selected_lon = np.where(
            (lon_topo >= map_longitudes[0]) & (lon_topo <= map_longitudes[1])
        )[0]
        selected_lat = np.where(
            (lat_topo >= map_latitudes[0]) & (lat_topo <= map_latitudes[1])
        )[0]
        lon_topo = lon_topo[selected_lon]
        lat_topo = lat_topo[selected_lat]
        topo = topo[selected_lat, :]
        topo = topo[:, selected_lon]
        # make sure to take these arrays in ascending lons and lats
        ascending_lon = np.argsort(lon_topo)
        ascending_lat = np.argsort(lat_topo)
        lon_topo, lat_topo = lon_topo[ascending_lon], lat_topo[ascending_lat]
        topo = topo[ascending_lat, :]
        topo = topo[:, ascending_lon]
        if kwargs["shaded_topo"]:
            # get topography gradient
            grad_x, grad_y = np.gradient(topo)
            slope = np.pi / 2.0 - np.arctan(np.sqrt(grad_x**2 + grad_y**2))
            aspect = np.arctan2(grad_x, grad_y)
            altitude = np.pi / 4.0  # sun angle
            azimuth = np.pi / 2.0  # sun direction
            topo = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(
                slope
            ) * np.cos((azimuth - np.pi / 2.0) - aspect)
        # levels_topo = np.linspace(-500., topo.max(), 20)
        # print('Plot the topography.')
        # transform data
        # fast version
        lon_g, lat_g = np.meshgrid(lon_topo, lat_topo, indexing="xy")
        trans_data = map_axis.projection.transform_points(
            data_coords, lon_g, lat_g, topo
        )
        X = trans_data[..., 0]
        Y = trans_data[..., 1]
        Z = trans_data[..., 2]
        map_axis.imshow(
            Z,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap=kwargs["topo_cmap"],
            norm=kwargs["topo_cnorm"],
            origin="lower",
            alpha=kwargs["topo_alpha"],
            interpolation="bilinear",
            zorder=-1,
        )

    # ------------ DRAW MERIDIANS AND PARALLELS --------------
    LON0 = (int(map_longitudes[0] / 0.5) + 1.0) * 0.5
    LON1 = (int(map_longitudes[1] / 0.5) + 1.0) * 0.5
    # lon_ticks = np.arange(map_longitudes[0], map_longitudes[1]+0.5, 0.5)
    lon_ticks = np.arange(LON0 - 0.5, LON1 + 0.5, 0.5)
    LAT0 = (int(map_latitudes[0] / 0.5) + 1.0) * 0.5
    LAT1 = (int(map_latitudes[1] / 0.5) + 1.0) * 0.5
    # lat_ticks = np.arange(map_latitudes[0], map_latitudes[1]+0.5, 0.5)
    lat_ticks = np.arange(LAT0 - 0.5, LAT1 + 0.5, 0.5)
    # --------------------------------------------------------

    # ----------- DRAW BORDERS -------------
    brds = ctp.feature.BORDERS
    # --------------------------------------

    gl = map_axis.gridlines(
        draw_labels=True, linewidth=1, alpha=0.5, color="k", linestyle="--"
    )
    gl.right_labels = right_labels
    gl.left_labels = left_labels
    gl.top_labels = top_labels
    gl.bottom_labels = bottom_labels

    if kwargs.get("coastlines", True):
        # print('Add coast lines.')
        map_axis.add_feature(
            ctp.feature.GSHHSFeature(
                scale="full", levels=kwargs.get("coastline_levels", [1, 2]), zorder=0.49
            )
        )  # , rasterized=True))
    # comment this to save time on plotting high-resolution oceans
    if kwargs.get("oceans", False):
        oceans = ctp.feature.OCEAN
        map_axis.add_feature(
            ctp.feature.NaturalEarthFeature(
                category=oceans.category,
                name=oceans.name,
                scale="10m",
                facecolor="#0076b482",
            )
        )

    if faults is not None:
        map_axis.add_geometries(
            faults["geometry"], crs=data_coords, facecolor="none", edgecolor="k"
        )
    # plot optional elements
    props = dict(
        boxstyle="round", facecolor="white", edgecolor=None, alpha=0.6, pad=0.2
    )

    # plot seismic stations
    if seismic_stations is not None:
        # print('Add seismic stations.')
        for s in range(len(seismic_stations["stations"])):
            if (
                (seismic_stations["longitude"][s] > map_longitudes[1])
                or (seismic_stations["longitude"][s] < map_longitudes[0])
                or (seismic_stations["latitude"][s] > map_latitudes[1])
                or (seismic_stations["latitude"][s] < map_latitudes[0])
            ):
                continue
            map_axis.plot(
                seismic_stations["longitude"][s],
                seismic_stations["latitude"][s],
                marker="v",
                color="k",
                markersize=markersize,
                transform=data_coords,
                zorder=1,
            )
            if seismic_stations["stations"][s] != "":
                map_axis.text(
                    seismic_stations["longitude"][s] + 0.02,
                    seismic_stations["latitude"][s],
                    seismic_stations["stations"][s],
                    fontsize=text_size,
                    transform=data_coords,
                    zorder=2,
                    bbox=props,
                )

    return map_axis


def add_scale_bar(
    ax, x_start, y_start, distance, source_crs, orientation="longitudinal", **kwargs
):
    """
    Parameters
    -----------
    ax: GeoAxes instance
        The axis on which we want to add a scale bar.
    x_start: float
        The x coordinate of the left end of the scale bar,
        given in the axis coordinate system, i.e. from 0 to 1.
    y_start: float
        The y coordinate of the left end of the scale bar,
        given in the axis coordinate system, i.e. from 0 to 1.
    distance: float
        The distance covered by the scale bar, in km.
    source_crs: cartopy.crs
        The coordinate system in which the data are written.
    orientation: string, default to 'longitudinal'
        Either 'longitudinal' or 'latitudinal'. Determine the orientation
        of the scale bar.
    """
    from cartopy.geodesic import Geodesic
    from cartopy.crs import PlateCarree

    G = Geodesic()

    # default values
    kwargs["lw"] = kwargs.get("lw", 2)
    kwargs["color"] = kwargs.get("color", "k")

    data_coords = PlateCarree()
    # transform the axis coordinates into display coordinates
    display = ax.transAxes.transform([x_start, y_start])
    # take display coordinates into data coordinates
    data = ax.transData.inverted().transform(display)
    # take data coordinates into lon/lat
    lon_start, lat_start = data_coords.transform_point(data[0], data[1], source_crs)
    # get the coordinates of the end of the scale bar
    if orientation == "latitudinal":
        lon_end, lat_end, _ = np.asarray(
            G.direct([lon_start, lat_start], 0.0, 1000.0 * distance)
        )[0]
    elif orientation == "longitudinal":
        # first, compute distance a function of longitude at
        # the given latitude lat_start
        dist_per_lon = 0.0
        longitudes = np.linspace(lon_start, lon_start + 1.0, 100)
        for i in range(1, len(longitudes)):
            dist_per_lon += utils.two_point_distance(
                (longitudes[i - 1] + 180.0) % 360.0 - 180.0,
                lat_start,
                0.0,
                (longitudes[i] + 180.0) % 360.0 - 180.0,
                lat_start,
                0.0,
            )
        lon_end = ((lon_start + distance / dist_per_lon) + 180.0) % 360.0 - 180.0
        lat_end = lat_start
    else:
        print("`orientation` should be 'longitudinal' or 'latitudinal'.")
        return
    ax.plot([lon_start, lon_end], [lat_start, lat_end], transform=data_coords, **kwargs)
    ax.text(
        (lon_start + lon_end) / 2.0,
        (lat_start + lat_end) / 2.0 - 0.001,
        "{:.0f}km".format(distance),
        transform=data_coords,
        ha="center",
        va="top",
    )
    return
