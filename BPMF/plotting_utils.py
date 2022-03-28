from .config import cfg
from . import dataset
from . import db_h5py
from . import utils

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from obspy.core import UTCDateTime as udt

def plot_template(idx,
                  db_path_T='template_db_2/',
                  db_path=cfg.dbpath,
                  n_stations=10,
                  stations=None,
                  mv_view=True,
                  show=True):
    # ---------------------------
    font = {'family' : 'sans-serif', 'weight' : 'normal', 'size' : 14}
    plt.rc('font', **font)
    # ---------------------------
    template = dataset.Template('template{:d}'.format(idx),
                                db_path_T,
                                db_path=db_path,
                                attach_waveforms=True)
    if hasattr(template, 'loc_uncertainty'):
        uncertainty_label = r'($\Delta r$ = {:.2f} km)'.\
                format(template.location_uncertainty)
    elif hasattr(template, 'cov_mat'):
        uncertainty_label = r'($\Delta X$={:.2f}km, $\Delta Y$={:.2f}km, $\Delta Z$={:.2f}km)'.\
                format(np.sqrt(template.cov_mat[0, 0]),
                       np.sqrt(template.cov_mat[1, 1]),
                       np.sqrt(template.cov_mat[2, 2]))
    else:
        uncertainty_label = ''
    if stations is not None:
        template.subnetwork(stations)
        n_stations = len(stations)
    else:
        # select the n_stations closest stations
        template.n_closest_stations(n_stations)
    sta = list(template.stations)
    n_stations = min(n_stations, len(sta))
    #sta.sort()
    n_components = len(template.channels)
    plt.figure('template_{:d}_from_{}'.format(idx, db_path+db_path_T),
               figsize=(18, 9))
    if mv_view:
        MVs = np.column_stack([template.s_moveouts,
                               template.s_moveouts,
                               template.p_moveouts])
        MVs -= MVs.min()
        time = np.arange(template.traces[0].data.size + MVs.max())/template.sampling_rate
    else:
        time = np.arange(template.traces[0].data.size)/template.sampling_rate
    for s in range(n_stations):
        for c in range(n_components):
            ax = plt.subplot(n_stations, n_components, s*n_components+c+1)
            lab = '{}.{}'.format(sta[s], template.channels[c])
            if mv_view:
                id1 = MVs[s, c]
                id2 = id1 + template.traces[0].data.size
                plt.plot(time[id1:id2], template.traces.select(station=sta[s])[c].data, label=lab)
                if c < 2:
                    plt.axvline(time[int((id1+id2)/2)], lw=2, ls='--', color='k')
                else:
                    plt.axvline(time[id1] + 1., lw=2, ls='--', color='k')
            else:
                plt.plot(time, template.traces.select(station=sta[s])[c].data, label=lab)
            plt.xlim((time[0], time[-1]))
            plt.yticks([])
            if c < 2:
                plt.legend(loc='upper left', frameon=False, handlelength=0.1, borderpad=0.)
            else:
                plt.legend(loc='upper right', frameon=False, handlelength=0.1, borderpad=0.)
            if s == n_stations-1:
                plt.xlabel('Time (s)')
            else:
                plt.xticks([])
    plt.subplots_adjust(bottom=0.07, top=0.94, hspace=0.04, wspace=0.12)
    plt.suptitle('Template {:d}, location: {:.2f}$^{{\mathrm{{o}}}}$E,'
                 '{:.2f}$^{{\mathrm{{o}}}}$N,{:.2f}km {}'
                   .format(template.template_idx,
                           template.longitude,
                           template.latitude,
                           template.depth,
                           uncertainty_label), fontsize=16)
    if show:
        plt.show()

def plot_match(catalog,
               event_index,
               db_path_T='template_db_1',
               n_stations=10,
               stations=None,
               show=True):
    """
    Plot the template waveforms (in red) on top of the continuous data (in blue)
    for detection #event_index from catalog (python dictionary with metadata).
    Path variables are used to locate where to read the template and the
    extracted waveforms from the matched filter search.

    Parameters
    ------------
    catalog: dictionary
        Dictionary with metadata such as the template matching
        output filenames, the event indexes within each day and the 
        template index.
    event_index: scalar integer
        Index of the event to plot.
    db_path_T: string, default to 'template_db_1'
        Database name of the template waveform and metadata files.
    n_stations: integer, default to 10
        Number of stations to plot.
    stations: list of strings or None, default to None
        If not None, only plot the stations provided in the list.
    show: boolean, optional.
        If True, plt.show() is called at the end.
    """
    # plotting parameters: font sizes can be tuned to display well
    # on the monitor
    old_params = plt.rcParams.copy()
    font = {'family' : 'sans-serif', 'weight' : 'normal', 'size' : 14}
    plt.rc('font', **font)
    plt.rcParams.update({'ytick.labelsize'  :  14})
    plt.rcParams['svg.fonttype'] = 'none'
    # read the template and event data
    filename = catalog.filenames[event_index].decode('utf-8')
    tid = catalog.tid
    index = catalog.indices[event_index]
    M, T = db_h5py.read_multiplet_new_version(filename, index, tid,
                                              return_tp=True,
                                              db_path=catalog.db_path,
                                              db_path_T=db_path_T,
                                              db_path_M=catalog.db_path_M)
    n_stations = min(n_stations, len(T.stations))
    if stations is not None:
        T.subnetwork(stations)
        n_stations = len(stations)
    else:
        # select the n_stations closest stations
        T.n_closest_stations(n_stations)
    st_sorted = np.asarray(T.stations)
    st_sorted = np.sort(st_sorted)
    I_s = np.argsort(np.asarray(T.stations))
    ns = len(T.stations)
    nc = len(M.components)
    strdate = M[0].stats.starttime.strftime('%Y-%m-%d')
    strtime = M[0].stats.starttime.strftime('%Y,%m,%d -- %H:%M:%S')
    fig, axes = plt.subplots(
            num=f'event{index}_{strdate}_tp{tid}',
            nrows=ns, ncols=nc, figsize=(18, 9))
    fig.suptitle(f'Template {tid} ({M.latitude:.2f}/{M.longitude:.2f}/{M.depth:.2f}km):'
                 f' Detection on {strtime}')
    # create a moveout array (n_stations x n_components)
    mv = np.stack([T.s_moveouts, T.s_moveouts, T.p_moveouts], axis=1)
    # event waveforms were extracted from detection_time - cfg.buffer_extracted_events
    # for plotting the sequence of interest, we fix t_min somewhere around 0 + cfg.buffer_extracted_events
    # i.e. t_min is somewhere around the detection time
    mv_max = max(T.s_moveouts.max(), T.p_moveouts.max())/T.sampling_rate
    t_min = cfg.buffer_extracted_events - 0.5
    t_max = t_min + cfg.template_len + mv_max + 0.5
    idx_min = utils.sec_to_samp(t_min, sr=T.sampling_rate)
    idx_max = min(M.traces[0].data.size,
                  utils.sec_to_samp(t_max, sr=T.sampling_rate))
    time  = np.linspace(t_min, t_max, idx_max-idx_min)
    time -= time.min()
    #-----------------------------------------
    for s in range(ns):
        print(st_sorted[s])
        for c in range(nc):
            axes[s, c].plot(
                    time, M.select(station=st_sorted[s])[c].data[idx_min:idx_max],
                    color='C0', label=f'{st_sorted[s]}.{M.components[c]}', lw=0.75)
            idx1 = utils.sec_to_samp(min(cfg.buffer_extracted_events, cfg.multiplet_len/4.),
                                     sr=cfg.sampling_rate)\
                   + mv[I_s[s], c]
            idx2 = idx1 + T.traces[0].data.size
            if idx2 > (time.size + idx_min):
                idx2 = time.size + idx_min
            Max = M.select(station=st_sorted[s])[c].data[idx1:idx2].max()
            if Max == 0.:
                Max = 1.
            data_template = T.traces.select(station=st_sorted[s])[c].data[:idx2-idx1]
            # scale the template data such that it matches the data amplitudes
            data_toplot   = data_template/data_template.max() * Max
            axes[s, c].plot(
                    time[idx1-idx_min:idx2-idx_min], data_toplot,
                    color='C3', lw=0.8)
            plt.xlim(time.min(), time.max())
            if s == ns-1: 
                axes[s, c].set_xlabel('Time (s)')
            else:
                axes[s, c].set_xticks([])
            if c < 2:
                axes[s, c].legend(
                        loc='upper left', framealpha=0.5,
                        handlelength=0.1, borderpad=0.2)
            else:
                axes[s, c].legend(
                        loc='upper right', framealpha=0.5,
                        handlelength=0.1, borderpad=0.2)
    plt.subplots_adjust(top=0.95, bottom=0.04, left=0.07,
                        right=0.95, hspace=0.2, wspace=0.2)
    if show:
        plt.show()
    plt.rcParams = old_params
    return fig

def plot_detection_matrix(X, datetimes=None, stack=None,
                          title=None, ax=None, show=True,
                          **kwargs):

    kwargs['time_min'] = kwargs.get('time_min', None)
    kwargs['time_max'] = kwargs.get('time_max', None)
    kwargs['text_offset'] = kwargs.get('text_offset', 0.1)
    kwargs['text_size'] = kwargs.get('text_size', plt.rcParams['font.size'])
    kwargs['datetime_format'] = kwargs.get('datetime_format', '%Y,%m,%d--%H:%M:%S')

    if datetimes is not None:
        # reorder X
        new_order = np.argsort(datetimes)
        X = X[new_order, :]
        datetimes = datetimes[new_order]
    n_detections = X.shape[0]
    if ax is None:
        fig = plt.figure('detection_matrix', figsize=(18, 9))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ax.set_title(title)
    time = np.linspace(0., X.shape[-1]/cfg.sampling_rate, X.shape[-1])
    time_min = kwargs['time_min'] if kwargs['time_min'] is not None else time.min()
    time_max = kwargs['time_max'] if kwargs['time_max'] is not None else time.max()
    time -= time_min
    if stack is not None:
        offset = 2.
        ax.plot(time, stack, color='C3', label='SVDWF Stack')
    else:
        offset = 0.
    for i in range(n_detections):
        label = 'Individual Events' if i == 0 else ''
        ax.plot(time, utils.max_norm(X[i, :]) + offset,
                lw=0.75, color='k', label=label)
        if datetimes is not None:
            plt.text(0.50, offset+kwargs['text_offset'],
                     udt(datetimes[i]).strftime(kwargs['datetime_format']),
                     bbox={'facecolor': 'white', 'alpha': 0.75},
                     fontsize=kwargs['text_size'])
        offset += 2.
    ax.set_ylabel('Offset normalized amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0., time_max-time_min)
    ax.legend(loc='upper right')
    plt.subplots_adjust(top=0.96, bottom=0.06)
    if show:
        plt.show()
    return fig

def plot_recurrence_times(catalogs=None, tids=None,
                          db_path_M='matched_filter_1',
                          db_path=cfg.dbpath,
                          magnitudes=False,
                          unique=True, plot_identity=True,
                          start_date='2012,05,01', end_date='2013,10,01',
                          ax=None, matplotlib_kwargs={}, scat_kwargs={},
                          mag_kwargs={}):
    # ------------------------------------------------
    #        Scattering plot kwargs
    scat_kwargs['edgecolor'] = scat_kwargs.get('edgecolor', 'k')
    scat_kwargs['linewidths'] = scat_kwargs.get('linewidths', 0.5)
    scat_kwargs['s'] = scat_kwargs.get('s', 25)
    # ------------------------------------------------
    #       Magnitude plot kwargs
    mag_kwargs['clb_size'] = mag_kwargs.get('clb_size', '2%')
    mag_kwargs['tick_space'] = mag_kwargs.get('tick_space', 1)
    # ------------------------------------------------
    start_date = np.datetime64(str(udt(start_date)))
    end_date = np.datetime64(str(udt(end_date)))
    OT = []
    WT = []
    M  = []
    n_total = 0
    for tid in tids:
        if catalogs is None:
            try:
                catalog = db_h5py.read_catalog_multiplets(
                        f'multiplets{tid}', db_path=db_path, db_path_M=db_path_M)
            except OSError:
                # no detection for this template, need to update template list
                continue
        else:
            # use the AggregatedCatalogs instance
            attributes = ['origin_times']
            if unique:
                attributes.append('unique_events')
            if magnitudes:
                attributes.append('magnitudes')
            catalog = catalogs.catalogs[tid].return_as_dic(attributes=attributes)
        if 'unique_events' in catalog.keys() and unique:
            mask = catalog['unique_events']
        else:
            mask = np.ones(catalog['origin_times'].size, dtype=np.bool)
        # diff[i] = ot[i+1] - ot[i], i in [0, n_events-1]
        # diff[i] is the waiting time of event i+1
        # therefore, all arrays/lists will have to start from 1
        # to match the waiting times
        mask = mask[1:]
        n_events = np.sum(mask)
        if n_events == 0:
            continue
        waiting_times = np.diff(catalog['origin_times'])[mask]
        n_total += n_events
        OT.extend(catalog['origin_times'][1:][mask])
        if magnitudes and ('magnitudes' in catalog.keys()):
            mag_ = catalog['magnitudes'][1:][mask]
            if mag_.size == 0:
                continue
            if mag_.min() < -5.:
                mag_ = -10. * np.ones(mag_.size, dtype=np.float32)
        else:
            mag_ = -10. * np.ones(n_events, dtype=np.float32)
        M.extend(mag_)
        WT.extend(waiting_times)
    # python list version: convert to numpy arrays
    OT = np.array([udt(ot) for ot in OT]).astype('datetime64[ms]')
    if len(OT) < 2:
        return
    WT = np.asarray(WT)
    M  = np.asarray(M)
    mask_mag = M == -10.
    WT /= (3600.*24.) # conversion to days
    #--------------------------------
    if magnitudes:
        # exclude the edges of the color map to avoid 
        # having dots that are too dark
        if np.sum(~mask_mag) > 0:
            vmin = mag_kwargs.get('vmin', np.percentile(M[~mask_mag], 2.))
            vmax = mag_kwargs.get('vmax', np.percentile(M[~mask_mag], 99.5))
            cNorm = Normalize(vmin=vmin, vmax=vmax)
        else:
            cNorm = Normalize(vmin=-10., vmax=0.)
        scalar_map = ScalarMappable(norm=cNorm, cmap='magma')
        colors = scalar_map.to_rgba(M[~mask_mag])
    #--------------------------------
    if ax is None:
        fig = plt.figure('catalog_recurrence_times', figsize=(18, 9))
        ax = fig.add_subplot(111)
        ax.set_rasterization_zorder(1)
        ax.set_title('Temporal distribution of the detections '
                     '({:d} detections)'.format(n_total))
    else:
        fig = ax.get_figure()
    #------------------------------
    #-------- plot events without mag -----
    sc2 = ax.scatter(OT[mask_mag], WT[mask_mag], marker='v',
                     zorder=1, **scat_kwargs, **matplotlib_kwargs)
    if plot_identity:
        time_0 = OT[mask_mag].min() - np.timedelta64(int(1000.*WT[mask_mag][0]*3600.*24.), 'ms')
        time = np.arange(time_0, end_date,
                         (end_date-time_0)/1000.)
        timestamps = np.array([udt(str(t)).timestamp for t in time])
        timestamps -= timestamps[0]
        ax.plot(time, timestamps/(3600.*24.), ls='--', color='C0', lw=4, zorder=2)
    #------------------------------
    #------- plot events with mag -------
    if magnitudes:
        # make these symbols twice larger than the magnitude-less events
        scat_kwargs_mag = scat_kwargs.copy()
        scat_kwargs_mag['s'] *= 2
        I = np.argsort(M[~mask_mag])
        sc = ax.scatter(OT[~mask_mag][I], WT[~mask_mag][I], zorder=2,
                        color=colors[I], marker='o', **scat_kwargs_mag)
    #--------------------------
    ax.set_xlim(start_date, end_date)
    ax.set_ylim(10**np.floor(np.log10(WT.min())),
                10**(np.ceil(np.log10(WT.max()))))
    ax.semilogy()
    ax.grid(axis='x')
    ax.set_ylabel('Recurrence Times (days)')
    if magnitudes:
        # plot the colorbar
        clb_ticks = np.arange(cNorm.vmin, cNorm.vmax, mag_kwargs['tick_space'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=mag_kwargs['clb_size'], pad=0.08, axes_class=plt.Axes)
        clb = plt.colorbar(scalar_map, cax=cax, label=r'$M_{L}$',
                           orientation='vertical', ticks=clb_ticks)
    plt.subplots_adjust(bottom=0.4,
                        top=0.8,
                        left=0.06,
                        right=0.99)
    return fig

def plot_catalog(tids=None, db_path_T=None, db_path_M=None, catalog=None,
                 ax=None, remove_multiples=True, scat_kwargs={},
                 cmap=None, db_path=cfg.dbpath):

    if cmap is None:
        try:
            import colorcet as cc
            cmap = cc.cm.bjy
        except Exception as e:
            print(e)
            cmap = 'viridis'

    # ------------------------------------------------
    #        Scattering plot kwargs
    scat_kwargs['edgecolor'] = scat_kwargs.get('edgecolor', 'k')
    scat_kwargs['linewidths'] = scat_kwargs.get('linewidths', 0.5)
    scat_kwargs['s'] = scat_kwargs.get('s', 10)
    scat_kwargs['zorder'] = scat_kwargs.get('zorder', 0)
    # ------------------------------------------------

    if catalog is None:
        # if catalog is None, tids, db_path_T and db_path_M
        # should be given
        # ------------------
        # compile detections from these templates in a single
        # earthquake catalog
        catalog_filenames = [f'multiplets{tid}catalog.h5' for tid in tids]
        AggCat = dataset.AggregatedCatalogs(
                filenames=catalog_filenames, db_path_M=db_path_M, db_path=db_path)
        AggCat.read_data(items_in=['origin_times', 'location', 'unique_events'])
        catalog = AggCat.flatten_catalog(
                attributes=['origin_times', 'latitude', 'longitude', 'depth'],
                unique_events=True)
    cNorm = Normalize(vmin=catalog['latitude'].min(),
                      vmax=catalog['latitude'].max())
    scalar_map = ScalarMappable(norm=cNorm, cmap=cmap)
    scalar_map.set_array([])
    
    # plot catalog
    if ax is None:
        fig = plt.figure('earthquake_catalog', figsize=(18, 9))
        ax = fig.add_subplot(111)
    else:
        # use the user-provided axis
        fig = ax.get_figure()
    ax.set_title('{:d} events'.format(len(catalog['origin_times'])))
    ax.set_xlabel('Calendar Time')
    ax.set_ylabel('Longitude')
    times = np.array([str(udt(time)) for time in catalog['origin_times']],
                     dtype='datetime64')
    ax.scatter(times, catalog['longitude'],
               color=scalar_map.to_rgba(catalog['latitude']),
               rasterized=True, **scat_kwargs)
    ax.set_xlim(times.min(), times.max())
    
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="2%", pad=0.08)
    plt.colorbar(scalar_map, cax, orientation='vertical', label='Latitude')
    return fig

# ---------------------------------------------------------------
#                Utils for maps
# ---------------------------------------------------------------

def initialize_map(map_longitudes, map_latitudes,
                   map_axis=None, seismic_stations=None,
                   GPS_stations=None, text_size=14, markersize=10,
                   topography_file=None, path_topo='',
                   faults=None,
                   right_labels=False, left_labels=True,
                   bottom_labels=True, top_labels=False,
                   **kwargs):
    """Initialize map instance with Cartopy. 

    """
    import cartopy as ctp

    kwargs['topo_alpha'] = kwargs.get('topo_alpha', 0.30)
    kwargs['downsample_faults'] = kwargs.get('downsample_faults', True)
    kwargs['shaded_topo'] = kwargs.get('shaded_topo', True)
    kwargs['topo_cmap'] = kwargs.get('topo_cmap', 'gray')
    kwargs['topo_cnorm'] = kwargs.get('topo_cnorm', None)
    kwargs['fault_zorder'] = kwargs.get('fault_zorder', 1.01)
    figsize = kwargs.get('figsize', (15, 15))

    map_corners = [map_longitudes[0],
                   map_latitudes[0],
                   map_longitudes[1],
                   map_latitudes[1]]
    
    data_coords = ctp.crs.PlateCarree()
    if map_axis is None:
        #projection = ctp.crs.PlateCarree()
        projection = ctp.crs.Mercator(central_longitude=sum(map_longitudes)/2.,
                                      min_latitude=map_latitudes[0],
                                      max_latitude=map_latitudes[1])
        fig = plt.figure(kwargs.get('figname', 'map'), figsize=figsize)
        map_axis = fig.add_subplot(111, projection=projection)
        map_axis.set_rasterization_zorder(1)
        map_axis.set_extent([map_longitudes[0], 
                             map_longitudes[1],
                             map_latitudes[0],
                             map_latitudes[1]],
                            crs=data_coords)

    RES = '10m'

    if topography_file is not None:
        # -----------------------
        # get topography
        import netCDF4
        with netCDF4.Dataset(os.path.join(path_topo,
                             topography_file), 'r') as f:
            if 'z' in f.variables:
                topo = f.variables['z'][:].data
            elif 'Band1' in f.variables:
                topo = f.variables['Band1'][:].data
            if 'lon' in f.variables:
                lon_topo = f.variables['lon'][:].data
            elif 'x' in f.variables:
                lon_topo = f.variables['x'][:].data
            if 'lat' in f.variables:
                lat_topo = f.variables['lat'][:].data
            elif 'y' in f.variables:
                lat_topo = f.variables['y'][:].data
        # select relevant area
        selected_lon = np.where((lon_topo >= map_longitudes[0]) &
                                (lon_topo <= map_longitudes[1]))[0]
        selected_lat = np.where((lat_topo >= map_latitudes[0]) &
                                (lat_topo <= map_latitudes[1]))[0]
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
        if kwargs['shaded_topo']:
            # get topography gradient
            grad_x, grad_y = np.gradient(topo)
            slope = np.pi/2. - np.arctan(np.sqrt(grad_x**2 + grad_y**2))
            aspect = np.arctan2(grad_x, grad_y)
            altitude = np.pi/4. # sun angle
            azimuth = np.pi/2. # sun direction
            topo = np.sin(altitude) * np.sin(slope)\
                    + np.cos(altitude) * np.cos(slope)\
                    * np.cos((azimuth - np.pi/2.) - aspect)
        #levels_topo = np.linspace(-500., topo.max(), 20)
        #print('Plot the topography.')
        # transform data
        # fast version
        t1 = give_time()
        lon_g, lat_g = np.meshgrid(lon_topo, lat_topo, indexing='xy')
        trans_data = map_axis.projection.transform_points(
                data_coords, lon_g, lat_g, topo)
        X = trans_data[..., 0]
        Y = trans_data[..., 1]
        Z = trans_data[..., 2]
        map_axis.imshow(Z,
                        extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap=kwargs['topo_cmap'],
                        norm=kwargs['topo_cnorm'],
                        origin='lower',
                        alpha=kwargs['topo_alpha'],
                        interpolation='bilinear',
                        zorder=-1)
        t2 = give_time()
        #print('Topography plotted in {:.2f}s'.format(t2-t1))

    #------------ DRAW MERIDIANS AND PARALLELS --------------
    LON0 = (int(map_longitudes[0]/0.5) + 1.) * 0.5
    LON1 = (int(map_longitudes[1]/0.5) + 1.) * 0.5
    #lon_ticks = np.arange(map_longitudes[0], map_longitudes[1]+0.5, 0.5)
    lon_ticks = np.arange(LON0-0.5, LON1+0.5, 0.5)
    LAT0 = (int(map_latitudes[0]/0.5) + 1.) * 0.5
    LAT1 = (int(map_latitudes[1]/0.5) + 1.) * 0.5
    #lat_ticks = np.arange(map_latitudes[0], map_latitudes[1]+0.5, 0.5)
    lat_ticks = np.arange(LAT0-0.5, LAT1+0.5, 0.5)
    #--------------------------------------------------------
    
    #----------- DRAW BORDERS -------------
    brds = ctp.feature.BORDERS
    #--------------------------------------
    
    gl = map_axis.gridlines(draw_labels=True,
                            linewidth=1,
                            alpha=0.5,
                            color='k',
                            linestyle='--')
    gl.right_labels = right_labels
    gl.left_labels = left_labels
    gl.top_labels = top_labels
    gl.bottom_labels = bottom_labels
    
    if kwargs.get('coastlines', True):
        #print('Add coast lines.')
        map_axis.add_feature(ctp.feature.GSHHSFeature(
            scale='full', levels=kwargs.get('coastline_levels', [1, 2]),
            zorder=0.49))#, rasterized=True))
    # comment this to save time on plotting high-resolution oceans
    if kwargs.get('oceans', False):
        oceans = ctp.feature.OCEAN
        map_axis.add_feature(ctp.feature.NaturalEarthFeature(
            category=oceans.category,
            name=oceans.name,
            scale='10m',
            facecolor='#0076b482'))

    if faults is not None:
        map_axis.add_geometries(faults['geometry'], crs=data_coords, 
                facecolor='none', edgecolor='k')
    # plot optional elements
    props = dict(boxstyle='round', facecolor='white',
                 edgecolor=None, alpha=0.6, pad=0.2)

    # plot seismic stations
    if seismic_stations is not None:
        #print('Add seismic stations.')
        for s in range(len(seismic_stations['stations'])):
            if (seismic_stations['longitude'][s] > map_longitudes[1]) or\
               (seismic_stations['longitude'][s] < map_longitudes[0]) or\
               (seismic_stations['latitude'][s] > map_latitudes[1]) or\
               (seismic_stations['latitude'][s] < map_latitudes[0]):
                continue
            map_axis.plot(seismic_stations['longitude'][s],
                          seismic_stations['latitude'][s],
                          marker='v',
                          color='k',
                          markersize=markersize,
                          transform=data_coords,
                          zorder=1)
            if seismic_stations['stations'][s] != '':
                map_axis.text(seismic_stations['longitude'][s]+0.02,
                              seismic_stations['latitude'][s],
                              seismic_stations['stations'][s],
                              fontsize=text_size,
                              transform=data_coords,
                              zorder=2,
                              bbox=props)
    if GPS_stations is not None:
        #print('Add GPS stations.')
        for s in range(len(GPS_stations['stations'])):
            if (GPS_stations['longitude'][s] > map_longitudes[1]) or\
               (GPS_stations['longitude'][s] < map_longitudes[0]) or\
               (GPS_stations['latitude'][s] > map_latitudes[1]) or\
               (GPS_stations['latitude'][s] < map_latitudes[0]):
                continue
            map_axis.plot(GPS_stations['longitude'][s],
                          GPS_stations['latitude'][s],
                          marker='s',
                          color='C4',
                          markersize=markersize,
                          transform=data_coords,
                          zorder=1)
            if GPS_stations['stations'][s] != '':
                map_axis.text(GPS_stations['longitude'][s]+0.01,
                              GPS_stations['latitude'][s],
                              GPS_stations['stations'][s],
                              fontsize=text_size,
                              transform=data_coords,
                              zorder=2,
                              bbox=props)

    return map_axis

def add_scale_bar(ax, x_start, y_start, distance, source_crs,
                  azimuth=90., **kwargs):
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
    """
    from cartopy.geodesic import Geodesic
    G = Geodesic()
 
    # default values
    kwargs['lw'] = kwargs.get('lw', 2)
    kwargs['color'] = kwargs.get('color', 'k')

    data_coords = ctp.crs.PlateCarree()
    # transform the axis coordinates into display coordinates
    display = ax.transAxes.transform([x_start, y_start])
    # take display coordinates into data coordinates
    data = ax.transData.inverted().transform(display)
    # take data coordinates into lon/lat
    lon_start, lat_start = data_coords.transform_point(
            data[0], data[1], source_crs)
    # get the coordinates of the end of the scale bar
    lon_end, lat_end, _ = np.asarray(G.direct(
            [lon_start, lat_start], azimuth, 1000.*distance))[0]
    ax.plot([lon_start, lon_end], [lat_start, lat_end],
            transform=data_coords, **kwargs)
    ax.text((lon_start+lon_end)/2., (lat_start+lat_end)/2.-0.001,
            '{:.0f}km'.format(distance), transform=data_coords,
            ha='center', va='top')
    return

