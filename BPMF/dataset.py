import os

import numpy as np
import h5py as h5

from .config import cfg
from . import utils
from . import catalog_utils

import obspy as obs
import pandas as pd
import copy
import datetime
from obspy.core import UTCDateTime as udt

from time import time as give_time

class Network():
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
        self.where = os.path.join(cfg.network_path, network_file)
    
    @property
    def n_stations(self):
        return np.int32(len(self.stations))

    @property
    def n_components(self):
        return np.int32(len(self.components))

    def read(self):
        """
        Reads the metadata from the file at self.where

        Note: This function can be modified to match the user's
        data convention.
        """
        networks = []
        stations = []
        components = []
        with open(self.where, 'r') as file:
            # read in start and end dates
            columns = file.readline().strip().split()
            self.start_date = udt(columns[0])
            self.end_date = udt(columns[1])

            # read in component names
            columns = file.readline().strip().split()
            for component in columns[1:]:
                components.append(component)
            self.components = components
            
            data_centers = []
            networks     = []
            locations    = []

            # read in station names and coordinates
            latitude, longitude, depth = [], [], []
            for line in file:
                columns = line.strip().split()
                data_centers.append(columns[0])
                networks.append(columns[1])
                stations.append(columns[2])
                locations.append(columns[3])
                latitude.append(np.float32(columns[4]))
                longitude.append(np.float32(columns[5]))
                depth.append(-1.*np.float32(columns[6]) / 1000.)  # convert m to km

            self.data_centers = data_centers
            self.networks     = networks
            self.stations     = stations
            self.locations    = locations
            self.latitude     = np.asarray(latitude, dtype=np.float32)
            self.longitude    = np.asarray(longitude, dtype=np.float32)
            self.depth        = np.asarray(depth, dtype=np.float32)

    def datelist(self):
        dates = []
        date = self.start_date
        while date <= self.end_date:
            dates.append(date)
            date += datetime.timedelta(days=1)

        return dates

    def stations_idx(self, stations):
        if not isinstance(stations, list) and not isinstance(stations, np.ndarray):
            stations = [stations]
        idx = []
        for station in stations:
            idx.append(self.stations.index(station))
        return idx

    def subset(self, stations, components, method='keep'):
        """
        Parameters
        -----------
        stations: list or array of strings
            Stations to keep or discard, depending on the method.
        components: list of array of strings
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

        if method == 'discard':
            for station in stations:
                if station in self.stations:
                    idx = subnetwork.stations.index(station)
                    subnetwork.stations.remove(station)
                    np.delete(subnetwork.latitude, idx)
                    np.delete(subnetwork.longitude, idx)
                    np.delete(subnetwork.depth, idx)
                    np.delete(subnetwork.networks, idx)
                else:
                    print('{} not a network station'.format(station))
            for component in components:
                if component in self.components:
                    idx = subnetwork.components.index(component)
                    subnetwork.components.remove(component)
                else:
                    print('{} not a network component'.format(station))
        elif method == 'keep':
            sta_indexes = [self.stations.index(sta) for sta in stations]
            subnetwork.stations = stations
            subnetwork.latitude = [self.latitude[s] for s in sta_indexes]
            subnetwork.longitude = [self.longitude[s] for s in sta_indexes]
            subnetwork.depth = [self.depth[s] for s in sta_indexes]
            subnetwork.networks = [self.networks[s] for s in sta_indexes]
            subnetwork.components = components
        else:
            print('method should be "keep" or "discard"!')
            return
        return subnetwork


    @property
    def interstation_distances(self):
        """Compute the distance between all station pairs.

        """
        if (hasattr(self, '_interstation_distances') and
                self._interstation_distances.shape[0] == self.n_stations):
            # was already computed and the size of the network was unchanged
            return self._interstation_distances
        else:
            from cartopy.geodesic import Geodesic

            G = Geodesic()

            intersta_dist = np.zeros((len(self.stations), len(self.stations)),
                                     dtype=np.float64)
            for s in range(len(self.stations)):
                d = G.inverse(np.array([[self.longitude[s], self.latitude[s]]]),
                              np.hstack((self.longitude.reshape(-1, 1),
                                         self.latitude.reshape(-1, 1))))
                # d is in m, convert it to km
                d = np.asarray(d)[:, 0]/1000.
                intersta_dist[s, :] = np.sqrt(d.squeeze()**2 + (self.depth[s] - self.depth))

            # return distance in km
            self._interstation_distances = intersta_dist
            return self._interstation_distances


class Template(object):
    """A Template class to handle template data and metadata.  


    """

    def __init__(self, template_filename, db_path_T,
                 db_path=cfg.dbpath, attach_waveforms=False):
        """Read the template metadata and data.

        Parameters
        ------------
        template_filename: string
            Base name of the template data and metadata files.  
            The metadata file is assumed to be {template_filename}meta.h5  
            The data file is assumed to be {template_filename}wav.h5  
        db_path_T: string
            Name of the folder where template files are stored.
        db_path: string, default to cfg.dbpath
            Name of the root folder where output files are stored.
        attach_waveforms: boolean, default to False
            If True, read the waveforms from {template_filename}wav.h5
            *AND* builds an obspy stream attribute.
        """
        self.db_path = db_path
        self.db_path_T = db_path_T
        self.filename = template_filename
        self.where = os.path.join(
                db_path, db_path_T, template_filename)
        # load metadata
        with h5.File(self.where + 'meta.h5', 'r') as f:
            for key in f.keys():
                self.__setattr__(key, f[key][()])
        # alias for template_idx:
        if hasattr(self, 'template_idx'):
            self.tid = self.template_idx
        else:
            self.template_idx = self.tid
        self.stations = self.stations.astype('U')
        self.channels = self.channels.astype('U')
        # keep copies of the original network-wide attributes
        # for when we will select subnetworks
        for attr in ['stations', 'p_moveouts', 's_moveouts', 'travel_times']:
            self.__setattr__('network_{}'.format(attr),
                             #self.__getattr__(attr).copy()
                             getattr(self, attr).copy())
        self.network_reference_absolute_time =\
                copy.copy(self.reference_absolute_time)
        if type(self.network_s_moveouts.flat[0]) is np.float32:
            self.network_s_moveouts = utils.sec_to_samp(
                    self.network_s_moveouts, sr=self.sampling_rate)
            self.s_moveouts = utils.sec_to_samp(
                    self.s_moveouts, sr=self.sampling_rate)
        if type(self.network_p_moveouts.flat[0]) is np.float32:
            self.network_p_moveouts = utils.sec_to_samp(
                    self.network_p_moveouts, sr=self.sampling_rate)
            self.p_moveouts = utils.sec_to_samp(
                    self.p_moveouts, sr=self.sampling_rate)
        if attach_waveforms:
            self.read_waveforms()
            self.traces = obs.Stream()
            for s, sta in enumerate(self.stations):
                for c, cha in enumerate(self.channels):
                    tr = obs.Trace()
                    tr.data = self.network_waveforms[s, c, :]
                    tr.stats.station = sta
                    tr.stats.channel = cha
                    tr.stats.sampling_rate = cfg.sampling_rate
                    self.traces += tr

    def read_waveforms(self):
        """Read template data.  


        Read the template waveforms from the template data file
        defined when instanciating the Template class.
        """
        # load waveforms
        with h5.File(self.where + 'wav.h5', 'r') as f:
            self.network_waveforms = f['waveforms'][()]

    def subnetwork(self, subnet_stations):
        """Adjust the template attributes to the requested subnetwork.  

        Parameters
        -------------
        subnet_stations: list of strings
            List of the station names to keep.
        """
        if type(subnet_stations) != np.array:
            subnet_stations = np.asarray(subnet_stations).astype('U')
        else:
            subnet_stations = subnet_stations.astype('U')
        self.stations = subnet_stations
        # get the index map from the whole network to
        # the subnetwork
        self.map_to_subnet = np.int32([np.where(self.network_stations == sta)[0]
                                       for sta in self.stations]).squeeze()
        # attach the waveforms
        self.waveforms = self.network_waveforms[self.map_to_subnet, :, :]
        # update moveouts
        self.p_moveouts = self.network_p_moveouts[self.map_to_subnet]
        self.s_moveouts = self.network_s_moveouts[self.map_to_subnet]
        # update travel times
        self.travel_times = self.network_travel_times[self.map_to_subnet]

    def subtemplate(self, subnet_stations):
        """Restrict the template to the requested subnetwork.  

        Parameters
        -------------
        subnet_stations: list of strings
            List of the station names to keep.
        """
        if type(subnet_stations) != np.ndarray:
            subnet_stations = np.asarray(subnet_stations).astype('U')
        else:
            subnet_stations = subnet_stations.astype('U')
        # get the index map from the whole network to the subnetwork
        self.map_to_subnet = np.int32([np.where(self.network_stations == sta)[0]
                                       for sta in subnet_stations]).squeeze()
        # attach the waveforms
        self.network_waveforms = self.network_waveforms[self.map_to_subnet, :, :]
        # update moveouts
        self.network_p_moveouts = self.network_p_moveouts[self.map_to_subnet]
        self.network_s_moveouts = self.network_s_moveouts[self.map_to_subnet]
        # update travel times
        self.network_travel_times = self.network_travel_times[self.map_to_subnet]
        # update stations
        self.network_stations = copy.copy(subnet_stations)
        self.stations = copy.copy(subnet_stations)
        # update data availability
        self.template_data_availability = \
                self.template_data_availability[self.map_to_subnet]

    def n_closest_stations(self, n, available_stations=None):
        """Adjust the template attributes to the `n` closest stations.  


        Find the `n` closest stations and call `subnetwork` to adjust the
        template attributes to these stations.

        Parameters
        ----------------
        n: scalar, int
            The `n` closest stations.
        available_stations: list of strings, default to None
            The list of stations from which we search the closest stations.
            If some stations are known to not have available data, the user
            may choose to not include these in the closest stations.
        """
        index_pool = np.arange(len(self.network_stations))
        # limit the index pool to available stations
        if available_stations is not None:
            availability = np.in1d(self.network_stations, available_stations.astype('U'))
            valid = availability & self.template_data_availability
        else:
            valid = self.template_data_availability
        index_pool = index_pool[valid]
        self.closest_stations = \
                index_pool[np.argsort(self.source_receiver_distances[index_pool])]
        # make sure we return a n-vector
        if self.closest_stations.size < n:
            missing = n - self.closest_stations.size
            remaining_indexes = np.setdiff1d(np.argsort(self.source_receiver_distances),
                                             self.closest_stations)
            self.closest_stations = np.hstack( (self.closest_stations,
                                                remaining_indexes[:missing]) )
        self.subnetwork(self.network_stations[self.closest_stations[:n]])

    def n_best_SNR_stations(self, n, available_stations=None):
        """Adjust the template attributes to the `n` best SNR stations.  


        Find the `n` best SNR stations and call `subnetwork` to adjust the
        template attributes to these stations.

        Parameters
        ----------------
        n: scalar, int
            The `n` closest stations.
        available_stations: list of strings, default to None
            The list of stations from which we search the closest stations.
            If some stations are known to not have available data, the user
            may choose to not include these in the closest stations.
        """

        index_pool = np.arange(len(self.network_stations))
        # limit the index pool to available stations
        if available_stations is not None:
            availability = np.in1d(self.network_stations, available_stations.astype('U'))
            valid = availability & self.template_data_availability
        else:
            valid = self.template_data_availability
        index_pool = index_pool[valid]
        self.best_SNR_stations = \
                index_pool[np.argsort(self.SNR[index_pool])[::-1]]
        # make sure we return a n-vector
        if self.best_SNR_stations.size < n:
            missing = n - self.best_SNR_stations.size
            remaining_indexes = np.setdiff1d(np.argsort(self.SNR)[::-1],
                                             self.best_SNR_stations)
            self.best_SNR_stations = np.hstack( (self.best_SNR_stations,
                                                remaining_indexes[:missing]) )
        self.subnetwork(self.network_stations[self.best_SNR_stations[:n]])

    def distance(self, latitude, longitude, depth):
        """Compute distance between template and a given location.  

        Parameters
        -----------
        latitude: scalar, float
            Latitude of the target location.
        longitude: scalar, float
            Longitude of the target location.
        depth: scalar, float
            Depth of the target location, in km.
        """
        from .utils import two_point_distance
        return two_point_distance(self.latitude, self.longitude, self.depth,
                                  latitude, longitude, depth)

    @property
    def hmax_unc(self):
        if hasattr(self, '_hmax_unc'):
            return self._hmax_unc
        else:
            self.hor_ver_uncertainties()
            return self._hmax_unc

    @property
    def vmax_unc(self):
        if hasattr(self, '_vmax_unc'):
            return self._vmax_unc
        else:
            self.hor_ver_uncertainties()
            return self._vmax_unc

    @property
    def az_hmax_unc(self):
        if hasattr(self, '_az_hmax_unc'):
            return self._az_hmax_unc
        else:
            self.hor_ver_uncertainties()
            return self._az_hmax_unc

    def hor_ver_uncertainties(self):
        """Compute the horizontal and vertical uncertainties on location.  

        The vertical uncertainty is taken as the maximum vertical range
        covered by the uncertainty ellipsoid.  
        The horizontal uncertainty is taken as the maximum horizontal range
        covered by the uncertainty ellipsoid.
        
        New Attributes
        ----------------
        hmax_unc: scalar, float
            The maximum horizontal uncertainty, taken as the maximum
            horizontal range covered by the uncertainty ellipsoid.
        vmax_unc: scalar, float
            The maximum vertical uncertainty, taken as the maximum
            vertical range covered by the uncertainty ellipsoid.
        az_hmax_unc: scalar, float
            The azimuth (angle from north) of the maximum horizontal
            uncertainty.

        Note: hmax + vmax does not have to be equal to the
        max_loc, the latter simply being the length of the
        longest semi-axis of the uncertainty ellipsoid.
        """
        w, v = np.linalg.eigh(self.cov_mat)
        # the eigenvalues are the variances (units of [distance**2])
        # in the eigendirections, we need the standard deviations
        # (units of [distance])
        std = np.sqrt(w)
        # check the vertical components of all semi-axes:
        vertical_unc = np.abs(std*v[2, :])
        # keep the maximum for the vertical uncertainty
        max_vertical = vertical_unc.max()
        # check the horizontal components of all semi-axes:
        horizontal_unc = np.sqrt(np.sum((std[np.newaxis, :]*v[:2, :])**2, axis=0))
        # keep the maximum as the horizontal uncertainty
        max_horizontal = horizontal_unc.max()
        direction_hmax = v[:, horizontal_unc.argmax()]
        azimuth_hmax = np.arctan2(direction_hmax[0], direction_hmax[1])
        azimuth_hmax = (azimuth_hmax*180./np.pi)%180.
        # these private attributes should be called via their property names
        self._hmax_unc = max_horizontal
        self._vmax_unc = max_vertical
        self._az_hmax_unc = azimuth_hmax

    def write(self, path, filename=None, fields={}):
        """Write template data and metadata files.  

        Parameters
        -----------
        path: string
            Path to the folder where to write the files.
        filename: string, default to None
            Base name of the data and metadata files. The data filename
            is {filename}wav.h5 and the metadata filename is {filename}meta.h5.
            If None, the template id is used to define the filename.
        fields: dictionary, default to empty dictionary 
            Map between the template attributes and the hdf5 field names.
            Example: `fields['stations'] = 'network_stations'` means that the
            template attribute `self.network_stations` will be stored as the
            `'stations'` field of the hdf5 file.
        """
        if filename is None:
            filename = f'template{self.tid}'
        # the key-word arguments map the attribute names to the dataset entries
        for attr in ['longitude', 'latitude', 'depth', 'tid', 'channels',
                'cov_mat', 'max_location_uncertainty', 'duration',
                'sampling_rate', 'SNR', 'source_receiver_distances',
                'template_data_availability', 'origin_time']:
            kwargs.setdefault(attr, attr)
        for attr in ['stations', 'p_moveouts', 's_moveouts',
                'reference_absolute_time', 'travel_times']:
            kwargs.setdefault(attr, 'network_'+attr)
        full_fn = os.path.join(path, filename)
        with h5.File(full_fn+'meta.h5', mode='w') as f:
            for attr in kwargs.keys():
                attr_ = getattr(self, kwargs[attr])
                if (isinstance(attr_, np.ndarray)
                        and (attr_.dtype.kind == np.dtype('U').kind)):
                    attr_ = attr_.astype('S')
                f.create_dataset(attr, data=attr_)
        with h5.File(full_fn+'wav.h5', mode='w') as f:
            f.create_dataset('waveforms', data=self.waveforms)


class TemplateGroup(object):
    """A TemplateGroup class to handle groups of templates.  


    """

    def __init__(self, tids, db_path_T, db_path=cfg.dbpath):
        """Read the templates' data and metadata.  

        Parameters
        -----------
        tids: list or nd.array
            List of template ids in the group.
        db_path_T: string
            Name of the folder with template files.
        db_path: string, default to cfg.dbpath
            Name of the folder with output files.
        """
        self.templates = []
        self.tids = tids
        self.n_templates = len(tids)
        self.db_path_T = db_path_T
        self.db_path = db_path
        self.tids_map = {}
        for t, tid in enumerate(tids):
            self.templates.append(
                    Template(f'template{tid}', db_path_T, db_path=db_path))
            self.tids_map[tid] = t

    def attach_intertp_distances(self):
        """Compute distance between all template pairs.  

        """
        print('Computing the inter-template distances...')
        self.intertp_distances = intertp_distances_(
                templates=self.templates, return_as_pd=True)

    def attach_directional_errors(self):
        """Compute the length of the uncertainty ellipsoid
        inter-template direction.  

        New Attributes 
        ----------
        directional_errors: (n_templates, n_templates) pandas DataFrame
            The length, in kilometers, of the uncertainty ellipsoid in the
            inter-template direction.  
            Example: self.directional_errors.loc[tid1, tid2] is the width of
            template tid1's uncertainty ellipsoid in the direction of
            template tid2.
        """
        print('Computing the inter-template directional errors...')
        from cartopy import crs
        # ----------------------------------------------
        #      Define the projection used to
        #      work in a cartesian space
        # ----------------------------------------------
        data_coords = crs.PlateCarree()
        longitudes = np.float32([self.templates[i].longitude for i in range(self.n_templates)])
        latitudes = np.float32([self.templates[i].latitude for i in range(self.n_templates)])
        depths = np.float32([self.templates[i].depth for i in range(self.n_templates)])
        projection = crs.Mercator(central_longitude=np.mean(longitudes),
                                  min_latitude=latitudes.min(),
                                  max_latitude=latitudes.max())
        XY = projection.transform_points(data_coords, longitudes, latitudes)
        cartesian_coords = np.stack([XY[:, 0], XY[:, 1], depths], axis=1)
        # compute the directional errors
        dir_errors = np.zeros((self.n_templates, self.n_templates), dtype=np.float32)
        for t in range(self.n_templates):
            unit_direction = cartesian_coords - cartesian_coords[t, :]
            unit_direction /= np.sqrt(np.sum(unit_direction**2, axis=1))[:, np.newaxis]
            # this operation produced NaNs for i=t
            unit_direction[np.isnan(unit_direction)] = 0.
            # compute the length of the covariance ellipsoid
            # in the direction that links the two earthquakes
            cov_dir = np.abs(np.sum(
                self.templates[t].cov_mat.dot(unit_direction.T)*unit_direction.T, axis=0))
            # covariance is unit of [distance**2], therefore we need the sqrt:
            dir_errors[t, :] = np.sqrt(cov_dir)
        # format it in a pandas DataFrame
        dir_errors = pd.DataFrame(columns=[tid for tid in self.tids],
                                  index=[tid for tid in self.tids],
                                  data=dir_errors)
        self.directional_errors = dir_errors


    def attach_ellipsoid_distances(self, substract_errors=True):
        """
        Combine inter-template distances and directional errors
        to compute the minimum inter-uncertainty ellipsoid distances.
        This quantity can be negative if the ellipsoids overlap.
        """
        import pandas as pd
        if not hasattr(self, 'intertp_distances'):
            self.attach_intertp_distances()
        if not hasattr(self, 'directional_errors'):
            self.attach_directional_errors()
        if substract_errors:
            ellipsoid_distances = self.intertp_distances.values\
                                - self.directional_errors.values\
                                - self.directional_errors.values.T
        else:
            ellipsoid_distances = self.intertp_distances.values
        self.ellipsoid_distances = pd.DataFrame(
                columns=[tid for tid in self.tids],
                index=[tid for tid in self.tids],
                data=ellipsoid_distances)
    
    def read_stacks(self, **SVDWF_kwargs):
        self.db_path_M = SVDWF_kwargs.get('db_path_M', 'none')
        self.stacks = []
        for t, tid in enumerate(self.tids):
            stack = utils.SVDWF_multiplets(
                    tid, db_path=self.db_path,
                    db_path_T=self.db_path_T,
                    **SVDWF_kwargs)
            self.stacks.append(stack)

    def read_waveforms(self):
        for template in self.templates:
            template.read_waveforms()

    def plot_cc(self, cmap='inferno'):
        if not hasattr(self, 'intertp_cc'):
            print('Should call self.template_similarity first')
            return
        import matplotlib.pyplot as plt
        fig = plt.figure('template_similarity', figsize=(18, 9))
        ax = fig.add_subplot(111)
        tids1_g, tids2_g = np.meshgrid(self.tids, self.tids, indexing='ij')
        pc = ax.pcolormesh(tids1_g, tids2_g, self.intertp_cc.values, cmap=cmap)
        ax.set_xlabel('Template id')
        ax.set_ylabel('Template id')
        fig.colorbar(pc, label='Correlation Coefficient')
        return fig

    def template_similarity(self, distance_threshold=5.,
                            n_stations=10, max_lag=10,
                            device='cpu'):
        """
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
        """
        import pandas as pd
        import fast_matched_filter as fmf
        if not hasattr(self, 'ellipsoid_distances'):
            self.attach_ellipsoid_distances()
        for template in self.templates:
            template.read_waveforms()
            template.n_closest_stations(n_stations)
        print('Computing the similarity matrix...')
        # format arrays for FMF
        tp_array = np.stack([tp.network_waveforms for tp in self.templates],
                             axis=0)
        data = tp_array.copy()
        tp_array = tp_array[..., max_lag:-max_lag]
        moveouts = np.zeros(tp_array.shape[:-1], dtype=np.int32)
        intertp_cc = np.zeros((self.n_templates, self.n_templates),
                              dtype=np.float32)
        n_stations, n_components = moveouts.shape[1:]
        # use FMF on one template at a time against all others
        for t in range(self.n_templates):
            #print(f'--- {t} / {self.n_templates} ---')
            template = self.templates[t]
            weights = np.zeros(tp_array.shape[:-1], dtype=np.float32)
            weights[:, template.map_to_subnet, :] = 1.
            weights /= np.sum(weights, axis=(1, 2))[:, np.newaxis, np.newaxis]
            above_thrs = self.ellipsoid_distances[self.tids[t]] > distance_threshold
            weights[above_thrs, ...] = 0.
            below_thrs = self.ellipsoid_distances[self.tids[t]] < distance_threshold
            for s in range(n_stations):
                for c in range(n_components):
                    # use trick to keep station and component dim
                    slice_ = np.index_exp[:, s:s+1, c:c+1, :]
                    data_ = data[(t,)+slice_[1:]]
                    # discard all templates that have weights equal to zero
                    keep = (weights[:, s, c] != 0.)\
                          &  (np.sum(tp_array[slice_], axis=-1).squeeze() != 0.)
                    if (np.sum(keep) == 0) or (np.sum(data[(t,)+slice_[1:]]) == 0):
                        # occurs if this station is not among the 
                        # n_stations closest stations
                        # or if no data were available
                        continue
                    cc = fmf.matched_filter(
                            tp_array[slice_][keep, ...], moveouts[slice_[:-1]][keep, ...],
                            weights[slice_[:-1]][keep, ...], data[(t,)+slice_[1:]],
                            1, arch=device)
                    # add best contribution from this channel to
                    # the average inter-template CC
                    intertp_cc[t, keep] += np.max(cc, axis=-1)
        # make the CC matrix symmetric by averaging the lower
        # and upper triangles
        intertp_cc = (intertp_cc + intertp_cc.T)/2.
        self.intertp_cc = pd.DataFrame(
                columns=[tid for tid in self.tids],
                index=[tid for tid in self.tids],
                data=intertp_cc)

    # -------------------------------------------
    #       GrowClust related methods
    # -------------------------------------------
    def cross_correlate(self, duration, offset_start_S, offset_start_P,
                        max_lag=20, n_stations=30):
        """
        Create an EventFamily instance to access its methods.
         --- Should be rewritten properly ---
        """
        family = EventFamily(
                self.tids[0], self.db_path_T, self.db_path_M, db_path=self.db_path)
        family.detection_waveforms = \
                np.float32([np.mean(stack.data, axis=0) for stack in self.stacks])
        family.n_events = len(self.tids)
        # trim waveforms
        family.trim_waveforms(duration, offset_start_S, offset_start_P)
        # cross-correlate trimmed waveforms
        family.cross_correlate(
                n_stations=n_stations, max_lag=max_lag, device='precise')
        self.CCs_stations = family.stations
        new_shape = (self.n_templates, self.n_templates, -1)
        self.CCs_P = family.CCs_P.reshape(new_shape)
        self.CCs_S = family.CCs_S.reshape(new_shape)
        self.lags_P = family.lags_P.reshape(new_shape)
        self.lags_S = family.lags_S.reshape(new_shape)
        self.max_lag = max_lag
        del family

    def read_GrowClust_output(self, filename, path, add_results_to_db=False):
        print('Reading GrowClust output from {}'.
                format(os.path.join(path, filename)))
        ot_, lon_, lat_, dep_, err_h_, err_v_, err_t_, tids_ = \
                [], [], [], [], [], [], [], []
        with open(os.path.join(path, filename), 'r') as f:
            for line in f.readlines():
                line = line.split()
                year, month, day, hour, minu, sec = line[:6]
                # correct date if necessary
                if int(day) == 0:
                    date_ = udt(f'{year}-{month}-01')
                    date_ -= datetime.timedelta(days=1)
                    year, month, day = date_.year, date_.month, date_.day
                # correct seconds if necessary
                sec = float(sec)
                if sec == 60.:
                    sec -= 0.001
                ot_.append(udt(f'{year}-{month}-{day}T{hour}:{minu}:{sec}').timestamp)
                tid = int(line[6])
                latitude, longitude, depth = list(map(float, line[7:10]))
                lon_.append(longitude)
                lat_.append(latitude)
                dep_.append(depth)
                mag = float(line[10])
                q_id, cl_id, cluster_pop = list(map(int, line[11:14]))
                n_pairs, n_P_dt, n_S_dt = list(map(int, line[14:17]))
                rms_P, rms_S = list(map(float, line[17:19]))
                err_h, err_v, err_t = list(map(float, line[19:22])) # errors in km and sec
                err_h_.append(err_h)
                err_v_.append(err_v)
                err_t_.append(err_t)
                latitude_init, longitude_init, depth_init =\
                        list(map(float, line[22:25]))
                tids_.append(tid)
        for t, tid in enumerate(tids_):
            tt = self.tids_map[tid]
            self.templates[tt].relocated_latitude = lat_[t]
            self.templates[tt].relocated_longitude = lon_[t]
            self.templates[tt].relocated_depth = dep_[t]
            self.templates[tt].reloc_err_h = err_h_[t]
            self.templates[tt].reloc_err_v = err_v_[t]
            if add_results_to_db:
                keys = ['relocated_longitude', 'relocated_latitude',
                        'relocated_depth', 'reloc_err_h', 'reloc_err_v']
                with h5.File(os.path.join(
                    self.db_path, self.db_path_T, f'template{tid}meta.h5'), 'a') as f:
                    for key in keys:
                        if key in f.keys():
                            del f[key]
                        f.create_dataset(key, data=getattr(self.templates[tt], key))

    def write_GrowClust_stationlist(self, filename, path,
                                    network_filename='all_stations.in'):
        """
        This routine assumes that cross_correlate was called
        shortly before and that self.template still has the same
        set of stations as the ones used for the inter-event CCs.
        """
        net = Network(network_filename)
        net.read()
        subnet = net.subset(
                self.CCs_stations, net.components, method='keep')
        with open(os.path.join(path, filename), 'w') as f:
            for s in range(len(subnet.stations)):
                f.write('{:<5}\t{:.6f}\t{:.6f}\t{:.3f}\n'.
                        format(subnet.stations[s], subnet.latitude[s], 
                               subnet.longitude[s], -1000.*subnet.depth[s]))

    def write_GrowClust_eventlist(self, filename, path):
        from obspy.core import UTCDateTime as udt
        # fake date
        ot = udt('2000-01-01')
        # fake mag
        mag = 1.
        with open(os.path.join(path, filename), 'w') as f:
            for t, tid in enumerate(self.tids):
                # all events are given the template location
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
                        format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
                               ot.second, self.templates[t].latitude,
                               self.templates[t].longitude, self.templates[t].depth,
                               mag, tid))

    def write_GrowClust_CC(self, filename, path, CC_threshold=0.):
        if not hasattr(self, 'CCs_S'):
            print('Need to cross_correlate first.')
            return
        sr = self.templates[0].sampling_rate
        with open(os.path.join(path, filename), 'w') as f:
            for t1, tid1 in enumerate(self.tids):
                for t2, tid2 in enumerate(self.tids):
                    if t2 == t1:
                        continue
                    f.write('#\t{}\t{}\t0.0\n'.format(tid1, tid2))
                    for s in range(len(self.CCs_stations)):
                        # CCs that are zero are pairs that have to be skipped
                        if self.CCs_S[t1, t2, s] > CC_threshold:
                            f.write('  {:>5} {} {:.4f} S\n'.
                                    format(self.CCs_stations[s],
                                           self.lags_S[t1, t2, s]/sr,
                                           self.CCs_S[t1, t2, s]))
                        if self.CCs_P[t1, t2, s] > CC_threshold:
                            f.write('  {:>5} {} {:.4f} P\n'.
                                    format(self.CCs_stations[s],
                                           self.lags_P[t1, t2, s]/sr,
                                           self.CCs_P[t1, t2, s]))

class Stack(object):

    def __init__(self,
                 stations,
                 components,
                 tid=None,
                 sampling_rate=cfg.sampling_rate):

        self.stations = stations
        self.components = components
        self.sampling_rate = sampling_rate
        if isinstance(self.stations, str):
            self.stations = [self.stations]
        if isinstance(self.components, str):
            self.components = [self.components]
        if not isinstance(self.stations, list):
            self.stations = list(self.stations)
        if not isinstance(self.components, list):
            self.components = list(self.components)
        if tid is not None:
            self.template_idx = tid

    def add_data(self, waveforms):

        self.waveforms = waveforms
        self.traces = obs.Stream()
        for s, sta in enumerate(self.stations):
            for c, cp in enumerate(self.components):
                tr = obs.Trace()
                tr.data = self.waveforms[s, c, :]
                tr.stats.station = sta
                # not really a channel, but the component
                tr.stats.channel = cp
                tr.stats.sampling_rate = self.sampling_rate
                self.traces += tr

    def SVDWF_stack(self, detection_waveforms, freqmin, freqmax,
                    expl_var=0.4, max_singular_values=5,
                    wiener_filter_colsize=None):
        filtered_data = np.zeros_like(detection_waveforms)
        for s in range(len(self.stations)):
            for c in range(len(self.components)):
                filtered_data[:, s, c, :] = utils.SVDWF(
                        detection_waveforms[:, s, c, :],
                        max_singular_values=max_singular_values,
                        expl_var=expl_var,
                        freqmin=freqmin,
                        freqmax=freqmax,
                        sampling_rate=self.sampling_rate,
                        wiener_filter_colsize=wiener_filter_colsize)
                if np.sum(filtered_data[:, s, c, :]) == 0:
                    print('Problem with station {} ({:d}), component {} ({:d})'.
                            format(self.stations[s], s, self.components[c], c))
        stacked_waveforms = np.mean(filtered_data, axis=0)
        norm = np.max(stacked_waveforms, axis=-1)[..., np.newaxis]
        norm[norm == 0.] = 1.
        stacked_waveforms /= norm
        self.add_data(stacked_waveforms)
        self.data = filtered_data

    def read_data(self,
                  filename,
                  db_path_S,
                  db_path=cfg.dbpath):

        with h5.File(os.path.join(db_path, db_path_S,
            '{}meta.h5'.format(filename)), mode='r') as f:
            file_stations = f['stations'][()].astype('U').tolist()
            file_components = f['components'][()].astype('U').tolist()
        with h5.File(os.path.join(db_path, db_path_S,
            '{}wav.h5'.format(filename)), mode='r') as f:
            file_waveforms = f['waveforms'][()]
        station_map = []
        for s in range(len(self.stations)):
            station_map.append(file_stations.index(self.stations[s]))
        component_map = []
        for c in range(len(self.components)):
            component_map.append(file_components.index(self.components[c]))
        station_map = np.int32(station_map)
        component_map = np.int32(component_map)
        self.waveforms = file_waveforms[station_map, :, :][:, component_map, :]
        self.traces = obs.Stream()
        for s, sta in enumerate(self.stations):
            for c, cp in enumerate(self.components):
                tr = obs.Trace()
                tr.data = self.waveforms[s, c, :]
                tr.stats.station = sta
                # not really a channel, but the component
                tr.stats.channel = cp
                tr.stats.sampling_rate = self.sampling_rate
                self.traces += tr

class Catalog(object):

    def __init__(self, filename, db_path_M, db_path=cfg.dbpath):
        self.filename = filename
        self.db_path_M = db_path_M
        self.db_path = db_path
        self.full_filename = os.path.join(
                db_path, db_path_M, filename)

    def read_data(self, items_in=[], items_out=[]):
        """
        Attach the requested attributes to the Catalog instance.

        Parameters
        -----------
        items_in: list of strings, default to an empty list
            List of items to read from the catalog file.
            If an empty list is provided, then all items are read.
        items_out: list of strings, default to an empty list
            List of items to reject when reading from the catalog file.
            If an empty list is provided, no items are discarded.
        """
        if not isinstance(items_in, list):
            items_in = [items_in]
        if not isinstance(items_out, list):
            items_out = [items_out]
        # attach 'template_idx' as a default item
        items_in = list(set(items_in+['template_idx']))
        with h5.File(self.full_filename, mode='r') as f:
            for key in f.keys():
                if key in items_out:
                    continue
                elif (key in items_in) or len(items_in) == 1:
                    setattr(self, key, f[key][()])
        # temporary:
        if hasattr(self, 'location'):
            self.latitude, self.longitude, self.depth =\
                    self.location
        # alias
        self.tid = self.template_idx

    def flatten_catalog(self, attributes=[], unique_events=False):
        """
        Outputs a catalog with one row for each requested attribute.

        Parameters
        -----------
        attributes: list of strings, default to an empty list
            List of all the attributes, in addition to origin_times
            and tids, that will be included in the flat catalog.
        unique_events: boolean, default to False
            If True, only returns the events flagged as unique.

        Returns
        ------------
        flat_catalog: dictionary
            Dictionary with one entry for each requested attribute.
            Each entry contains an array of size n_events.
        """
        if not hasattr(self, 'origin_times'):
            print('Catalog needs to have the origin_times attribute '
                  'to return a flat catalog.')
            return
        n_events = len(self.origin_times)
        flat_catalog = {}
        flat_catalog['origin_times'] = np.asarray(self.origin_times)
        flat_catalog['tids'] = np.ones(n_events, dtype=np.int32)*self.tid
        for attr in attributes:
            if not hasattr(self, attr):
                print(f'Catalog does not have {attr}')
                continue
            attr_ = getattr(self, attr)
            if not isinstance(attr_, list)\
                    and not isinstance(attr_, np.ndarray):
                flat_catalog[attr] = \
                        np.ones(n_events, dtype=np.dtype(type(attr_)))*attr_
            else:
                #flat_catalog[attr] = np.asarray(attr_)[selection]
                flat_catalog[attr] = np.asarray(attr_)
                if flat_catalog[attr].shape[0] != n_events:
                    # this condition should work for spotting
                    # list-like attributes that are not of the
                    # shape (n_events, n_attr)
                    flat_catalog[attr] = np.repeat(flat_catalog[attr], n_events).\
                            reshape((n_events,)+flat_catalog[attr].shape)
                else:
                    flat_catalog[attr] = flat_catalog[attr]
        if unique_events:
            selection = self.unique_events
            for attr in flat_catalog.keys():
                flat_catalog[attr] = flat_catalog[attr][selection]
        return flat_catalog

    def return_as_dic(self, attributes=[]):
        catalog = {}
        for attr in attributes:
            if hasattr(self, attr):
                catalog[attr] = getattr(self, attr)
            else:
                print(f'Template {self.tid} catalog has no attribute {attr}')
        return catalog

class AggregatedCatalogs(object):

    def __init__(self, catalogs=None, filenames=None,
                 db_path_M=None, db_path=cfg.dbpath):
        """
        Must either provide catalogs (list of instances of Catalog),
        or list of filenames.
        """
        if catalogs is not None:
            # index the single-template catalogs by their tid
            # in a dictionary
            if isinstance(catalog.template_idx, np.ndarray):
                self.catalogs = {catalog.template_idx[0]: catalog
                                 for catalog in catalogs}
            else:
                self.catalogs = {catalog.template_idx: catalog
                                 for catalog in catalogs}
            self.tids = list(self.catalogs.keys())
        else:
            self.db_path = db_path
            self.db_path_M = db_path_M
            self.filenames = filenames

    def add_recurrence_times(self):
        for tid in self.tids:
            self.catalogs[tid].recurrence_times =\
                    np.hstack(([np.nan], np.diff(self.catalogs[tid].origin_times)))

    def read_data(self, items_in=[], items_out=[]):
        """
        Attach the requested attributes to the Catalog instances.

        Parameters
        -----------
        items_in: list of strings, default to an empty list
            List of items to read from the catalog file.
            If an empty list is provided, then all items are read.
        items_out: list of strings, default to an empty list
            List of items to reject when reading from the catalog file.
            If an empty list is provided, no items are discarded.
        """
        if not isinstance(items_in, list):
            items_in = [items_in]
        if not isinstance(items_out, list):
            items_out = [items_out]
        # initialize the dictionary of Catalog instances
        self.catalogs = {}
        for filename in self.filenames:
            # initialize a Catalog instance
            catalog = Catalog(filename, self.db_path_M, db_path=self.db_path)
            # attach the requested attributes
            catalog.read_data(items_in=items_in, items_out=items_out)
            # fill the dictionary
            if isinstance(catalog.template_idx, np.ndarray):
                self.catalogs[catalog.template_idx[0]] = catalog
            else:
                self.catalogs[catalog.template_idx] = catalog
        self.tids = list(self.catalogs.keys())

    def flatten_catalog(self, attributes=[], chronological_order=True,
                        unique_events=False):
        flat_catalogs = [self.catalogs[tid].flatten_catalog(
            attributes=attributes, unique_events=unique_events)
            for tid in self.tids]
        flat_agg_catalog = {}
        for attr in flat_catalogs[0].keys():
            flat_agg_catalog[attr] = \
                    np.concatenate(
                            [flat_cat[attr] for flat_cat in flat_catalogs],
                             axis=0)
        if chronological_order:
            order = np.argsort(flat_agg_catalog['origin_times'])
            for attr in flat_agg_catalog.keys():
                flat_agg_catalog[attr] = flat_agg_catalog[attr][order]
        return flat_agg_catalog

    def remove_multiples(self, db_path_T, n_closest_stations=10,
                         dt_criterion=3., distance_criterion=1.,
                         similarity_criterion=-1., return_catalog=False):
        """
        Search for events detected by multiple templates.

        Parameters
        -----------
        db_path_T: string
            Name of the directory where template files are stored.
        n_closest_stations: integer, default to 10
            In case template similarity is taken into account,
            this is the number of stations closest to each template
            that are used in the calculation of the average cc.
        dt_criterion: float, default to 3
            Time interval, in seconds, under which two events are
            examined for redundancy.
        distance_criterion: float, default to 1
            Distance threshold, in kilometers, between two uncertainty
            ellipsoids under which two events are examined for redundancy.
        similarity_criterion: float, default to -1
            Template similarity threshold, in terms of average CC, over
            which two events are examined for redundancy. The default
            value of -1 is always verified, meaning that similarity is
            actually not taken into account.
        return_catalog: boolean, default to False
            If True, returns a flatten catalog.
        """
        self.db_path_T = db_path_T
        catalog = self.flatten_catalog(
                attributes=['latitude', 'longitude', 'depth',
                            'correlation_coefficients'])
        # define an alias for tids
        catalog['template_ids'] = catalog['tids']
        self.TpGroup = TemplateGroup(self.tids, db_path_T)
        self.TpGroup.attach_ellipsoid_distances(substract_errors=True)
        if similarity_criterion > -1.:
            self.TpGroup.template_similarity(distance_threshold=distance_criterion,
                                             n_stations=n_closest_stations)
        # -----------------------------------
        t1 = give_time()
        print('Searching for events detected by multiple templates')
        print('All events occurring within {:.1f} sec, with uncertainty '
              'ellipsoids closer than {:.1f} km will and '
              'inter-template CC larger than {:.2f} be considered the same'.
              format(dt_criterion, distance_criterion, similarity_criterion))
        n_events = len(catalog['origin_times'])
        unique_events = np.ones(n_events, dtype=np.bool)
        for n1 in range(n_events):
            if not unique_events[n1]:
                continue
            tid1 = catalog['template_ids'][n1]
            # apply the time criterion
            dt_n1 = (catalog['origin_times'] - catalog['origin_times'][n1])
            temporal_neighbors = (dt_n1 < dt_criterion) & (dt_n1 >= 0.)\
                                & unique_events
            # comment this line if you keep best CC
            #temporal_neighbors[n1] = False
            # get indices of where the above selection is True
            candidates = np.where(temporal_neighbors)[0]
            if len(candidates) == 0:
                continue
            # get template ids of all events that passed the time criterion
            tids_candidates = np.int32([catalog['template_ids'][idx]
                                       for idx in candidates])
            # apply the spatial criterion to the distance between
            # uncertainty ellipsoids
            ellips_dist = self.TpGroup.ellipsoid_distances[tid1].\
                    loc[tids_candidates].values
            if similarity_criterion > -1.:
                similarities = self.TpGroup.intertp_cc[tid1].\
                        loc[tids_candidates].values
                multiples = candidates[np.where(
                    (ellips_dist < distance_criterion)\
                   & (similarities >= similarity_criterion))[0]]
            else:
                multiples = candidates[np.where(
                    ellips_dist < distance_criterion)[0]]
            # comment this line if you keep best CC
            #if len(multiples) == 0:
            #    continue
            # uncomment if you keep best CC
            if len(multiples) == 1:
                continue
            else:
                unique_events[multiples] = False
                # find best CC and keep it
                ccs = catalog['correlation_coefficients'][multiples]
                best_cc = multiples[ccs.argmax()]
                unique_events[best_cc] = True
        t2 = give_time()
        print('{:.2f}s to flag the multiples'.format(t2-t1))
        # -------------------------------------------
        catalog['unique_events'] = unique_events
        for tid in self.tids:
            selection = catalog['tids'] == tid
            unique_events_t = catalog['unique_events'][selection]
            self.catalogs[tid].unique_events = unique_events_t
        if return_catalog:
            return catalog


class EventFamily(object):

    def __init__(self, tid, db_path_T, db_path_M, db_path=cfg.dbpath):
        """
        Initializes an EventFamily instance and attaches the
        Template instance corresponding to this family.
        """
        self.tid = tid
        self.db_path_T = db_path_T
        self.db_path_M = db_path_M
        self.db_path = db_path
        self.template = Template(f'template{tid}', db_path_T, db_path=db_path)
        self.template.read_waveforms()
        self.sr = self.template.sampling_rate

    def attach_catalog(self, items_in=[], items_out=[]):
        """
        Creates a Catalog instance and call Catalog.read_data()
        """
        filename = f'multiplets{self.tid}catalog.h5'
        self.catalog = Catalog(filename, self.db_path_M, db_path=self.db_path)
        self.catalog.read_data(items_in=items_in, items_out=items_out)

    def check_template_reloc(self):
        """
        If templates were relocated, overwrite the catalog's location
        with the template's relocated hypocenter
        """
        for attr in ['longitude', 'latitude', 'depth']:
            if hasattr(self.template, 'relocated_'+attr):
                setattr(self.catalog, attr, getattr(self.template, 'relocated_'+attr))

    def find_closest_stations(self, n_stations, available_stations=None):
        """
        Here for consistency with EventFamilyGroup and write the
        cross_correlate method such that EventFamilyGroup can inherit
        from it.
        """
        self.template.n_closest_stations(
                n_stations, available_stations=available_stations)
        self.stations = self.template.stations
        self.map_to_subnet = self.template.map_to_subnet

    def read_data(self, **kwargs):
        """
        Call fetch_detection_waveforms from utils.
        Read waveforms from the event waveforms that were
        extracted at the time of detection.
        """
        # self.event_ids tell us in which order the events were read,
        # which depends on the kwargs given to fetch_detection_waveforms
        # force ordering to be chronological
        kwargs['ordering'] = 'origin_times'
        kwargs['flip_order'] = True # why did I do that?? This seems totally unnecessary
        if (kwargs.get('unique_events', False)\
                and np.sum(self.catalog.unique_events) == 0):
            self.detection_waveforms = []
            self.event_ids, self.event_ids_str = [], []
            self.n_events = 0
        else:
            self.detection_waveforms, _, self.event_ids = \
                    utils.fetch_detection_waveforms(
                            self.tid, self.db_path_T, self.db_path_M,
                            return_event_ids=True, **kwargs)
            self.n_events = self.detection_waveforms.shape[0]
            self.event_ids_str = [f'{self.tid},{event_id}' for event_id in self.event_ids]

    def trim_waveforms(self, duration, offset_start_S, offset_start_P,
                       t0=cfg.buffer_extracted_events,
                       S_window_time=4., P_window_time=1.):
        """
        Trim the waveforms using the P- and S-wave moveouts from the template.

        Parameters
        ------------
        duration: scalar, float
            Duration, in seconds, of the trimmed windows.
        offset_start_S: scalar, float
            Time, in seconds, taken BEFORE the S wave.
        offset_start_P: scalar, float
            Time, in seconds, taken BEFORE the P wave.
        t0: float, optional
            Time, in seconds, taken before the detection time
            when the waveforms were extracted at the time
            of detection. Default to the value written in
            the parameter file.
        S_window_time: scalar, float
            Time between the beginning of the S-wave template
            window and the predicted S-wave arrival time.
        P_window_time: scalar, float
            Time between the beginning of the P-wave template
            widow and the predicted P-wave arrival time.
        """
        if not hasattr(self, 'detection_waveforms'):
            print('Need to call read_data first.')
            return
        if self.n_events == 0:
            print('No events were read, probably because only '
                  'unique events were requested.')
            return
        # convert all times from seconds to samples
        duration = utils.sec_to_samp(duration, sr=self.sr)
        offset_start_S = utils.sec_to_samp(offset_start_S, sr=self.sr)
        offset_start_P = utils.sec_to_samp(offset_start_P, sr=self.sr)
        S_window_time = utils.sec_to_samp(S_window_time, sr=self.sr)
        P_window_time = utils.sec_to_samp(P_window_time, sr=self.sr)
        t0 = utils.sec_to_samp(t0, sr=self.sr)
        new_shape = self.detection_waveforms.shape[:-1] + (duration,)
        self.trimmed_waveforms = np.zeros(new_shape, dtype=np.float32)
        _, n_stations, _, n_samples = self.detection_waveforms.shape
        for s in range(n_stations):
            # P-wave window on vertical components
            P_start = t0 + self.template.network_p_moveouts[s] + P_window_time - offset_start_P
            P_end = P_start + duration
            if P_start < n_samples:
                P_end = min(n_samples, P_end)
                self.trimmed_waveforms[:, s, 2, :P_end-P_start] = \
                        self.detection_waveforms[:, s, 2, P_start:P_end]
            # S-wave window on horizontal components
            S_start = t0 + self.template.network_s_moveouts[s] + S_window_time - offset_start_S
            S_end = S_start + duration
            if S_start < n_samples:
                S_end = min(n_samples, S_end)
                self.trimmed_waveforms[:, s, :2, :S_end-S_start] = \
                        self.detection_waveforms[:, s, :2, S_start:S_end]

    def read_trimmed_waveforms(self, duration, offset_start, net, target_SR,
                               tt_phases=['S', 'S', 'P'], norm_rms=True,
                               buffer=2., unique_events=False, **preprocess_kwargs):
        """
        Read waveforms from raw data and refilter/resample. Extra key-word
        arguments will be passed to the preprocessing routine.
        """
        from . import event_extraction
        if not hasattr(self, 'catalog'):
            self.attach_catalog()
        preprocess_kwargs['target_SR'] = target_SR
        detection_waveforms  = []
        # reshape travel times
        station_indexes = np.int32([self.template.network_stations.tolist().index(sta)
                for sta in net.stations])
        # use travel times according to requested phase on each channel
        phase_index = {'S': 1, 'P': 0}
        tts = np.stack([self.template.network_travel_times\
                [station_indexes, phase_index[tt_phases[c]]]
            for c in range(len(tt_phases))], axis=1)
        if unique_events:
            selection = self.catalog.unique_events
        else:
            selection = np.ones(len(self.catalog.origin_times), dtype=np.bool)
        print(f'Reading {np.sum(selection)} events...')
        for ot in self.catalog.origin_times[selection]:
            event = event_extraction.extract_event_realigned(
                    ot, net, tts, duration=duration+2.*buffer,
                    offset_start=offset_start-buffer, folder='raw')
            event = event_extraction.preprocess_event(
                    event, target_duration=duration+2.*buffer,
                    **preprocess_kwargs)
            if len(event) > 0:
                detection_waveforms.append(utils.get_np_array(
                    event, net, verbose=False))
            else:
                detection_waveforms.append(np.zeros(
                    len(net.stations), len(net.components),
                    utils.sec_to_samp(target_duration, sr=target_SR),
                    dtype=np.float32))
        detection_waveforms = np.stack(detection_waveforms, axis=0)
        if norm_rms:
            # one normalization factor for each 3-comp seismogram
            norm = np.std(detection_waveforms, axis=(2, 3))[..., np.newaxis, np.newaxis]
            norm[norm == 0.] = 1.
            detection_waveforms /= norm
        # trim the waveforms
        buffer = utils.sec_to_samp(buffer, sr=target_SR)
        self.trimmed_waveforms = detection_waveforms[..., buffer:-buffer]
        # update SR
        self.sr = target_SR
        # add metadata
        self.n_events = self.trimmed_waveforms.shape[0]
        self.event_ids = np.arange(len(self.catalog.origin_times), dtype=np.int32)
        self.event_ids = self.event_ids[selection]
        self.event_ids_str = [f'{self.tid},{event_id}' for event_id in self.event_ids]

    def read_trimmed_waveforms_raw(
            self, duration, offset_start, net,
            tt_phases=['S', 'S', 'P'],
            buffer=2., unique_events=False, **preprocess_kwargs):
        """
        Read waveforms from raw data and remove instrument response if requested.
        Extra key-word arguments will be passed to the preprocessing routine.
        This should not be used to resample all traces to the same sampling
        rate, instead use read_trimmed_waveforms.
        """
        from . import event_extraction
        if not hasattr(self, 'catalog'):
            self.attach_catalog()
        detection_waveforms  = []
        # reshape travel times
        station_indexes = np.int32([self.template.network_stations.tolist().index(sta)
                for sta in net.stations])
        # use travel times according to requested phase on each channel
        phase_index = {'S': 1, 'P': 0}
        tts = np.stack([self.template.network_travel_times\
                [station_indexes, phase_index[tt_phases[c]]]
            for c in range(len(tt_phases))], axis=1)
        if unique_events:
            selection = self.catalog.unique_events
        else:
            selection = np.ones(len(self.catalog.origin_times), dtype=np.bool)
        events = []
        for ot in self.catalog.origin_times[selection]:
            event = event_extraction.extract_event_realigned(
                    ot, net, tts, duration=duration+2.*buffer,
                    offset_start=offset_start-buffer, folder='raw',
                    attach_response=True)
            event = event_extraction.preprocess_event(
                    event, target_duration=duration+2.*buffer,
                    **preprocess_kwargs)
            # now that the preprocessing is done, remove the sides
            for tr in event:
                tr.trim(starttime=tr.stats.starttime+buffer,
                        endtime=tr.stats.endtime-buffer)
            events.append(event)
        self.trimmed_events = events
        # add metadata
        self.n_events = len(events)
        self.event_ids = np.arange(len(self.catalog.origin_times), dtype=np.int32)
        self.event_ids = self.event_ids[selection]
        self.event_ids_str = [f'{self.tid},{event_id}' for event_id in self.event_ids]
        if self.n_events != np.sum(selection):
            print('Number of extracted events does not match '
                  f'expected number ({self.n_events} vs {np.sum(selection)}).')


    def cross_correlate(self, n_stations=40, max_lag=10,
                        paired=None, device='cpu', available_stations=None):
        """
        Parameters
        -----------
        n_stations: integer, default to 40
            The number of stations closest to each template used in
            the computation of the average CC.
        max_lag: integer, default to 10
            Maximum lag, in samples, allowed when searching for the
            maximum CC on each channel. This is to account for small
            discrepancies in windowing that could occur for two templates
            highly similar but associated to slightly different locations.
        paired: (n_events, n_events) boolean array, default to None
            If not None, this array is used to determine for which
            events the CC should be computed. This is mostly useful
            when cross correlating large data sets, with potentially
            many redundant events.
        """
        import fast_matched_filter as fmf
        if not hasattr(self, 'trimmed_waveforms'):
            print('The EventFamily instance needs the trimmed_waveforms '
                  'attribute, see trim_waveforms.')
            return
        self.max_lag = max_lag
        self.find_closest_stations(n_stations, available_stations=available_stations)
        print('Finding the best inter-event CCs...')
        # format arrays for FMF
        slice_ = np.index_exp[:, self.map_to_subnet, :, :]
        data_arr = self.trimmed_waveforms.copy()[slice_]
        norm_data = np.std(data_arr, axis=-1)[..., np.newaxis]
        norm_data[norm_data == 0.] = 1.
        data_arr /= norm_data
        template_arr = data_arr[..., max_lag:-max_lag]
        n_stations, n_components = data_arr.shape[1:-1]
        # initialize ouputs
        if paired is None:
            paired = np.ones((self.n_events, self.n_events), dtype=np.bool)
            self.paired = paired
        output_shape = (np.sum(paired), n_stations)
        CCs_S = np.zeros(output_shape, dtype=np.float32)
        lags_S = np.zeros(output_shape, dtype=np.int32)
        CCs_P = np.zeros(output_shape, dtype=np.float32)
        lags_P = np.zeros(output_shape, dtype=np.int32)
        # re-arrange input arrays to pass pieces of array
        data_arr_P = np.ascontiguousarray(data_arr[:, :, 2:3, :])
        template_arr_P = np.ascontiguousarray(template_arr[:, :, 2:3, :])
        moveouts_arr_P = np.zeros((self.n_events, 1, 1), dtype=np.int32)
        weights_arr_P = np.ones((self.n_events, 1, 1), dtype=np.float32)
        data_arr_S = np.ascontiguousarray(data_arr[:, :, :2, :])
        template_arr_S = np.ascontiguousarray(template_arr[:, :, :2, :])
        moveouts_arr_S = np.zeros((self.n_events, 1, 2), dtype=np.int32)
        weights_arr_S = 0.5*np.ones((self.n_events, 1, 2), dtype=np.float32)
        # free some space
        del data_arr, template_arr
        counter = 0
        for n in range(self.n_events):
            print(f'------ {n+1}/{self.n_events} -------')
            selection = paired[n, :]
            counter_inc = np.sum(selection)
            for s in range(n_stations):
                # use trick to keep station and component dim
                slice_ = np.index_exp[selection, s:s+1, :, :]
                cc_S = fmf.matched_filter(
                        template_arr_S[slice_], moveouts_arr_S[selection, ...],
                        weights_arr_S[selection, ...], data_arr_S[(n,)+slice_[1:]],
                        1, check_zeros=False, arch=device)
                cc_P = fmf.matched_filter(
                        template_arr_P[slice_], moveouts_arr_P[selection, ...],
                        weights_arr_P[selection, ...], data_arr_P[(n,)+slice_[1:]],
                        1, check_zeros=False, arch=device)
                # get best CC and its lag
                CCs_S[counter:counter+counter_inc, s] = np.max(cc_S, axis=-1)
                lags_S[counter:counter+counter_inc, s] = np.argmax(cc_S, axis=-1) - max_lag
                CCs_P[counter:counter+counter_inc, s] = np.max(cc_P, axis=-1)
                lags_P[counter:counter+counter_inc, s] = np.argmax(cc_P, axis=-1) - max_lag
                # N.B: lags[n1, n2] is the ev1-ev2 time
            counter += counter_inc
        self.CCs_S = CCs_S
        self.lags_S = lags_S
        self.CCs_P = CCs_P
        self.lags_P = lags_P
        self.paired = paired
        self.max_lag = max_lag

    def plot_alignment(self, pair_id, s,
                       components = ['N', 'E', 'Z']):
        """
        Check visually what the max correlation alignment is worth.
        This also demonstrates that we use the following convention:
        Argmax(CC(pair[i, j])) == tt_j - tt_i
        i.e. tt_2 - tt_1 in GrowClust
        """
        import matplotlib.pyplot as plt
        if not hasattr(self, 'CCs_S'):
            print('Need to run self.cross_correlate first!')
            return
        pairs = np.column_stack(np.where(self.paired))
        evid1, evid2 = pairs[pair_id]
        ss = self.map_to_subnet[s]
        fig, axes = plt.subplots(
                num=f'ev{evid1}_ev{evid2}_station_{s}',
                nrows=3, ncols=1, figsize=(18, 9))
        time = np.arange(self.trimmed_waveforms.shape[-1], dtype=np.float32)
        for c in range(len(components)):
            phase = 'S' if c < 2 else 'P'
            axes[c].plot(
                    time, utils.max_norm(self.trimmed_waveforms[evid1, ss, c, :]),
                    color='k', label=f'Ev. {evid1}: {components[c]} cp. - {phase} wave')
            if phase == 'S':
                mv = self.max_lag + self.lags_S[pair_id, s]
                CC = self.CCs_S[pair_id, s]
            else:
                mv = self.max_lag + self.lags_P[pair_id, s]
                CC = self.CCs_P[pair_id, s]
            axes[c].plot(
                    time[:-2*self.max_lag] + mv,
                    utils.max_norm(self.trimmed_waveforms[evid2, ss, c, self.max_lag:-self.max_lag]),
                    color='C3', label=f'Ev. {evid2}: {components[c]} cp. - {phase} wave'
                    f'\nCC={CC:.2f}\nLag: {mv-self.max_lag}sp')
            axes[c].axvline(self.max_lag, color='k')
            axes[c].legend(loc='upper left')
            axes[c].set_ylabel('Normalized Amp.')
            axes[c].set_xlabel('Time (samples)')
        plt.subplots_adjust(hspace=0.3)
        return fig


    # -------------------------------------------
    #       GrowClust related methods
    # -------------------------------------------
    def read_GrowClust_output(self, filename, path):
        print('Reading GrowClust output from {}'.
                format(os.path.join(path, filename)))
        ot_, lon_, lat_, dep_, err_h_, err_v_, err_t_ = \
                [], [], [], [], [], [], []
        with open(os.path.join(path, filename), 'r') as f:
            for line in f.readlines():
                line = line.split()
                year, month, day, hour, minu, sec = line[:6]
                # correct date if necessary
                if int(day) == 0:
                    date_ = udt(f'{year}-{month}-01')
                    date_ -= datetime.timedelta(days=1)
                    year, month, day = date_.year, date_.month, date_.day
                # correct seconds if necessary
                sec = float(sec)
                if sec == 60.:
                    sec -= 0.001
                ot_.append(udt(f'{year}-{month}-{day}T{hour}:{minu}:{sec}').timestamp)
                event_id = int(line[6])
                latitude, longitude, depth = list(map(float, line[7:10]))
                lon_.append(longitude)
                lat_.append(latitude)
                dep_.append(depth)
                mag = float(line[10])
                q_id, cl_id, cluster_pop = list(map(int, line[11:14]))
                n_pairs, n_P_dt, n_S_dt = list(map(int, line[14:17]))
                rms_P, rms_S = list(map(float, line[17:19]))
                err_h, err_v, err_t = list(map(float, line[19:22])) # errors in km and sec
                err_h_.append(err_h)
                err_v_.append(err_v)
                err_t_.append(err_t)
                latitude_init, longitude_init, depth_init =\
                        list(map(float, line[22:25]))
        cat = {}
        cat['origin_time'] = ot_
        cat['longitude'] = lon_
        cat['latitude'] = lat_
        cat['depth'] = dep_
        cat['error_hor'] = err_h_
        cat['error_ver'] = err_v_
        cat['error_t'] = err_t_
        self.relocated_catalog =\
                pd.DataFrame(data=cat)

    def write_GrowClust_stationlist(self, filename, path,
                                    network_filename='all_stations.in'):
        """
        This routine assumes that cross_correlate was called
        shortly before and that self.template still has the same
        set of stations as the ones used for the inter-event CCs.
        """
        net = Network(network_filename)
        net.read()
        subnet = net.subset(
                self.stations, net.components, method='keep')
        with open(os.path.join(path, filename), 'w') as f:
            for s in range(len(subnet.stations)):
                f.write('{:<5}\t{:.6f}\t{:.6f}\t{:.3f}\n'.
                        format(subnet.stations[s], subnet.latitude[s], 
                               subnet.longitude[s], -1000.*subnet.depth[s]))

    def write_GrowClust_eventlist(self, filename, path):
        from obspy.core import UTCDateTime as udt
        if not hasattr(self, 'catalog'):
            self.attach_catalog()
        with open(os.path.join(path, filename), 'w') as f:
            for n in range(self.n_events):
                nn = self.event_ids[n]
                ot = udt(self.catalog.origin_times[nn])
                if hasattr(self.catalog, relocated_catalog):
                    # start from the relocated catalog
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
                            format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
                                   ot.second, self.catalog.relocated_latitude[nn],
                                   self.catalog.relocated_longitude[nn],
                                   self.catalog.relocated_depth[nn],
                                   self.catalog.magnitudes[nn], self.event_ids[n]))
                else:
                    # all events are given the template location
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
                            format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
                                   ot.second, self.template.latitude,
                                   self.template.longitude, self.template.depth,
                                   self.catalog.magnitudes[nn], self.event_ids[n]))

    def write_GrowClust_CC(self, filename, path, CC_threshold=0.):
        if not hasattr(self, 'CCs_S'):
            print('Need to compute the inter-event CCs first.')
            return
        with open(os.path.join(path, filename), 'w') as f:
            n1, n2 = np.meshgrid(np.arange(self.n_events, dtype=np.int32),
                                 np.arange(self.n_events, dtype=np.int32),
                                 indexing='ij')
            n1  = n1.flatten()[self.paired.flatten()]
            n2  = n2.flatten()[self.paired.flatten()]
            for n in range(len(n1)):
                f.write('#\t{}\t{}\t0.0\n'.
                        format(self.event_ids[n1[n]], self.event_ids[n2[n]]))
                for s in range(len(self.stations)):
                    # CCs that are zero are pairs that have to be skipped
                    if self.CCs_S[n, s] > CC_threshold:
                        f.write('  {:>5} {} {:.4f} S\n'.
                                format(self.stations[s],
                                       self.lags_S[n, s]/self.sr,
                                       self.CCs_S[n, s]))
                    if self.CCs_P[n, s] > CC_threshold:
                        f.write('  {:>5} {} {:.4f} P\n'.
                                format(self.stations[s],
                                       self.lags_P[n, s]/self.sr,
                                       self.CCs_P[n, s]))

class EventFamilyGroup(EventFamily):

    def __init__(self, tids, db_path_T, db_path_M, db_path=cfg.dbpath):
        self.tids = tids
        self.n_templates = len(tids)
        self.db_path_T = db_path_T
        self.db_path_M = db_path_M
        self.db_path = db_path
        self.families = {tid: EventFamily(tid, db_path_T, db_path_M, db_path=db_path)
                         for tid in tids}
        self.sr = self.families[self.tids[0]].sr

    def check_template_reloc(self):
        """
        If templates were relocated, overwrite the catalog's location
        with the template's relocated hypocenter
        """
        if not hasattr(self, 'aggcat'):
            print('This EventFamilyGroup instance has no AggregatedCatalogs. '
                  'Do nothing.')
        else:
            for tid in self.tids:
                template = self.families[tid].template
                for attr in ['longitude', 'latitude', 'depth']:
                    if hasattr(template, 'relocated_'+attr):
                        setattr(self.aggcat.catalogs[tid], attr, getattr(template, 'relocated_'+attr))

    def attach_catalog(self, dt_criterion=3., distance_criterion=5.,
                       similarity_criterion=0.33, n_closest_stations=10,
                       items_in=[]):
        """
        Creates an AggregatedCatalogs instance and
        call AggregatedCatalogs.read_data(), as well as
        AggregatedCatalogs.remove_multiples(). This is crucial to note
        that remove_multiples returns a catalog that is ordered in time.
        When reading data from the single-template catalogs, these are
        not ordered in time after concatenation.
        """
        # force 'origin_times' and 'magnitudes' to be among the items
        items_in = items_in + ['origin_times', 'magnitudes',
                    'correlation_coefficients', 'unique_events']
        filenames = [f'multiplets{tid}catalog.h5' for tid in self.tids]
        self.aggcat = AggregatedCatalogs(
                filenames=filenames, db_path_M=self.db_path_M, db_path=self.db_path)
        self.aggcat.read_data(items_in=items_in + ['location'])
        self.check_template_reloc()
        self.catalog = self.aggcat.remove_multiples(
                self.db_path_T, dt_criterion=dt_criterion,
                distance_criterion=distance_criterion,
                similarity_criterion=similarity_criterion,
                n_closest_stations=n_closest_stations, return_catalog=True)
        # load the rest of the requested attributes
        rest = [item for item in items_in if item not in self.catalog.keys()]
        self.catalog.update(self.aggcat.flatten_catalog(
            attributes=rest, chronological_order=True))

    def find_closest_stations(self, n_stations, available_stations=None):
        # overriden method from parent class
        stations = []
        for tid in self.tids:
            self.families[tid].template.n_closest_stations(
                    n_stations, available_stations=available_stations)
            stations.extend(self.families[tid].template.stations)
        stations, counts = np.unique(stations, return_counts=True)
        sorted_ind = np.argsort(counts)[::-1]
        self.stations = stations[sorted_ind[:n_stations]]
        network_stations = self.families[self.tids[0]].template.network_stations
        self.map_to_subnet = np.int32([np.where(network_stations == sta)[0]
                                       for sta in self.stations]).squeeze()

    def pair_events(self, random_pairing_frac=0., random_max=2):
        if not hasattr(self, 'catalog'):
            # call attach_catalog() for calling remove_multiples()
            # !!! the catalog that is returned like that takes all
            # single-template catalogs and order them in time, so
            # it is ESSENTIAL to make sure that the data are ordered
            # the same way
            self.attach_catalog()
        # all valid events are the unique events and also
        # the events with highest CC from each family, to make
        # sure all families will end up being paired by at least one event
        valid_events = self.catalog['unique_events'].copy()
        highest_CC_events_idx = {}
        for tid in self.tids:
            selection = self.catalog['tids'] == tid
            highest_CC_events_idx[tid] = np.where(
                self.catalog['correlation_coefficients']
                == self.catalog['correlation_coefficients'][selection].max())[0][0]
            valid_events[highest_CC_events_idx[tid]] = True
        self.paired = np.zeros((self.n_events, self.n_events), dtype=np.bool)
        for n in range(self.n_events):
            if valid_events[n]:
                # if this is a valid event, pair it with
                # all other valid events
                self.paired[n, valid_events] = True
            # in all cases, pair the events with all other
            # events of the same family
            tid = self.catalog['tids'][n]
            self.paired[n, self.catalog['tids'] == tid] = True
            # OR --------------------
            # link non-unique events only to their best CC event
            #self.paired[n, highest_CC_events_idx[tid]] = True
            # -------------------
            # and add a few randomly selected connections
            unpaired = np.where(~self.paired[n, :])[0]
            n_random = min(random_max, int(random_pairing_frac*len(unpaired)))
            if n_random > 0:
                if len(unpaired) > 0:
                    random_choice = np.random.choice(
                            unpaired, size=n_random, replace=False)
                    self.paired[n, random_choice] = True
        np.fill_diagonal(self.paired, False)

    def read_data(self, **kwargs):
        # overriden method from parent class
        for tid in self.tids:
            if hasattr(self, 'aggcat'):
                # use the already loaded, and potentially edited, catalog
                kwargs['catalog'] = self.aggcat.catalogs[tid]
            self.families[tid].read_data(**kwargs)
        self.event_ids_str = \
                np.asarray([event_id for tid in self.tids
                           for event_id in self.families[tid].event_ids_str])
        self.n_events = len(self.event_ids_str)

    def read_trimmed_waveforms(self, *args, **kwargs):
        """
        See EventFamily.read_trimmed_waveforms
        """
        # overriden method from parent class
        for tid in self.tids:
            self.families[tid].read_trimmed_waveforms(
                    *args, **kwargs)
        # agglomerate all waveforms into one array
        self.trimmed_waveforms = np.concatenate(
                [self.families[tid].trimmed_waveforms for tid in self.tids],
                axis=0)
        # add metadata
        self.event_ids_str = \
                np.asarray([event_id for tid in self.tids
                           for event_id in self.families[tid].event_ids_str])
        self.sr = self.families[self.tids[0]].sr
        # reorder in chronological order
        OT = []
        for tid in self.tids:
            #self.families[tid].attach_catalog(items_in=['origin_times'])
            OT.extend(self.families[tid].catalog
                        .origin_times[self.families[tid].event_ids])
        OT = np.float64(OT) # and now these are the correct origin times
        sorted_ind = np.argsort(OT)
        self.trimmed_waveforms = self.trimmed_waveforms[sorted_ind, ...]
        self.event_ids_str = self.event_ids_str[sorted_ind]
        self.n_events = len(self.event_ids_str)
        self.event_ids = np.arange(self.n_events)

    def read_trimmed_waveforms_raw(self, *args, **kwargs):
        """
        See EventFamily.read_trimmed_waveforms_raw
        """
        # overriden method from parent class
        for tid in self.tids:
            self.families[tid].read_trimmed_waveforms_raw(
                    *args, **kwargs)
        # agglomerate all trimmed events into ine list
        self.trimmed_events = sum(
                [self.families[tid].trimmed_events for tid in self.tids], [])
        # add metadata
        self.event_ids_str = \
                np.asarray([event_id for tid in self.tids
                           for event_id in self.families[tid].event_ids_str])
        # reorder in chronological order
        OT = []
        for tid in self.tids:
            #self.families[tid].attach_catalog(items_in=['origin_times'])
            OT.extend(self.families[tid].catalog
                        .origin_times[self.families[tid].event_ids])
        OT = np.float64(OT) # and now these are the correct origin times
        sorted_ind = np.argsort(OT)
        self.trimmed_events = [self.trimmed_events[i] for i in sorted_ind]
        self.event_ids_str = self.event_ids_str[sorted_ind]
        self.n_events = len(self.event_ids_str)
        self.event_ids = np.arange(self.n_events)

    def trim_waveforms(self, *args, **kwargs):
        """
        See EventFamily.trim_waveforms
        """
        # overriden method from parent class
        for tid in self.tids:
           self.families[tid].trim_waveforms(
                   *args, **kwargs)
        # agglomerate all waveforms into one array
        self.trimmed_waveforms = np.concatenate(
                [self.families[tid].trimmed_waveforms for tid in self.tids
                    if self.families[tid].n_events > 0], axis=0)
        # reorder in chronological order
        #self.attach_catalog()
        # no this is wrong!!! attach_catalog() call remove_multiples()
        # which already merges single-template catalogs and order them
        # in time.... so to get the actual origin times that correspond
        # to the data that were loaded, we need to read from the single-
        # template catalog!!!
        #sorted_ind = np.argsort(self.catalog['origin_times'])
        OT = []
        for tid in self.tids:
            self.families[tid].attach_catalog(items_in=['origin_times'])
            OT.extend(self.families[tid].catalog
                        .origin_times[self.families[tid].event_ids])
        OT = np.float64(OT) # and now these are the correct origin times
        sorted_ind = np.argsort(OT)
        self.trimmed_waveforms = self.trimmed_waveforms[sorted_ind, ...]
        self.event_ids_str = self.event_ids_str[sorted_ind]
        #for attr in self.catalog.keys():
        #    self.catalog[attr] = self.catalog[attr][sorted_ind]
        self.event_ids = np.arange(self.n_events)

    # -------------------------------------------
    #       GrowClust related methods
    # -------------------------------------------
    def read_GrowClust_output(self, filename_out, filename_evid,
                              path_out, path_evid):
        print('Reading GrowClust output from {}'.
                format(os.path.join(path_out, filename_out)))
        print('Reading event ids from {}'.
                format(os.path.join(path_evid, filename_evid)))
        event_ids_map = pd.read_csv(
                os.path.join(path_evid, filename_evid), index_col=0)
        ot_, lon_, lat_, dep_, err_h_, err_v_, err_t_, evids_ = \
                [], [], [], [], [], [], [], []
        with open(os.path.join(path_out, filename_out), 'r') as f:
            for line in f.readlines():
                line = line.split()
                year, month, day, hour, minu, sec = line[:6]
                # correct date if necessary
                if int(day) == 0:
                    date_ = udt(f'{year}-{month}-01')
                    date_ -= datetime.timedelta(days=1)
                    year, month, day = date_.year, date_.month, date_.day
                # correct seconds if necessary
                sec = float(sec)
                if sec == 60.:
                    sec -= 0.001
                ot_.append(udt(f'{year}-{month}-{day}T{hour}:{minu}:{sec}').timestamp)
                event_id = int(line[6])
                evids_.append(event_id)
                latitude, longitude, depth = list(map(float, line[7:10]))
                lon_.append(longitude)
                lat_.append(latitude)
                dep_.append(depth)
                mag = float(line[10])
                q_id, cl_id, cluster_pop = list(map(int, line[11:14]))
                n_pairs, n_P_dt, n_S_dt = list(map(int, line[14:17]))
                rms_P, rms_S = list(map(float, line[17:19]))
                err_h, err_v, err_t = list(map(float, line[19:22])) # errors in km and sec
                err_h_.append(err_h)
                err_v_.append(err_v)
                err_t_.append(err_t)
                latitude_init, longitude_init, depth_init =\
                        list(map(float, line[22:25]))
        # convert all lists to np arrays
        ot_ = np.float64(ot_)
        lon_, lat_, dep_ = np.float32(lon_), np.float32(lat_), np.float32(dep_)
        err_h_, err_v_, err_t_ = \
                np.float32(err_h_), np.float32(err_v_), np.float32(err_t_)
        evids_ = np.int32(evids_)
        # distribute results over corresponding templates
        for tid in self.tids:
            selection = event_ids_map.index == tid
            event_ids_tid = event_ids_map['event_ids'][selection]
            cat = {}
            cat['origin_time'] = ot_[selection]
            cat['longitude'] = lon_[selection]
            cat['latitude'] = lat_[selection]
            cat['depth'] = dep_[selection]
            cat['error_hor'] = err_h_[selection]
            cat['error_ver'] = err_v_[selection]
            cat['error_t'] = err_t_[selection]
            cat['event_ids'] = event_ids_tid
            self.families[tid].relocated_catalog =\
                    pd.DataFrame(data=cat)
        # attach flattened version for convenience
        cat = {}
        cat['origin_time'] = ot_
        cat['longitude'] = lon_
        cat['latitude'] = lat_
        cat['depth'] = dep_
        cat['error_hor'] = err_h_
        cat['error_ver'] = err_v_
        cat['error_t'] = err_t_
        cat['event_ids'] = event_ids_map['event_ids']
        cat['tids'] = event_ids_map.index
        self.relocated_catalog =\
                pd.DataFrame(data=cat)
        self.n_events = len(self.relocated_catalog)
        self.event_ids = np.arange(self.n_events, dtype=np.int32)

    def write_GrowClust_eventlist(self, filename, path, fresh_start=True):
        """
        Note: The different with its single family counterpart is that
        here the catalog is already in the same order as self.event_ids
        (cf. trim_waveforms)
        """
        from obspy.core import UTCDateTime as udt
        if not hasattr(self, 'catalog'):
            self.attach_catalog()
        if hasattr(self, 'relocated_catalog') and not fresh_start:
            print('Give locations from relocated catalog')
        with open(os.path.join(path, filename), 'w') as f:
            for n in range(self.n_events):
                ot = udt(self.catalog['origin_times'][n])
                if hasattr(self, 'relocated_catalog') and not fresh_start:
                    # start from the relocated catalog
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
                            format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
                                   ot.second, self.relocated_catalog['latitude'].iloc[n],
                                   self.relocated_catalog['longitude'].iloc[n],
                                   self.relocated_catalog['depth'].iloc[n],
                                   self.catalog['magnitudes'][n], self.event_ids[n]))
                else:
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0.\t0.\t0.\t{}\n'.
                            format(ot.year, ot.month, ot.day, ot.hour, ot.minute,
                                   ot.second, self.catalog['latitude'][n],
                                   self.catalog['longitude'][n], self.catalog['depth'][n],
                                   self.catalog['magnitudes'][n], self.event_ids[n]))

    def write_GrowClust_eventids(self, filename, path):
        """
        For each flat integer event id used by GrowClust, we write
        the corresponding template id and event id so that we can
        link the relocated events to their template/detection.
        """
        with open(os.path.join(path, filename), 'w') as f:
            # write column names
            f.write('tids,event_ids\n')
            for n in range(self.n_events):
                f.write(f'{self.event_ids_str[n]}\n')
                #sep = self.event_ids_str[n].find('.')
                #f.write('{:d},{:d}\n'.format(
                #    int(self.event_ids_str[n][:sep]),
                #    int(self.event_ids_str[n][sep+1:])))

# --------------------------------------------------
# functions

def directional_errors_(templates=None, tids=None, return_as_pd=True,
                        db_path_T='template_db_2', db_path=cfg.dbpath):
    """
    Computes the length, in km, of the uncertainty ellipsoid in the 
    inter-template direction. This is interesting for checking how
    far two templates are given the uncertainties on their location.

    Parameters
    ------------
    templates: list of dataset.Template, default to None
        If not None, templates is a list of dataset.Template objects
        for which the inter-template directional errors are computed.
    tids: list or array of integers, default to None
        If not None (in the case where templates=None), use the path
        variables db_path_T and db_path to read a list of dataset.Template
        objects.
    return_as_pd: boolean, default to True
        If True, returns the inter-template directional errors
        in a pandas DataFrame.
    db_path_T: string, default to 'template_db_2'
        Folder where template files are stored.
    db_path: string, default to cfg.path
        Root directory where outputs are stored.
    Returns
    ----------
    directional_errors: (n_templates, n_templates) numpy ndarray
                        of pandas DataFrame
        The length, in kilometers, of the uncertainty ellipsoid in the
        inter-template direction.
    """
    from cartopy import crs
    if tids is None and templates is None:
        print('tids or templates should be specified!')
        return
    elif templates is None:
        templates = []
        full_path_tp = os.path.join(db_path, db_path_T)
        print(f'Reading templates from {full_path_tp}')
        for tid in tids:
            templates.append(Template(f'template{tid}', db_path_T,
                                      db_path=db_path))
    else:
        tids = [templates[t].template_idx for t in range(len(templates))]
    n_templates = len(tids)
    # ----------------------------------------------
    #      Define the projection used to
    #      work in a cartesian space
    # ----------------------------------------------
    data_coords = crs.PlateCarree()
    longitudes = np.float32([templates[i].longitude for i in range(n_templates)])
    latitudes = np.float32([templates[i].latitude for i in range(n_templates)])
    depths = np.float32([templates[i].depth for i in range(n_templates)])
    projection = crs.Mercator(central_longitude=np.mean(longitudes),
                              min_latitude=latitudes.min(),
                              max_latitude=latitudes.max())
    XY = projection.transform_points(data_coords, longitudes, latitudes)
    cartesian_coords = np.stack([XY[:, 0], XY[:, 1], depths], axis=1)
    # compute the directional errors
    dir_errors = np.zeros((n_templates, n_templates), dtype=np.float32)
    for t in range(n_templates):
        unit_direction = cartesian_coords - cartesian_coords[t, :]
        unit_direction /= np.sqrt(np.sum(unit_direction**2, axis=1))[:, np.newaxis]
        # this operation produced NaNs for i=t
        unit_direction[np.isnan(unit_direction)] = 0.
        # compute the length of the covariance ellipsoid
        # in the direction that links the two earthquakes
        dir_errors[t, :] =\
                np.sqrt(np.sum((np.dot(templates[t].cov_mat, unit_direction.T)**2), axis=0))
    # the square root of these lengths are the directional errors in km
    dir_errors = np.sqrt(dir_errors)
    if return_as_pd:
        dir_errors = pd.DataFrame(columns=[tid for tid in tids],
                                  index=[tid for tid in tids],
                                  data=dir_errors)
    return dir_errors

def intertp_distances_(templates=None, tids=None, return_as_pd=True,
                       db_path_T='template_db_2', db_path=cfg.dbpath):
    """
    Parameters
    ------------
    templates: list of dataset.Template, default to None
        If not None, templates is a list of dataset.Template objects
        for which the inter-template distances are computed.
    tids: list or array of integers, default to None
        If not None (in the case where templates=None), use the path
        variables db_path_T and db_path to read a list of dataset.Template
        objects.
    return_as_pd: boolean, default to True
        If True, returns the inter-template distances in a pandas DataFrame.
    db_path_T: string, default to 'template_db_2'
        Folder where template files are stored.
    db_path: string, default to cfg.path
        Root directory where outputs are stored.
    Returns
    ----------
    intertp_dist: (n_templates, n_templates) numpy ndarray
                  or pandas DataFrame
        The distances, in km, between hypocenters of pairs of templates.
    """
    from cartopy import geodesic
    if tids is None and templates is None:
        print('tids or templates should be specified!')
        return
    elif templates is None:
        templates = []
        full_path_tp = os.path.join(db_path, db_path_T)
        print(f'Reading templates from {full_path_tp}')
        for tid in tids:
            templates.append(Template(f'template{tid}', db_path_T,
                                      db_path=db_path))
    else:
        tids = [templates[t].template_idx for t in range(len(templates))]
    n_templates = len(tids)
    longitudes = np.float32([templates[i].longitude for i in range(n_templates)])
    latitudes = np.float32([templates[i].latitude for i in range(n_templates)])
    depths = np.float32([templates[i].depth for i in range(n_templates)])
    G = geodesic.Geodesic()

    intertp_dist = np.zeros((n_templates, n_templates), dtype=np.float64)
    for t, tid in enumerate(tids):
        d = G.inverse(np.array([[longitudes[t], latitudes[t]]]),
                      np.stack((longitudes, latitudes), axis=1))
        d = np.asarray(d)[:, 0].squeeze()/1000.
        intertp_dist[t, :] = np.sqrt(d**2 + (depths[t]-depths)**2)
    if return_as_pd:
        intertp_dist = pd.DataFrame(columns=[tid for tid in tids],
                                    index=[tid for tid in tids],
                                    data=intertp_dist)
    return intertp_dist

