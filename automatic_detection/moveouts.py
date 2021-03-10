import numpy as np
from .config import cfg
from obspy.geodetics.base import calc_vincenty_inverse
from os.path import isfile
import h5py as h5

def MV_object(filename, net, \
              relative=True, \
              relativeSP=False, \
              remove_airquakes=False, \
              subset_stations=None):
    """
    MV_object(filename, net, relative=True, relativeSP=False, remove_airquakes=False, subset_stations=None)\n
    net: network object \n
    relative: if True, the MV object is filled with the moveouts relative to the first S-wave arrival \n
    relativeSP: if True, the MV object is filled with the moveouts relative to the first P-wave arrival \n
    remove_airquakes: if True, remove the airquakes (test seismic source with positive elevation) from the grid \n
    subset_stations: if not None, is a boolean array with elements corresponding to the station indexes the user wants to select
    """
    #-------------------------------------
    #--- initialize the moveout object ---
    MV = Moveouts(filename, subset_stations)
    #-------------------------------------
    MV.attach_dist(net)
    if relativeSP == True:
        MV.s_relative_p, MV.s_relative_p_samp = MV.mvS(relative=False, relativeSP=relativeSP)
        MV.p_relative,   MV.p_relative_samp   = MV.mvP(relative=True)
    else:
        MV.s_relative,   MV.s_relative_samp   = MV.mvS(relative=relative)
        MV.p_relative,   MV.p_relative_samp   = MV.mvP(relative=relative)
    if remove_airquakes:
        print("Searching airquakes ...")
        idx_AQ = np.where(MV.depth.flatten() <= 0.)[0]
        idx_EQ = np.ones(MV.depth.size, dtype=np.bool)
        idx_EQ[idx_AQ] = False
        print("Airquakes removed: {:d} sources removed !".format(idx_AQ.size))
        MV.idx_EQ = idx_EQ
    else:
        MV.idx_EQ = None
    return MV

#def get_distances(source_latitudes,
#                  source_longitudes,
#                  source_depths,
#                  receiver_latitudes,
#                  receiver_longitudes,
#                  receiver_depths,
#                  ):
#    n_sources = len(source_latitudes)
#    n_stations = len(receiver_latitudes)
#    distances = np.zeros((n_sources, n_stations), dtype=np.float32)
#    for i in range(n_sources):
#        for s in range(n_stations):
#            dist, _, _ = calc_vincenty_inverse(source_latitudes[i], source_longitudes[i],
#                                               receiver_latitudes[s], receiver_longitudes[s])
#            dist /= 1000. # from km to m
#            dist = np.sqrt(dist**2 + (source_depths[i] - receiver_depths[s])**2)
#            distances[i, s] = dist
#    return distances

def get_distances(source_latitudes,
                  source_longitudes,
                  source_depths,
                  receiver_latitudes,
                  receiver_longitudes,
                  receiver_depths,
                  ):

    from cartopy.geodesic import Geodesic

    # initialize distance array
    distances = np.zeros((len(source_latitudes), len(receiver_latitudes)),
                         dtype=np.float32)

    # initialize the Geodesic instance
    G = Geodesic()
    for s in range(len(receiver_latitudes)):
        epi_distances = G.inverse(np.array([[receiver_longitudes[s], receiver_latitudes[s]]]),
                                  np.hstack((source_longitudes[:, np.newaxis],
                                             source_latitudes[:, np.newaxis]))
                                  )
        distances[:, s] = np.asarray(epi_distances)[:, 0].squeeze()/1000.
        distances[:, s] = np.sqrt(distances[:, s]**2 + (source_depths - receiver_depths[s])**2)
                              
    return distances
    
class Moveouts():
    def __init__(self, filename, subset_stations):
        self.filename  = filename
        with h5.File(cfg.moveouts_path + self.filename + '.h5', mode='r') as f:
            self.latitude  = f['latitude'][()]
            self.longitude = f['longitude'][()]
            self.depth     = f['depth'][()]
            if 'simplified_grid' in f.keys():
                self.simplified_grid = f['simplified_grid'][()]
        self.n_sources = self.longitude.size
        self.subset_stations = subset_stations

    def attach_dist(self, net):
        """
        attach_dist(self, net)\n
        Attach the matrices of source - station distances to the moveout object.
        """
        f = h5.File(cfg.moveouts_path + self.filename + '.h5', mode='r')
        if 'distances' in f.keys():
            distances = f['distances'][()]
        else:
            distances = get_distances(self.latitude,
                                      self.longitude,
                                      self.depth,
                                      net.latitude,
                                      net.longitude,
                                      net.depth)
            f.close()
            with h5.File(cfg.moveouts_path + self.filename + '.h5', mode='a') as f:
                f.create_dataset('distances', data=distances, compression='gzip')
        if self.subset_stations is not None:
            self.distances = distances[:, self.subset_stations]
        else:
            self.distances = distances

    def get_closest_stations(self, data_availability, n_closest_stations):
        """
        get_closest_stations(self, data_availability, n_closest_stations)\n
        """
        n_stations_array = self.distances.shape[-1] # total number of stations in the array
        # map from the reduced array (with only the available stations) to the whole array:
        reference_indexation = np.arange(n_stations_array)[data_availability] 
        closest_stations_indexes = np.zeros((self.n_sources, n_closest_stations), dtype=np.int32)
        for i in range(self.n_sources):
            closest_available_stations = np.argsort(self.distances[i,data_availability])
            closest_stations_indexes[i,:]      = reference_indexation[closest_available_stations][:n_closest_stations]
        self.closest_stations_indexes = closest_stations_indexes

    def mvS(self, relative=True, relativeSP=False):
        """
        mvS(self, relative=True, relativeSP=False)\n
        relative : if True, returns relative moveouts \n
        relativeSP : if True, returns moveouts relative to
                     first P-wave arrival \n
        """
        with h5.File(cfg.moveouts_path + self.filename + '.h5', mode='r') as f:
            mvS = f['s_moveouts'][()]
        mvS_samp = np.int32(mvS * cfg.sampling_rate)
        if self.subset_stations is not None:
            mvS =           mvS[:,self.subset_stations]
            mvS_samp = mvS_samp[:,self.subset_stations]
        #-----------------------------------------------------------
        if relativeSP == True:
            with h5.File(cfg.moveouts_path + self.filename + '.h5', mode='r') as f:
                mvP = f['p_moveouts'][()]
            mvP_samp = np.int32(mvP * cfg.sampling_rate)
            for i in range(self.n_sources):
                mvS[i,:] -= mvP[i,:].min()
                mvS_samp[i,:] -= mvP_samp[i,:].min()
        elif relative:
            for i in range(self.n_sources):
                mvS[i,:] -= mvS[i,:].min()
                mvS_samp[i,:] -= mvS_samp[i,:].min()
        return np.round(mvS, decimals=2), mvS_samp
    
    def mvP(self, relative=True):
        """
        mvP(self, relative=True) \n
        relative : if True, returns relative moveouts \n
        """
        with h5.File(cfg.moveouts_path + self.filename + '.h5', mode='r') as f:
            mvP = f['p_moveouts'][()]
        mvP_samp = np.int32(mvP * cfg.sampling_rate)
        if self.subset_stations is not None:
            mvP =           mvP[:, self.subset_stations]
            mvP_samp = mvP_samp[:, self.subset_stations]
        #---------------------------------------------------------
        if relative:
            for i in range(self.n_sources):
                mvP[i,:] -= mvP[i,:].min()
                mvP_samp[i,:] -= mvP_samp[i,:].min()
        return np.round(mvP, decimals=2), mvP_samp

    def simplify_grid(self, moveouts, threshold):
        from .clib import find_similar_sources
        redundant_sources = find_similar_sources(moveouts, threshold)
        sources_simplified_grid = np.arange(self.n_sources)[~redundant_sources]
        return sources_simplified_grid
