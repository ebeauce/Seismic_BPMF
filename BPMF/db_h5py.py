import os

from .config import cfg
from . import data, common, dataset

import h5py as h5
import numpy as np
import pandas as pd
from numpy import asarray as ar

from obspy import UTCDateTime as udt
from obspy import Stream, Trace

#=================================================================================
#                FUNCTIONS FOR DETECTIONS
#=================================================================================

def write_detections(filename,
                     detections,
                     db_path=cfg.dbpath):
    """
    write_detection(filename, detections, db_path=cfg.dbpath)\n
    detections is a dictionary
    """
    f_meta = os.path.join(db_path, filename + 'meta.h5')
    with h5.File(f_meta, mode='w') as f:
        for item in detections.keys():
            if item == 'waveforms':
                # the waveforms are written in a separate file
                continue
            f.create_dataset(item, data=detections[item], compression='gzip')
    #-------------------------------------------------------------------
    f_wave = os.path.join(db_path, filename + 'wav.h5')
    with h5.File(f_wave, mode='w') as f:
        f.create_dataset('waveforms', data=detections['waveforms'], compression='gzip')

def read_detections(filename,
                    attach_waveforms=True,
                    db_path=cfg.dbpath):
    """
    read_detections(filename, attach_waveforms=True, db_path=cfg.dbpath)\n
    """
    detections = {}
    with h5.File(os.path.join(db_path, filename + 'meta.h5'), mode='r') as f:
        for item in f.keys():
            detections.update({item : f[item][()]})
    if attach_waveforms:
        with h5.File(os.path.join(db_path, filename + 'wav.h5'), mode='r') as f:
            detections.update({'waveforms' : f['waveforms'][()]})
    detections['stations']   = detections['stations'].astype('U')
    detections['components'] = detections['components'].astype('U')
    return detections

def read_template_database(filename,
                           db_path=cfg.dbpath):
    """
    read_template_database(filename, db_path=cfg.template_db_path')\n
    """
    template_db = {}
    with h5.File(os.path.join(db_path, filename + '.h5'), mode='r') as f:
        for item in list(f.keys()):
            template_db[item] = f[item][()]
    return template_db

#=================================================================================
#                       FUNCTIONS FOR TEMPLATES
#=================================================================================

def initialize_template(metadata):
    T = Stream()
    for st in metadata['stations']:
        for ch in metadata['channels']:
            T += Trace(data=np.zeros(np.int32(metadata['duration']*metadata['sampling_rate']), dtype=np.float32))
            T[-1].stats.station = st
            T[-1].stats.channel = ch
            T[-1].stats.sampling_rate = metadata['sampling_rate']
    T.metadata = metadata
    return T

def write_template(filename,
                   template_metadata,
                   waveforms,
                   db_path=cfg.dbpath,
                   db_path_T='template_db_1/'):
    # recently changed template to template_metadata
    # therefore, it is now required to pass template.metadata
    # instead of template
    f_meta = os.path.join(db_path, db_path_T, filename+'meta.h5')
    f_wave = os.path.join(db_path, db_path_T, filename+'wav.h5')
    with h5.File(f_meta, 'w') as fm:
        for key in list(template_metadata.keys()):
            if type(template_metadata[key]) == np.ndarray:
                if isinstance(template_metadata[key][0], str):
                    template_metadata[key] = template_metadata[key].astype('S')
                fm.create_dataset(key, data = template_metadata[key], compression='gzip')
            else:
                fm.create_dataset(key, data = template_metadata[key])
    with h5.File(f_wave, 'w') as fw:
        fw.create_dataset('waveforms', data = waveforms, compression='gzip')


def read_template(filename,
                  db_path=cfg.dbpath,
                  db_path_T='template_db_1',
                  attach_waveforms=False):
    f_meta = os.path.join(db_path, db_path_T, filename+'meta.h5')
    f_wave = os.path.join(db_path, db_path_T, filename+'wav.h5')
    with h5.File(f_meta, mode='r') as fm:
        metadata = {}
        for key in fm.keys():
            metadata.update({key:fm[key][()]})
    with h5.File(f_wave, mode='r') as fw:
        waveforms = fw['waveforms'][()]
    metadata['stations'] = metadata['stations'].astype('U')
    metadata['channels'] = metadata['channels'].astype('U')
    T = initialize_template(metadata)
    for s,st in enumerate(metadata['stations']):
        for c,ch in enumerate(metadata['channels']):
            T.select(station=st, channel=ch)[0].data = waveforms[s,c,:]
    if attach_waveforms:
        T.waveforms = waveforms
    return T

def read_template_new_version(filename,
                              db_path=cfg.dbpath,
                              db_path_T='template_db_1',
                              attach_waveforms=False):

    T = dataset.Template(filename,
                         db_path_T,
                         db_path=db_path,
                         attach_waveforms=attach_waveforms)

    return T

def read_template_list(database_index,
                       db_path=cfg.dbpath,
                       db_path_T='template_db_1',
                       attach_waveforms=True,
                       well_relocated_templates=False,
                       mining_activity=True,
                       new_version=False):
    with h5.File(os.path.join(db_path, db_path_T, database_index + '.h5'), mode='r') as f:
        if well_relocated_templates:
            template_indexes = np.intersect1d(f['well_relocated_template_indexes'][()], f['template_indexes'][()])
        else:
            template_indexes = f['template_indexes'][()]
        if not mining_activity:
            template_indexes = np.setdiff1d(template_indexes, f['mining_activity'][()])
    templates = []
    for tid in template_indexes:
        if new_version:
            templates.append(read_template_new_version('template{:d}'.format(tid),
                                           db_path=db_path,
                                           db_path_T=db_path_T,
                                           attach_waveforms=attach_waveforms))
        else:
            templates.append(read_template('template{:d}'.format(tid),
                                           db_path=db_path,
                                           db_path_T=db_path_T,
                                           attach_waveforms=attach_waveforms))
    return templates

def read_template_list_parallel(database_index,
                                db_path=cfg.dbpath,
                                db_path_T='template_db_1',
                                attach_waveforms=True,
                                well_relocated_templates=False,
                                mining_activity=True,
                                new_version=False):
    import itertools
    import concurrent.futures

    with h5.File(os.path.join(db_path, db_path_T, database_index + '.h5'), mode='r') as f:
        if well_relocated_templates:
            template_indexes = np.intersect1d(f['well_relocated_template_indexes'][()], f['template_indexes'][()])
        else:
            template_indexes = f['template_indexes'][()]
        if not mining_activity:
            template_indexes = np.setdiff1d(template_indexes, f['mining_activity'][()])
    if new_version:
        data_loader = lambda tid, db_path_T, db_path, attach_waveforms:\
                 read_template_new_version('template{:d}'.format(tid),
                                           db_path=db_path,
                                           db_path_T=db_path_T,
                                           attach_waveforms=attach_waveforms)
    else:
        data_loader = lambda tid, db_path_T, db_path, attach_waveforms:\
                 read_template('template{:d}'.format(tid),
                               db_path=db_path,
                               db_path_T=db_path_T,
                               attach_waveforms=attach_waveforms)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(data_loader, tid, db_path_T, db_path, attach_waveforms)\
                   for tid in iter(template_indexes)]
        templates = [fut.result() for fut in futures]
    return templates


def get_template_ids_list(database_index,
                          db_path=cfg.dbpath,
                          db_path_T='template_db_1',
                          well_relocated_templates=False,
                          remove_mining=False):
    with h5.File(os.path.join(db_path, db_path_T, database_index + '.h5'), mode='r') as f:
        if well_relocated_templates:
            template_indexes = np.intersect1d(f['well_relocated_template_indexes'][()], f['template_indexes'][()])
        else:
            template_indexes = f['template_indexes'][()]
        if remove_mining and 'mining_activity' in f.keys():
            template_indexes = np.setdiff1d(template_indexes, f['mining_activity'][()])
    return template_indexes

def select_template_ids(db_path_T,
                        fields=[],
                        vmin_vmax=[[]],
                        db_path=cfg.dbpath,
                        file_format='csv',
                        remove_mining=False,
                        stable_loc_only=False):

    if file_format == 'csv':
        meta_db = pd.read_csv(os.path.join(db_path, db_path_T,
                              'metadata_db.csv'), sep='\t')
        tids = meta_db['tid'].values
        # initialize mask
        mask = np.ones(len(tids), dtype=np.bool)
        for i, field in enumerate(fields):
            print('Select template ids with {} greater than {:.1f} '\
                  'and lower than {:.1f}'.format(field, vmin_vmax[i][0], vmin_vmax[i][1]))
            mask = (meta_db[field].values >= vmin_vmax[i][0])\
                 & (meta_db[field].values <= vmin_vmax[i][1])\
                 & mask
        if stable_loc_only and ('stable_location' in meta_db.keys()):
            mask = meta_db['stable_location'].values & mask
        tids = tids[mask]
    elif file_format == 'hdf5':
        with h5.File(os.path.join(
            db_path, db_path_T, 'metadata_db.h5'), 'r') as f:
            tids = f['tid'][()]
            # initialize mask
            mask = np.ones(len(tids), dtype=np.bool)
            for i, field in enumerate(fields):
                print('Select template ids with {} greater than {:.1f} '\
                      'and lower than {:.1f}'.format(field, vmin_vmax[i][0], vmin_vmax[i][1]))
                mask = (f[field][()] >= vmin_vmax[i][0])\
                     & (f[field][()] <= vmin_vmax[i][1])\
                     & mask
        if stable_loc_only and ('stable_location' in f.keys()):
            mask = f['stable_location'][()] & mask
            tids = tids[mask]
    else:
        print('file_format must be csv or hdf5! (exit without doing anything)')
        return
    if remove_mining:
        with h5.File(os.path.join(
            db_path, db_path_T, 'database_index.h5'), 'r') as f:
            mining = f['mining_activity'][()]
        tids = np.setdiff1d(tids, mining)
    print('{:d} templates were selected'.format(len(tids)))
    return tids

def write_template_metadata(database_index,
                            db_path_T,
                            db_path_M=None,
                            db_path=cfg.dbpath,
                            file_format='csv'):
    """
    Parameters:
    ------------
    database_index: string,
        Name of the hdf5 file containing the list of template ids.
    db_path_T: string,
        Name of the folder containing the template files.
    db_path_M: string, default to None.
        If not None, use it to get the number of detections per template.
    db_path: string, default to cfg.dbpath
        Name of the root folder containing the result files.
    file_format: string, default to 'csv'
        Must be either 'csv' or 'hdf5'.
        If 'csv', then pandas is used to write the csv file.
        If 'hdf5', then h5py is used to write the hdf5 file.
    """

    tids = get_template_ids_list(database_index,
                                 db_path=db_path,
                                 db_path_T=db_path_T)

    fields = ['latitude', 'longitude', 'depth',
              #'lon_unc', 'lat_unc', 'dep_unc', 'max_unc',\
              'hmax_unc', 'vmax_unc', 'max_unc',
              'tid']
    metadata = {}
    for field in fields:
        metadata[field] = []
    if db_path_M is not None:
        n_detections = []
    for tid in tids:
        T = dataset.Template(f'template{tid}', db_path_T, db_path=db_path)
        T.hor_ver_uncertainties()
        metadata['tid'].append(T.tid)
        metadata['latitude'].append(T.latitude)
        metadata['longitude'].append(T.longitude)
        metadata['depth'].append(T.depth)
        metadata['hmax_unc'].append(T.hmax_unc)
        metadata['vmax_unc'].append(T.vmax_unc)
        metadata['max_unc'].append(T.max_location_uncertainty)
        #with h5.File(os.path.join(db_path, db_path_T,
        #             'template{:d}meta.h5'.format(tid)), 'r') as f:
        #    metadata['tid'].append(tid)
        #    metadata['latitude'].append(f['latitude'][()])
        #    metadata['longitude'].append(f['longitude'][()])
        #    metadata['depth'].append(f['depth'][()])
        #    metadata['lon_unc'].append(np.sqrt(f['cov_mat'][0, 0]))
        #    metadata['lat_unc'].append(np.sqrt(f['cov_mat'][1, 1]))
        #    metadata['dep_unc'].append(np.sqrt(f['cov_mat'][2, 2]))
        #    metadata['max_unc'].append(f['max_location_uncertainty'][()])
        if db_path_M is not None:
            ot = read_catalog_multiplets('multiplets{:d}'.format(tid),
                                         db_path_M=db_path_M, db_path=db_path,
                                         object_from_cat='origin_times')
            n_detections.append(len(ot))
    if db_path_M is not None:
        metadata['n_detections'] = np.int32(n_detections)
    if file_format == 'csv':
        meta_db = pd.DataFrame.from_dict(metadata)
        meta_db.to_csv(os.path.join(db_path, db_path_T, 'metadata_db.csv'), sep='\t')
    elif file_format == 'hdf5':
        with h5.File(os.path.join(db_path,
                                  db_path_T,
                                  'metadata_db.h5'), 'w') as f:
            for field in fields:
                f.create_dataset(field, data=np.asarray(metadata[field]))
    else:
        print('file_format must be csv or hdf5! (exit without doing anything)')
        

def epicentral_distances_all(db_path_T,
                             database_index='database_index',
                             db_path=cfg.dbpath,
                             remove_mining=False):
    
    from cartopy.geodesic import Geodesic

    tids = get_template_ids_list(database_index,
                                 db_path=db_path,
                                 db_path_T=db_path_T,
                                 remove_mining=remove_mining)

    with h5.File(os.path.join(db_path,
                              db_path_T,
                              'metadata_db.h5'), 'r') as f:
        indexes = np.in1d(f['tid'][()], tids)
        longitude = f['longitude'][indexes]
        latitude = f['latitude'][indexes]
        depth = f['depth'][indexes]

    G = Geodesic()

    intertp_dist = np.zeros((len(tids), len(tids)), dtype=np.float64)
    for t, tid in enumerate(tids):
        d = G.inverse(np.array([[longitude[t], latitude[t]]]),
                      np.hstack((longitude.reshape(-1, 1),
                                 latitude.reshape(-1, 1))))
        d = np.asarray(d)[:, 0]
        intertp_dist[t, :] = d.squeeze()

    # return distance in km
    return intertp_dist/1000.

def hypocentral_distances_all(db_path_T,
                              tids=None,
                              database_index='database_index',
                              db_path=cfg.dbpath,
                              remove_mining=False):
    
    from cartopy.geodesic import Geodesic

    if tids is None:
        tids = get_template_ids_list(database_index,
                                     db_path=db_path,
                                     db_path_T=db_path_T,
                                     remove_mining=remove_mining)

    with h5.File(os.path.join(db_path,
                              db_path_T,
                              'metadata_db.h5'), 'r') as f:
        indexes = np.in1d(f['tid'][()], tids)
        longitude = f['longitude'][indexes]
        latitude = f['latitude'][indexes]
        depth = f['depth'][indexes]

    G = Geodesic()

    intertp_dist = np.zeros((len(tids), len(tids)), dtype=np.float64)
    for t, tid in enumerate(tids):
        d = G.inverse(np.array([[longitude[t], latitude[t]]]),
                      np.hstack((longitude.reshape(-1, 1),
                                 latitude.reshape(-1, 1))))
        d = np.asarray(d)[:, 0].squeeze()/1000.
        intertp_dist[t, :] = np.sqrt(d**2 + (depth[t]-depth)**2)

    return intertp_dist

def epicentral_distances(tid,
                         db_path_T,
                         database_index='database_index',
                         db_path=cfg.dbpath,
                         remove_mining=False):
    
    from cartopy.geodesic import Geodesic

    tids = get_template_ids_list(database_index,
                                 db_path=db_path,
                                 db_path_T=db_path_T,
                                 remove_mining=remove_mining)
    if tid not in tids:
        print('{:d} is not in the list of template ids!'.format(tid))
        return
    tid_index = np.where(tids == tid)[0][0]

    with h5.File(os.path.join(db_path,
                              db_path_T,
                              'metadata_db.h5'), 'r') as f:
        indexes = np.in1d(f['tid'][()], tids)
        longitude = f['longitude'][indexes]
        latitude = f['latitude'][indexes]
        depth = f['depth'][indexes]

    G = Geodesic()

    d = G.inverse(np.array([[longitude[tid_index], latitude[tid_index]]]),
                  np.hstack((longitude.reshape(-1, 1),
                             latitude.reshape(-1, 1))))
    intertp_dist = np.asarray(d)[:, 0].squeeze()

    # return distance in km
    return intertp_dist/1000.

def hypocentral_distances(tid,
                          db_path_T,
                          database_index='database_index',
                          db_path=cfg.dbpath,
                          remove_mining=False):
    
    from cartopy.geodesic import Geodesic

    tids = get_template_ids_list(database_index,
                                 db_path=db_path,
                                 db_path_T=db_path_T,
                                 remove_mining=remove_mining)
    if tid not in tids:
        print('{:d} is not in the list of template ids!'.format(tid))
        return
    tid_index = np.where(tids == tid)[0][0]

    with h5.File(os.path.join(db_path,
                              db_path_T,
                              'metadata_db.h5'), 'r') as f:
        indexes = np.in1d(f['tid'][()], tids)
        longitude = f['longitude'][indexes]
        latitude = f['latitude'][indexes]
        depth = f['depth'][indexes]

    G = Geodesic()

    d = G.inverse(np.array([[longitude[tid_index], latitude[tid_index]]]),
                  np.hstack((longitude.reshape(-1, 1),
                             latitude.reshape(-1, 1))))
    d = np.asarray(d)[:, 0].squeeze()/1000.
    intertp_dist = np.sqrt(d**2 + (depth[tid_index]-depth)**2)

    return intertp_dist

def select_tids_within_R(tid,
                         R,
                         db_path_T,
                         database_index='database_index',
                         db_path=cfg.dbpath,
                         remove_mining=False):

    tids = get_template_ids_list(database_index,
                                 db_path=db_path,
                                 db_path_T=db_path_T,
                                 remove_mining=remove_mining)

    distances = hypocentral_distances(tid,
                                      db_path_T,
                                      database_index=database_index,
                                      db_path=cfg.dbpath,
                                      remove_mining=remove_mining)

    subset_tids = tids[distances < R]

    return subset_tids

def select_for_cross_section(tids, lon1, lat1, lon2, lat2,
                             max_orthogonal_dist=20., max_parallel_dist=5., 
                             return_metadata=False,
                             db_path_T='template_db_2', db_path=cfg.dbpath):
    d2r = np.pi/180.
    r2d = 1./d2r
    R_earth = 6371. # km
    # take (lon1, lat1) as the origin of our reference frame
    # The distance spanned by 1 degree in longitude at lat1 is:
    alpha = np.pi - lat1*d2r
    r = R_earth*np.sin(alpha)
    # distance per degree in longitude (and not per radian!)
    dist_per_lon = r*d2r 
    # distance per degree in latitude
    dist_per_lat = R_earth*d2r
    # build the vector that is colinear to the section
    # (lon1, lat1) to (lon2, lat2)
    S = np.array([(lon2-lon1)*dist_per_lon, (lat2-lat1)*dist_per_lat])
    S_length = np.sqrt(np.sum(S**2))
    # make it a unit vector
    S /= S_length
    S_orth = np.array([-S[1], S[0]])
    # get the templates metadata
    md_file = os.path.join(db_path, db_path_T,
                           'metadata_db.csv')
    if not os.path.isfile(md_file):
        print('{} does not exit. You need to write'
              ' the csv metadata file before using this function'.
              format(md_file))
        return
    meta_db = pd.read_csv(md_file, sep='\t', index_col='tid')
    meta_db = meta_db.loc[tids]
    # determine the position vectors in this reference frame
    X = np.zeros((len(tids), 2), dtype=np.float64)
    X[:, 0] = (meta_db['longitude'].values - lon1)*dist_per_lon
    X[:, 1] = (meta_db['latitude'].values - lat1)*dist_per_lat
    # add these to the metadata frame
    meta_db['CS_parallel'] = np.sum(X*S[np.newaxis, :], axis=-1)
    meta_db['CS_orthogonal'] = np.sum(X*S_orth[np.newaxis, :], axis=-1)
    # compute the orthogonal distance to the section, i.e. scalar product
    # |y_2| = sqrt(||y||**2 - y_1**2)
    distance_to_section = np.abs(meta_db['CS_orthogonal'].values)
    # apply the distance threshold
    selection = (distance_to_section <= max_orthogonal_dist)\
               & (meta_db['CS_parallel'] >= -max_parallel_dist)\
               & (meta_db['CS_parallel'] - S_length <= max_parallel_dist)
    CS_geometry = {}
    CS_geometry['parallel'] = S
    CS_geometry['orthogonal'] = S_orth
    dist_to_degrees = np.array([1./dist_per_lon, 1./dist_per_lat])
    corner_1 = np.array([lon1, lat1]) + max_orthogonal_dist*S_orth*dist_to_degrees\
               - max_parallel_dist*S*dist_to_degrees
    corner_2 = np.array([lon2, lat2]) + max_orthogonal_dist*S_orth*dist_to_degrees\
               + max_parallel_dist*S*dist_to_degrees
    corner_3 = np.array([lon2, lat2]) - max_orthogonal_dist*S_orth*dist_to_degrees\
               + max_parallel_dist*S*dist_to_degrees
    corner_4 = np.array([lon1, lat1]) - max_orthogonal_dist*S_orth*dist_to_degrees\
               - max_parallel_dist*S*dist_to_degrees
    CS_geometry['corners'] = np.vstack((corner_1, corner_2, corner_3, corner_4))
    # CS_geometry inherit from CS attributes
    CS_geometry['lon_1'] = lon1
    CS_geometry['lat_1'] = lat1
    CS_geometry['lon_2'] = lon2
    CS_geometry['lat_2'] = lat2
    if return_metadata:
        return tids[selection], CS_geometry, meta_db[selection]
    else:
        return tids[selection], CS_geometry

#=================================================================================
#                       FUNCTIONS FOR MULTIPLETS
#=================================================================================

def write_multiplets(filename,
                     metadata,
                     waveforms,
                     db_path=cfg.dbpath):
    filename_meta = os.path.join(db_path, filename + 'meta.h5')
    filename_wave = os.path.join(db_path, filename + 'wav.h5')
    with h5.File(filename_meta, mode='w') as f:
        for item in metadata.keys():
            f.create_dataset(item, data=metadata[item], compression='gzip')
    with h5.File(filename_wave, mode='w') as f:
        f.create_dataset('waveforms', data=waveforms, compression='lzf')

def write_multiplets_bulk(filename,
                          metadata,
                          waveforms,
                          db_path=cfg.dbpath,
                          force_update=False):
    filename_meta = os.path.join(db_path, filename + 'meta.h5')
    filename_wave = os.path.join(db_path, filename + 'wav.h5')
    n_templates = len(metadata)
    with h5.File(filename_meta, mode='a') as f:
        for t in range(n_templates):
            if len(metadata[t]['origin_times']) == 0:
                # no detection
                continue
            group_name = str(metadata[t]['template_id'][0])
            if force_update and group_name in f.keys():
                # was already processed, but force overwritting
                del f[group_name]
                f.create_group(group_name)
            elif group_name in f.keys():
                # was already processed, and do not update
                continue
            else:
                # write it for the first time
                f.create_group(group_name)
            for key in metadata[t].keys():
                f[group_name].create_dataset(key, data=metadata[t][key], compression='gzip')
    with h5.File(filename_wave, mode='a') as f:
        for t in range(n_templates):
            if len(metadata[t]['origin_times']) == 0:
                # no detection
                continue
            group_name = str(metadata[t]['template_id'][0])
            if force_update and group_name in f.keys():
                # was already processed, but force overwritting
                del f[group_name]
                f.create_group(group_name)
            elif group_name in f.keys():
                # was already processed, and do not update
                continue
            else:
                # write it for the first time
                f.create_group(group_name)
            f[group_name].create_dataset('waveforms', data=waveforms[t]['waveforms'], compression='lzf')
            print('{:d} multiplets added for Template {:d}'.format(waveforms[t]['waveforms'].shape[0], metadata[t]['template_id'][0]))

def read_multiplet(filename,
                   idx,
                   tid,
                   return_tp=False,
                   db_path=cfg.dbpath,
                   db_path_T='template_db_1',
                   db_path_M='matched_filter_1'):
    """
    read_multiplet(filename, idx, db_path=cfg.dbpath) \n
    """
    S = Stream()
    f_meta = os.path.join(db_path, db_path_M, filename+'meta.h5')
    fm = h5.File(f_meta, 'r')
    T = read_template('template{:d}'.format(tid),
                      db_path=db_path,
                      db_path_T=db_path_T)
    f_wave = os.path.join(db_path, db_path_M, filename+'wav.h5')
    fw = h5.File(f_wave, 'r')
    waveforms = fw[str(tid)]['waveforms'][idx,:,:,:]
    fw.close()
    #---------------------------------
    stations   = fm[str(tid)]['stations'][:].astype('U')
    components = fm[str(tid)]['components'][:].astype('U')
    ns         = len(stations)
    nc         = len(components)
    #---------------------------------
    date = udt(fm[str(tid)]['origin_times'][idx])
    for s in range(ns):
        for c in range(nc):
            S += Trace(data = waveforms[s,c,:])
            S[-1].stats['station'] = stations[s]
            S[-1].stats['channel'] = components[c]
            S[-1].stats['sampling_rate'] = cfg.sampling_rate
            S[-1].stats.starttime = date
    S.s_moveouts  = T.metadata['s_moveouts']
    S.p_moveouts  = T.metadata['p_moveouts']
    #S.source_idx  = T.metadata['source_idx']
    S.template_ID = T.metadata['template_idx']
    S.latitude    = T.metadata['latitude']
    S.longitude   = T.metadata['longitude']
    S.depth       = T.metadata['depth']
    S.corr = fm[str(tid)]['correlation_coefficients'][idx]
    S.stations   = stations.tolist()
    S.components = components.tolist()
    fm.close()
    if return_tp:
        return S, T
    else:
        return S

def read_multiplet_new_version(filename,
                               idx,
                               tid,
                               return_tp=False,
                               db_path=cfg.dbpath,
                               db_path_T='template_db_1',
                               db_path_M='matched_filter_1'):
    """
    """
    if type(filename) == type(b''):
        filename = filename.decode('utf-8')
    S = Stream()
    f_meta = os.path.join(db_path, db_path_M, filename+'meta.h5')
    fm = h5.File(f_meta, 'r')
    T = dataset.Template('template{:d}'.format(tid),
                         db_path_T,
                         db_path=db_path,
                         attach_waveforms=True)
    f_wave = os.path.join(db_path, db_path_M, filename+'wav.h5')
    fw = h5.File(f_wave, 'r')
    waveforms = fw[str(tid)]['waveforms'][idx,:,:,:]
    fw.close()
    #---------------------------------
    stations   = fm[str(tid)]['stations'][:].astype('U')
    components = fm[str(tid)]['components'][:].astype('U')
    ns         = len(stations)
    nc         = len(components)
    #---------------------------------
    date = udt(fm[str(tid)]['origin_times'][idx])
    for s in range(ns):
        for c in range(nc):
            S += Trace(data = waveforms[s,c,:])
            S[-1].stats['station'] = stations[s]
            S[-1].stats['channel'] = components[c]
            S[-1].stats['sampling_rate'] = cfg.sampling_rate
            S[-1].stats.starttime = date
    S.s_moveouts  = T.s_moveouts
    S.p_moveouts  = T.p_moveouts
    S.template_ID = T.template_idx
    S.latitude    = T.latitude
    S.longitude   = T.longitude
    S.depth       = T.depth
    S.corr = fm[str(tid)]['correlation_coefficients'][idx]
    S.stations   = stations.tolist()
    S.components = components.tolist()
    fm.close()
    if return_tp:
        return S, T
    else:
        return S

def write_catalog_multiplets(filename,
                             catalog,
                             db_path=cfg.dbpath,
                             db_path_M='matched_filter_1'):
    """
    write_meta_multiplets(filename, metadata, categories, db_path=cfg.dbpath) \n
    """
    fmeta = os.path.join(db_path, db_path_M, filename+'catalog.h5')
    with h5.File(fmeta, mode='w') as fm:
        for category in list(catalog.keys()):
            fm.create_dataset(category, data=catalog[category])

def read_catalog_multiplets(filename,
                            object_from_cat='',
                            db_path=cfg.dbpath,
                            db_path_M='matched_filter_1'):
    """
    read_catalog_multiplets(filename, db_path=cfg.dbpath, db_path_M='matched_filter_1')
    """
    fmeta = os.path.join(db_path, db_path_M, filename+'catalog.h5')
    catalog = {}
    with h5.File(fmeta, mode='r') as fm:
        if object_from_cat == '':
            for item in fm.keys():
                catalog[item] = fm[item][()]
        else:
            catalog = fm[object_from_cat][()]
    return catalog


# =================================================================
#                  STACK
# =================================================================

def write_stack(stack,
                filename='stack',
                db_path_S='stack_db_1',
                db_path=cfg.dbpath):

    full_filename = os.path.join(db_path, db_path_S,
                                 '{}{:d}'.format(filename, stack.template_idx))
    with h5.File(full_filename+'meta.h5', mode='w') as f:
        f.create_dataset('template_idx', data=stack.template_idx)
        f.create_dataset('stations', data=np.asarray(stack.stations).astype('S'))
        f.create_dataset('components', data=np.asarray(stack.components).astype('S'))
        f.create_dataset('sampling_rate', data=stack.sampling_rate)
    with h5.File(full_filename+'wav.h5', mode='w') as f:
        f.create_dataset('waveforms', data=stack.waveforms)
