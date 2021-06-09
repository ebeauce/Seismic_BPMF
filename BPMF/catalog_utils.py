import os
import sys

libpath = '/home/ebeauce/libraries'
sys.path.append(libpath)
from .config import cfg
from . import dataset

import numpy as np
import pandas as pd
import cartopy as ctp

from time import time as give_time

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
    if tids is None and templates is None:
        print('tids or templates should be specified!')
        return
    elif templates is None:
        templates = []
        full_path_tp = os.path.join(db_path, db_path_T)
        print(f'Reading templates from {full_path_tp}')
        for tid in tids:
            templates.append(dataset.Template(f'template{tid}', db_path_T,
                                              db_path=db_path))
    else:
        tids = [templates[t].template_idx for t in range(len(templates))]
    n_templates = len(tids)
    # ----------------------------------------------
    #      Define the projection used to
    #      work in a cartesian space
    # ----------------------------------------------
    data_coords = ctp.crs.PlateCarree()
    longitudes = np.float32([templates[i].longitude for i in range(n_templates)])
    latitudes = np.float32([templates[i].latitude for i in range(n_templates)])
    depths = np.float32([templates[i].depth for i in range(n_templates)])
    projection = ctp.crs.Mercator(central_longitude=np.mean(longitudes),
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
            templates.append(dataset.Template(f'template{tid}', db_path_T,
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

def find_multiples(catalog, intertp_distances=None, directional_errors=None,
                   db_path_T='template_db_2', db_path=cfg.dbpath,
                   dt_criterion=3., distance_criterion=5.):
    """
    Parameters
    -----------
    catalog: dictionary
        Dictionary containing at least the following keys:
        origin_times, template_ids, correlation_coefficients.
    intertp_distances: pandas DataFrame, default to None
        If None, use the list of template ids read from catalog
        and the path variables db_path_T and db_path to build an
        (n_templates, n_templates) data frame of the inter-template
        distances.
    directional_errors: pandas DataFrame, default to None
        If None, use the list of template ids read from catalog
        and the path variables db_path_T and db_path to build an
        (n_templates, n_templates) data frame of the directional
        errors, i.e. the length of the uncertainty ellipsoid in
        the inter-template direction.
    db_path_T: string, default to 'template_db_2'
        Folder where template files are stored.
    db_path: string, default to cfg.path
        Root directory where outputs are stored.
    dt_criterion: float, default to 3
        Time, in seconds, below which two events are considered
        for redundancy.
    distance_criterion: float, default to 5
        Distance, in kilometers, below which two events are considered
        for redundancy. This distance criterion is applied to the 
        distance between the two uncertainty ellipsoid, hence the use
        of directional_errors.

    Returns
    --------
    unique_events: boolean (n_events,) array
        Array of booleans of same length as catalog['origin_times'].
        unique_events[n] is True if event n is not already included
        in the catalog due to detection by another template.
    """
    t1 = give_time()
    print('Searching for events detected by multiple templates')
    print('All events occurring within {:.1f}sec and with uncertainty '
          'ellipsoids closer than {:.1f}km will be considered the same'.
          format(dt_criterion, distance_criterion))
    tids = np.sort(np.unique(catalog['template_ids']))
    if intertp_distances is None:
        intertp_distances = interp_distances_(
                tids=tids, db_path_T=db_path_T, db_path=db_path,
                return_as_pd=True)
    if directional_errors is None:
        directional_errors = directional_errors_(
                tids=tids, db_path_T=db_path_T, db_path=db_path,
                return_as_pd=True)
    # combine inter-template distances and directional errors
    # to compute the minimum inter-uncertainty ellipsoid distances
    # this quantity can be negative if the ellipsoids overlap
    ellipsoid_distances = intertp_distances.values\
                        - directional_errors.values\
                        - directional_errors.values.T
    ellipsoid_distances = pd.DataFrame(columns=[tid for tid in tids],
                                       index=[tid for tid in tids],
                                       data=ellipsoid_distances)
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
        # apply the spatial criterion
        # get inter-template distances
        ellips_dist = ellipsoid_distances[tid1].loc[tids_candidates].values
        # include directional errors in criterion
        # i.e. we apply the distance criterion to the minimum distance
        # between the pairs of uncertainty ellipsoids
        multiples = candidates[np.where(ellips_dist < distance_criterion)[0]]
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
    return unique_events

