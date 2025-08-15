import os
import sys

from .config import cfg

import h5py as h5
import pandas as pd
import numpy as np
import glob


def load_pykonal_tts(filename, path):
    """Load the travel-time grid computed with Pykonal.

    Load the travel times previously computed with Pykonal and reformat the axes
    to follow NLLoc's convention.

    Parameters
    -----------
    filename: string
        Name of the travel-time file. Example: 'tts.h5'.
    path: string, default to `BPMF.cfg.MOVEOUTS_PATH`
        Name of the directory where the travel-time file is located.

    Returns
    ---------
    longitude: (n_longitude, n_latitude, n_depth) numpy.ndarray
        longitudes of the grid points, in decimals.
    latitude: (n_longitude, n_latitude, n_depth) numpy.ndarray
        latitudes of the grid points, in decimals.
    depth: (n_longitude, n_latitude, n_depth) numpy.ndarray
        depths of the grid points, in km.
    tts: dictionary
        dictionary with one entry per phase, e.g. `tts['p']`. each phase is itself
        made of sub-dictionaries, one for each station: e.g.
        `tts['p']['station1']`. `tts['p']['stationxx']` is an
        (n_longitude, n_latitude, n_depth) numpy.ndarray of travel times.
    """
    path_tts = os.path.join(path, filename)
    # load grid point coordinates
    with h5.File(path_tts, mode="r") as f:
        latitude = f["source_coordinates"]["latitude"][()]
        longitude = f["source_coordinates"]["longitude"][()]
        depth = f["source_coordinates"]["depth"][()]

    # load travel times
    tts = {}
    with h5.File(path_tts, mode="r") as f:
        for phase in ["P", "S"]:
            tts[phase] = {}
            for sta in f[f"tt_{phase}"].keys():
                tts[phase][sta] = f[f"tt_{phase}"][sta][()]

    # initial axis order is (depth, latitude, longitude) with decreasing depths

    # load travel times and re-arrange arrays such that axes are:
    # (longitude, latitude, depth) with increasing indexes corresponding
    # to increasing values

    # 1) reverse depth AND latitude axes
    longitude = longitude[::-1, ::-1, :]
    latitude = latitude[::-1, ::-1, :]
    depth = depth[::-1, ::-1, :]

    for phase in ["P", "S"]:
        for sta in tts[phase].keys():
            tts[phase][sta] = tts[phase][sta][::-1, ::-1, :]

    # 2) swap depth and longitude axes
    longitude = np.swapaxes(longitude, axis1=0, axis2=2)
    latitude = np.swapaxes(latitude, axis1=0, axis2=2)
    depth = np.swapaxes(depth, axis1=0, axis2=2)
    for phase in ["P", "S"]:
        for sta in tts[phase].keys():
            tts[phase][sta] = np.swapaxes(tts[phase][sta], axis1=0, axis2=2)

    return longitude, latitude, depth, tts


def read_NLLoc_outputs(filename, path):
    """Read the NLLoc output hyp file.

    Parameters
    -----------
    filename: string
        Name of the NLLoc output file.
    path: string
        Name of the NLLoc output directory.

    Returns
    ---------
    hypocenter: dictionary
        Dictionary with four fields: `origin_time`, `latitude`, `longitude`,
        `depth`.
    predicted_times: pandas.DataFrame
        `pandas.DataFrame` with the predicted arrival times and the residuals.
    """
    hypocenter = {}
    f = open(os.path.join(path, filename), mode="r")
    # read until relevant line
    for line in f:
        line_s = line.split()
        if line_s[0] == "NLLOC":
            success = line_s[2].strip('"')
            if success == "LOCATED":
                success = True
            else:
                success = False
        elif line_s[0] == "GEOGRAPHIC":
            hypocenter_info = line[:-1].split()
        elif line_s[0] == "QUALITY":
            tt_rms = float(line_s[8])
        elif line_s[0] == "STATISTICS":
            uncertainty_info = line[:-1].split()
        elif line_s[0] == "STAT_GEOG":
            hypocenter["exp_latitude"] = float(line_s[2])
            hypocenter["exp_longitude"] = float(line_s[4])
            hypocenter["exp_depth"] = float(line_s[6])
            break

    hypocenter["success"] = success
    hypocenter["origin_time"] = "{}-{}-{}T{}:{}:{}".format(
        hypocenter_info[2],
        hypocenter_info[3],
        hypocenter_info[4],
        hypocenter_info[5],
        hypocenter_info[6],
        max(0.0, float(hypocenter_info[7])),
    )
    try:
        hypocenter["origin_time"] = pd.Timestamp(hypocenter["origin_time"])
    except:
        print("Unreadable time: ", hypocenter["origin_time"])
        return None, None
    if float(hypocenter_info[7]) < 0.0:
        # it happens that NLLoc returns negative seconds
        hypocenter["origin_time"] -= pd.Timedelta(float(hypocenter_info[7]), unit="s")

    hypocenter["latitude"] = float(hypocenter_info[9])
    hypocenter["longitude"] = float(hypocenter_info[11])
    hypocenter["depth"] = float(hypocenter_info[13])
    hypocenter["tt_rms"] = tt_rms
    # warning! The covariance matrix is expressed
    # in a LEFT HANDED system
    # for a RIGHT HANDED system, we reverse Z-axis
    # which is initially pointing downward
    cov_mat = np.zeros((3, 3), dtype=np.float32)
    cov_mat[0, 0] = float(uncertainty_info[8])  # cov XX
    cov_mat[0, 1] = float(uncertainty_info[10])  # cov XY
    cov_mat[0, 2] = float(uncertainty_info[12])  # cov XZ
    cov_mat[1, 1] = float(uncertainty_info[14])  # cov YY
    cov_mat[1, 2] = float(uncertainty_info[16])  # cov YZ
    cov_mat[2, 2] = float(uncertainty_info[18])  # cov ZZ
    cov_mat[2, :] *= -1.0
    cov_mat[:, 2] *= -1.0
    # symmetrical matrix:
    hypocenter["cov_mat"] = cov_mat + cov_mat.T - np.diag(cov_mat.diagonal())
    # read until relevant line
    for line in f:
        if line[:5] == "PHASE":
            break
    predicted_times = {}
    predicted_times["P_residuals_sec"] = []
    predicted_times["P_tt_sec"] = []
    predicted_times["S_residuals_sec"] = []
    predicted_times["S_tt_sec"] = []
    predicted_times["stations_P"] = []
    predicted_times["stations_S"] = []
    for line in f:
        if line == "END_PHASE\n":
            break
        phase_info = line[:-1].split()
        if phase_info[4] == "P":
            predicted_times["stations_P"].append(phase_info[0])
            predicted_times["P_tt_sec"].append(float(phase_info[15]))
            predicted_times["P_residuals_sec"].append(float(phase_info[16]))
        elif phase_info[4] == "S":
            predicted_times["stations_S"].append(phase_info[0])
            predicted_times["S_tt_sec"].append(float(phase_info[15]))
            predicted_times["S_residuals_sec"].append(float(phase_info[16]))
    f.close()
    test = list(set(predicted_times["stations_P"]) - set(predicted_times["stations_S"]))
    if len(test) > 0:
        print("Unexpected output: Not the same stations for P and S waves.")
        return
    predicted_times["stations"] = predicted_times["stations_P"]
    del predicted_times["stations_P"]
    del predicted_times["stations_S"]
    predicted_times = pd.DataFrame(predicted_times)
    predicted_times.set_index("stations", inplace=True)
    return hypocenter, predicted_times


def write_NLLoc_inputs(
    longitude,
    latitude,
    depth,
    tts,
    net,
    output_path=cfg.NLLOC_INPUT_PATH,
    basename=cfg.NLLOC_BASENAME,
):
    """Write the hdr and buf travel-time files for NLLoc.

    Write the hdr and buf NLLoc files assuming the GLOBAL mode. In this mode,
    coordinates are given in geographic coordinates: longitude, latitude and
    depth. The origin of the grid is taken as the southwest corner. More
    information at: http://alomax.free.fr/nlloc/soft7.00/formats.html#_grid_

    Parameters
    -----------
    longitude: (n_longitude, n_latitude, n_depth) numpy.ndarray
        longitudes of the grid points, in decimals.
    latitude: (n_longitude, n_latitude, n_depth) numpy.ndarray
        latitudes of the grid points, in decimals.
    depth: (n_longitude, n_latitude, n_depth) numpy.ndarray
        depths of the grid points, in km.
    tts: dictionary
        dictionary with one entry per phase, e.g. `tts['p']`. each phase is itself
        made of sub-dictionaries, one for each station: e.g.
        `tts['p']['station1']`. `tts['p']['stationxx']` is an
        (n_longitude, n_latitude, n_depth) numpy.ndarray of travel times.
    net: dataset.Network class
        The `dataset.Network` instance with the station names and coordinates.
    output_path: string, default to 'cfg.NLLOC_INPUT_PATH'
        Path to the directory where NLLoc grid files are stored.
    basename: string, default to 'cfg.NLLOC_BASENAME'
        Basename of all NLLoc input files.
    """
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # --------------------------------------------------
    #   line 1 of the header file is common to all stations
    # --------------------------------------------------
    # get the number of grid points in each direction
    n_lon, n_lat, n_dep = longitude.shape
    # get the lon/lat of the southwestern corner
    lon_ori = longitude.min()
    lat_ori = latitude.min()
    z_ori = depth.min()
    print(f"The origin of the grid is: {lon_ori:.4f}, {lat_ori:.4f}," f" {z_ori:.3f}km")
    # get the grid spacing
    d_lon = longitude[1, 0, 0] - longitude[0, 0, 0]
    d_lat = latitude[0, 1, 0] - latitude[0, 0, 0]
    d_dep = depth[0, 0, 1] - depth[0, 0, 0]
    print(f"Longitude spacing: {d_lon:.3f}deg")
    print(f"Latitude spacing: {d_lat:.3f}deg")
    print(f"Depth spacing: {d_dep:.3f}km")
    # specify grid type
    grid_type = "TIME"
    # define line1
    line1 = (
        f"{n_lon} {n_lat} {n_dep} {lon_ori} {lat_ori} {z_ori} "
        f"{d_lon:.3f} {d_lat:.3f} {d_dep:.3f} {grid_type}\n"
    )
    # --------------------------------------------------
    #    line 2: station-specific line
    # --------------------------------------------------
    for s, sta in enumerate(net.stations):
        print(f"Station {sta}")
        for phase in tts.keys():
            print(f"--- Phase {phase.upper()}")
            filename = f"{basename}.{phase.upper()}.{sta}.time"
            line2 = f"{sta} {net.longitude[s]} {net.latitude[s]} " f"{net.depth[s]}\n"
            line3 = "TRANS GLOBAL\n"
            # write the header file
            with open(os.path.join(output_path, filename + ".hdr"), mode="w") as f:
                f.write(line1)
                f.write(line2)
                f.write(line3)
            # wriute the data binary file
            with open(os.path.join(output_path, filename + ".buf"), mode="w") as f:
                np.float32(tts[phase][sta].flatten()).tofile(f)
    print("Done!")


def write_NLLoc_obs(
    origin_time, picks, stations, filename, path=cfg.NLLOC_INPUT_PATH, err_min=0.04
):
    """Write the .obs file for NLLoc.

    Parameters
    -----------
    origin_time: string or datetime
        Origin, or reference, time of the picks.
    picks: pandas.DataFrame
        Attribute of an `dataset.Event` instance, produced by
        `dataset.Event.pick_PS_phases`.
    stations: List of strings
        List of the station names to use for the relocation.
    filename: string
        Name of the .obs file.
    path: string, default to `cfg.NLLOC_INPUT_PATH`
        Name of the directory where to save the .obs file.
    err_min: scalar float, default to 0.04
        Minimum error, in seconds, on phase picks.
    """
    from obspy import UTCDateTime as udt

    NLLoc = open(os.path.join(path, filename), "a")

    ot = udt(origin_time)

    for st in stations:
        # if st not in stations_to_use:
        #    continue
        if st in picks["P_abs_picks"].dropna().index:
            if "P_unc_sec" in picks.columns:
                err = min(err_min, picks.loc[st, "P_unc_sec"])
            else:
                err = err_min
            P_arrival_time = udt(picks.loc[st]["P_abs_picks"])
            NLLoc.write(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    st,  # station name
                    "?",  # instrument type
                    "?",  # component
                    "?",  # P phase onset (?)
                    "P",  # Phase type
                    "?",  # first motion
                    P_arrival_time.strftime("%Y%m%d"),
                    P_arrival_time.strftime("%H%M"),
                    P_arrival_time.strftime("%S.%f"),
                    "GAU",  # Gaussian errors
                    # max(dt, picks['picks_p'][st][1]), # uncertainty [s]
                    err,  # uncertainty [s]
                    "-1.0",  # coda duration
                    "-1.0",  # amplitude
                    "-1.0",  # period
                    "1",  # prior weight
                )
            )
        else:
            P_arrival_time = ot
            # create a fake pick item that will not be ised
            NLLoc.write(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    st,  # station name
                    "?",  # instrument type
                    "?",  # component
                    "?",  # P phase onset (?)
                    "P",  # Phase type
                    "?",  # first motion
                    P_arrival_time.strftime("%Y%m%d"),
                    P_arrival_time.strftime("%H%M"),
                    P_arrival_time.strftime("%S.%f"),
                    "GAU",  # Gaussian errors
                    0.0,  # uncertainty [s]
                    "-1.0",  # coda duration
                    "-1.0",  # amplitude
                    "-1.0",  # period
                    "0",  # prior weight
                )
            )
        if st in picks["S_abs_picks"].dropna().index:
            if "S_unc_sec" in picks.columns:
                err = min(err_min, picks.loc[st, "S_unc_sec"])
            else:
                err = err_min
            S_arrival_time = udt(picks.loc[st]["S_abs_picks"])
            NLLoc.write(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    st,  # station name
                    "?",  # instrument type
                    "?",  # component
                    "?",  # P phase onset (?)
                    "S",  # Phase type
                    "?",  # first motion
                    S_arrival_time.strftime("%Y%m%d"),
                    S_arrival_time.strftime("%H%M"),
                    S_arrival_time.strftime("%S.%f"),
                    "GAU",  # Gaussian errors
                    err,  # uncertainty [s]
                    "-1.0",  # coda duration
                    "-1.0",  # amplitude
                    "-1.0",  # period
                    "1",  # prior weight
                )
            )
        else:
            S_arrival_time = ot
            # create a fake pick item that will not be ised
            NLLoc.write(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    st,  # station name
                    "?",  # instrument type
                    "?",  # component
                    "?",  # P phase onset (?)
                    "S",  # Phase type
                    "?",  # first motion
                    S_arrival_time.strftime("%Y%m%d"),
                    S_arrival_time.strftime("%H%M"),
                    S_arrival_time.strftime("%S.%f"),
                    "GAU",  # Gaussian errors
                    0.0,  # uncertainty [s]
                    "-1.0",  # coda duration
                    "-1.0",  # amplitude
                    "-1.0",  # period
                    "0",  # prior weight
                )
            )

    NLLoc.write(" \n")
    NLLoc.close()


def write_NLLoc_control(
    ctrl_filename,
    out_filename,
    obs_filename,
    TRANS="GLOBAL",
    NLLoc_input_path=cfg.NLLOC_INPUT_PATH,
    NLLoc_output_path=cfg.NLLOC_OUTPUT_PATH,
    NLLoc_basename=cfg.NLLOC_BASENAME,
    method="EDT_OT_WT_ML",
    angle_grid="ANGLES_NO",
    grid="MISFIT",
    locsearch="OCT",
    phases=["P", "S"],
    excluded_obs={},
    n_depth_points=None,
    **kwargs,
):
    """Write the NLLoc control file.

    All input and output files created here for NLLoc will be deleted.
    Note that all additional key-word arguments to NLLoc, using the same
    parameter names as in NLLoc (http://alomax.free.fr/nlloc/).
    
    Parameters
    ----------
    ctrl_filename : str
        Name of the control file.
    out_filename : str
        Name of NLLoc's output file.
    obs_filename : str
        Name of the input observation file.
    TRANS : str, optional
        Geographic transformation. See NLLoc's documentation.
        Defaults to 'GLOBAL'.
    NLLoc_input_path : str, optional
        Path to NLLoc's input files, that is, travel-time tables.
        Defaults to 'cfg.NLLOC_INPUT_PATH'.
    NLLoc_output_path : str, optional
        Path to NLLoc's output files, that is, the results of location.
        Defaults to 'cfg.NLLOC_OUTPUT_PATH'.
    NLLoc_basename : str, optional
        Basename of the travel-time files. If you have multiple travel-time
        grids at the same `NLLoc_input_path` directory, use this argument to
        use one or the other. Defaults to 'cfg.NLLOC_BASENAME'.
    method : str, optional
        Name of the loss function. See NLLoc's documentation.
        Defaults to 'EDT_OT_WT_ML'.
    angle_grid : str, optional
        Alias for the 'angleMode' parameter of the 'LOCANGLES' command in the
        NLLoc control file. See NLLoc's documentation.
        Defaults to 'ANGLES_NO'.
    grid : str, optional
        Either of 'MISFIT' (default) or 'PROB_DENSITY'. Alias for the
        'gridType' parameter of the 'LOCGRID' command in the NLLoc control
        file. See NLLoc's documentation.
    locsearch : str, optional
        Either of 'GRID', 'MET' or 'OCT' (default). This parameter goes to the
        LOCSEARCH command in the NLLoc control file. It determines how the loss
        function is minimizes:
        - GRID: Grid search (very computationally expensive).
        - MET: Metropolis algorithm (MCMC). Efficient but may be stuck in a
               local minimum.
        - OCT: Oct tree importance sampling algorithm. This is a mix of sampling
               and grid search that allows the efficient search for the global
               minimum. See 'http://alomax.free.fr/nlloc/octtree/OctTree.html'
               for more details. This is the default option.
    phases : list of str, optional
        List of phases used by NonLinLoc. This list includes either "P", "S"
        or both. Defaults to ["P", "S"].
    excluded_obs : dict, optional
        Excluded observations using NLLoc's LOCEXCLUDE command:
            `LOCEXCLUDE sta ph`
        and `excluded_obs[sta] = ph`. Defaults to an empty dictionary.
    n_depth_points : int or None, optional
        If not None, only the first `n_depth_points` points are kept along 
        the depth axis in the grid.

        
    """
    # --------------------------------------------------
    #          generic parameters
    author = kwargs.get("author", "XXXX")
    affiliation = kwargs.get("affiliation", "????????")
    # --------------------------------------------------
    # for OCT
    kwargs.setdefault("initNumCells_x", 10)
    kwargs.setdefault("initNumCells_y", 10)
    kwargs.setdefault("initNumCells_z", 10)
    kwargs.setdefault("minNodeSize", 0.00001)
    kwargs.setdefault("maxNumNodes", 10000)
    kwargs.setdefault("numScatter", 1000)
    kwargs.setdefault("useStationsDensity", 1)
    kwargs.setdefault("stopOnMinNodeSize", 1)
    oct_args = [
        str(kwargs["initNumCells_x"]),
        str(kwargs["initNumCells_y"]),
        str(kwargs["initNumCells_z"]),
        f"{kwargs['minNodeSize']:f}",
        str(kwargs["maxNumNodes"]),
        str(kwargs["numScatter"]),
        str(kwargs["useStationsDensity"]),
        str(kwargs["stopOnMinNodeSize"]),
    ]
    # for GRID
    kwargs.setdefault("numSamplesDraw", 10)
    # for MET
    kwargs.setdefault("numSamples", 1000)
    kwargs.setdefault("numLearn", 1000)
    kwargs.setdefault("numEquil", 1000)
    kwargs.setdefault("numBeginSave", 1000)
    kwargs.setdefault("numSkip", 10)
    kwargs.setdefault("stepInit", -10)
    kwargs.setdefault("stepMin", 0.01)
    kwargs.setdefault("stepFact", 8.0)
    kwargs.setdefault("probMin", 0.1)
    met_args = [
        str(kwargs["numSamples"]),
        str(kwargs["numLearn"]),
        str(kwargs["numEquil"]),
        str(kwargs["numBeginSave"]),
        str(kwargs["numSkip"]),
        str(kwargs["stepInit"]),
        str(kwargs["stepMin"]),
        str(kwargs["stepFact"]),
        str(kwargs["probMin"]),
    ]
    fc = open(os.path.join(NLLoc_input_path, ctrl_filename), "w")
    fc.write("# ---------------------------\n")
    fc.write("#    Generic control file statements    \n")
    fc.write("# ---------------------------\n")
    fc.write("CONTROL  3  54321\n")
    fc.write(f"TRANS  {TRANS}\n")
    fc.write("# ---------------------------\n")
    fc.write("#    NLLoc control file statements    \n")
    fc.write("# ---------------------------\n")
    fc.write(f"LOCSIG  {author}  --  {affiliation}\n")
    in_fn = os.path.join(NLLoc_input_path, obs_filename)
    NLLoc_root = os.path.join(NLLoc_input_path, NLLoc_basename)
    out_fn = os.path.join(NLLoc_output_path, out_filename)
    fc.write(f"LOCFILES  {in_fn}  NLLOC_OBS  {NLLoc_root}  {out_fn}\n")
    # fc.write('LOCHYPOUT  SAVE_NLLOC_ALL  SAVE_HYPOINV_SUM\n')
    fc.write("LOCHYPOUT  SAVE_NLLOC_ALL\n")

    if locsearch == "OCT":
        fc.write("LOCSEARCH OCT " + " ".join(oct_args) + "\n")
    elif locsearch == "GRID":
        fc.write(f"LOCSEARCH GRID {kwargs['numSamplesDraw']}\n")
    elif locsearch == "MET":
        fc.write("LOCSEARCH  MET " + " ".join(met_args) + "\n")
    else:
        print("locsearch should be either of 'OCT', 'GRID' or 'MET'!")
        return
    # read header file to automatically determine grid dimensions
    fn = glob.glob(os.path.join(NLLoc_input_path, f"{NLLoc_basename}*hdr"))[0]
    with open(fn, "r") as fhdr:
        dim = fhdr.readline()
    # --------------------------------------------------------------
    #                  LOCGRID parameters
    locgrid_params = dim.split()[:-1]
    locgrid_params[0] = str(kwargs.get("xNum", locgrid_params[0]))
    locgrid_params[1] = str(kwargs.get("yNum", locgrid_params[1]))
    locgrid_params[2] = str(kwargs.get("zNum", locgrid_params[2]))
    locgrid_params[3] = str(kwargs.get("xOrig", locgrid_params[3]))
    locgrid_params[4] = str(kwargs.get("yOrig", locgrid_params[4]))
    locgrid_params[5] = str(kwargs.get("zOrig", locgrid_params[5]))
    locgrid_params[6] = str(kwargs.get("dx", locgrid_params[6]))
    locgrid_params[7] = str(kwargs.get("dy", locgrid_params[7]))
    locgrid_params[8] = str(kwargs.get("dz", locgrid_params[8]))
    if n_depth_points is not None:
        locgrid_params[2] = str(min(int(locgrid_params[2]), n_depth_points))
    fc.write("LOCGRID  " + "  ".join(locgrid_params) + f"  {grid}  SAVE\n")
    # --------------------------------------------------------------
    #               LOCMETH parameters
    maxDistStaGrid = kwargs.get("maxDistStaGrid", 5000)
    minNumberPhases = kwargs.get("minNumberPhases", 0)
    maxNumberPhases = kwargs.get("maxNumberPhases", -1)
    minNumberSphases = kwargs.get("minNumberSphases", -1)
    VpVsRatio = kwargs.get("VpVsRatio", -1)
    maxNum3DGridMemory = kwargs.get("maxNum3DGridMemory", 6)
    minDistStaGrid = kwargs.get("minDistStaGrid", -1)
    iRejectDuplicateArrivals = kwargs.get("iRejectDuplicateArrivals", 1)
    params = [
        method,
        maxDistStaGrid,
        minNumberPhases,
        maxNumberPhases,
        minNumberSphases,
        VpVsRatio,
        maxNum3DGridMemory,
        minDistStaGrid,
        iRejectDuplicateArrivals,
    ]
    fc.write("LOCMETH " + " ".join([str(p) for p in params]) + "\n")
    # --------------------------------------------------------------
    #              LOCGAU parameters
    SigmaTime = kwargs.get("SigmaTime", 0.02)
    CorrLen = kwargs.get("CorrLen", 5.0)
    fc.write(f"LOCGAU  {SigmaTime}  {CorrLen}\n")
    # --------------------------------------------------------------
    #             LOCGAU2 parameters
    SigmaTfraction = kwargs.get("SigmaTfraction", 0.05)
    SigmaTmin = kwargs.get("SigmaTmin", 0.02)
    SigmaTmax = kwargs.get("SigmaTmax", 10.0)
    fc.write(f"LOCGAU2 {SigmaTfraction} {SigmaTmin} {SigmaTmax}\n")
    # --------------------------------------------------------------
    for ph in phases:
        fc.write(f"LOCPHASEID  {ph.upper()}\n")
    # --------------------------------------------------------------
    #         LOCQUAL2ERR parameters
    #  define 5 levels of quality
    Err0 = kwargs.get("Err0", 0.1)
    Err1 = kwargs.get("Err1", 0.5)
    Err2 = kwargs.get("Err2", 1.0)
    Err3 = kwargs.get("Err3", 2.0)
    Err4 = kwargs.get("Err4", 99999.9)
    params = [Err0, Err1, Err2, Err3, Err4] 
    fc.write("LOCQUAL2ERR " + " ".join([str(p) for p in params]) + "\n")
    # --------------------------------------------------------------
    fc.write(f"LOCANGLES  {angle_grid}  5\n")
    # --------------------------------------------------------------
    #       LOCSTAWT parameters
    cutoffDist = kwargs.get("cutoffDist", 10000000.0)
    useStationsDensity = int(kwargs.setdefault("useStationsDensity", 1))
    fc.write(f"LOCSTAWT {useStationsDensity} {cutoffDist}\n")
    # --------------------------------------------------------------
    for sta, ph in excluded_obs.items():
        fc.write(f"LOCEXCLUDE {sta} {ph}\n")
