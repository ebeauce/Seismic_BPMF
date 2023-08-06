import os
import sys
import glob
import numpy as np
import h5py as h5
import pandas as pd
import pathlib

import obspy as obs
from obspy.core import UTCDateTime as udt
from time import time as give_time

from .config import cfg
from . import dataset


# -------------------------------------------------
#              Filtering routines
# -------------------------------------------------


def bandpass_filter(
    X,
    filter_order=4,
    freqmin=cfg.MIN_FREQ_HZ,
    freqmax=cfg.MAX_FREQ_HZ,
    f_Nyq=cfg.SAMPLING_RATE_HZ / 2.0,
    taper_alpha=0.01,
    zerophase=True,
):
    """
    Parameters
    -----------
    X: (n x m) numpy array,
        Numpy array of n observations of m samples each.
        Use X.reshape(1, -1) if you want to process a
        single observation.
    filter_order: integer scalar, default to 4,
        Order/number of corners of the bandpass filter.
    freqmin: scalar float, default to cfg.MIN_FREQ_HZ,
        Low frequency cutoff.
    freqmax: scalar float, default to cfg.MAX_FREQ_HZ,
        High frequency cutoff.
    f_Nyq: scalar float, default to cfg.SAMPLING_RATE_HZ/2,
        Nyquist frequency of the data. By definition,
        the Nyquist frequency is half the sampling rate.
    taper_alpha: scalar float, default to 0.01,
        Defines the sharpness of the edges of the taper
        function. We use the tukey window function. The
        default value of 0.01 produces sharp windows.

    Returns
    --------
    filtered_X: (n x m) numpy array,
        The input array X after filtering.
    """

    from scipy.signal import iirfilter, tukey

    # from scipy.signal import lfilter
    from scipy.signal import zpk2sos, sosfilt
    from scipy.signal import detrend

    # detrend the data
    X = detrend(X, type="constant", axis=-1)
    X = detrend(X, type="linear", axis=-1)

    # design the taper function
    taper = np.repeat(tukey(X.shape[1], alpha=taper_alpha), X.shape[0])
    taper = taper.reshape(X.shape[0], X.shape[1], order="F")

    # design the filter
    # filter_num, filter_den = iirfilter(filter_order,
    #                                   [freqmin/f_Nyq, freqmax/f_Nyq],
    #                                   btype='bandpass',
    #                                   ftype='butter',
    #                                   output='ba')
    # filtered_X = lfilter(filter_num,
    #                     filter_den,
    #                     X*taper)
    ## apply the filter a second time to have a
    ## zero phase filter
    # filtered_X = lfilter(filter_num,
    #                     filter_den,
    #                     X[::-1])[::-1]
    z, p, k = iirfilter(
        filter_order,
        [freqmin / f_Nyq, freqmax / f_Nyq],
        btype="bandpass",
        ftype="butter",
        output="zpk",
    )
    sos = zpk2sos(z, p, k)
    filtered_X = sosfilt(sos, X * taper)
    if zerophase:
        filtered_X = sosfilt(sos, filtered_X[:, ::-1])[:, ::-1]
    return filtered_X


def lowpass_chebyshev_I(
    X, freqmax, sampling_rate, order=8, max_ripple=5, zerophase=False
):
    from scipy.signal import cheby1, sosfilt

    nyquist = sampling_rate / 2.0

    sos = cheby1(
        order,
        max_ripple,
        freqmax / nyquist,
        analog=False,
        btype="lowpass",
        output="sos",
    )

    X = sosfilt(sos, X)
    if zerophase:
        X = sosfilt(sos, X[::-1])[::-1]
    return X


def lowpass_chebyshev_II(
    X, freqmax, sampling_rate, order=10, min_attenuation_dB=40.0, zerophase=False
):
    from scipy.signal import cheby2, sosfilt

    nyquist = sampling_rate / 2.0

    sos = cheby2(
        order,
        min_attenuation_dB,
        # freqmax/nyquist,
        freqmax,
        analog=False,
        fs=sampling_rate,
        btype="lowpass",
        output="sos",
    )

    X = sosfilt(sos, X)
    if zerophase:
        X = sosfilt(sos, X[::-1])[::-1]
    return X


def preprocess_stream(
    stream,
    freqmin=None,
    freqmax=None,
    target_SR=None,
    remove_response=False,
    remove_sensitivity=False,
    plot_resp=False,
    target_duration=None,
    target_starttime=None,
    target_endtime=None,
    minimum_length=0.75,
    minimum_chunk_duration=600.0,
    verbose=True,
    SR_decimals=1,
    decimation_method="simple",
    unit="VEL",
    n_threads=1,
    **kwargs,
):
    """
    Preprocesses a stream of seismic data.

    Parameters
    ----------
    stream : obspy.Stream
        A stream of seismic data.
    freqmin : float, optional
        The minimum frequency to bandpass filter the data.
    freqmax : float, optional
        The maximum frequency to bandpass filter the data.
    target_SR : float, optional
        The target sampling rate of the data.
    remove_response : bool, optional
        Whether to remove instrument response from the data.
    remove_sensitivity : bool, optional
        Whether to remove instrument sensitivity from the data.
    plot_resp : bool, optional
        Whether to plot the instrument response of the data.
    target_duration : float, optional
        The target duration of the data.
    target_starttime : obspy.UTCDateTime, optional
        The start time of the target data.
    target_endtime : obspy.UTCDateTime, optional
        The end time of the target data.
    minimum_length : float, optional
        The minimum length of the data as a fraction of the target duration.
    minimum_chunk_duration : float, optional
        The minimum duration of each data chunk.
    verbose : bool, optional
        Whether to print verbose output during processing.
    SR_decimals : int, optional
        The number of decimals to round sampling rate values to.
    decimation_method : str, optional
        The method used for decimation.
    unit : str, optional
        The unit of the data.
    n_threads : int, optional
        The number of threads over which preprocesing is parallelized.
    **kwargs : dict
        Other keyword arguments to pass to the function.

    Returns
    -------
    obspy.Stream
        A preprocessed stream of seismic data.
    """
    from functools import partial

    data_preprocessor = partial(
        _preprocess_stream,
        freqmin=freqmin,
        freqmax=freqmax,
        target_SR=target_SR,
        remove_response=remove_response,
        remove_sensitivity=remove_sensitivity,
        plot_resp=plot_resp,
        target_duration=target_duration,
        target_starttime=target_starttime,
        target_endtime=target_endtime,
        minimum_length=minimum_length,
        minimum_chunk_duration=minimum_chunk_duration,
        verbose=verbose,
        SR_decimals=SR_decimals,
        decimation_method=decimation_method,
        unit=unit,
        **kwargs,
    )
    if n_threads != 1:
        from concurrent.futures import ProcessPoolExecutor

        #for tr in stream:
        #    tr.stats.sampling_rate = np.round(
        #            tr.stats.sampling_rate, decimals=SR_decimals
        #            )
        stream, _ = _premerge(stream, verbose=verbose)

        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            # we need to group traces from same channels, therefore,
            # we use merge to fill gaps with masked arrays
            stream.merge()
            preprocessed_stream = list(executor.map(data_preprocessor, stream))
            try:
                preprocessed_stream = [
                    tr[0] for tr in preprocessed_stream if len(tr) > 0
                ]
            except Exception as e:
                print(e)
                print(preprocessed_stream)
                for i, tr in enumerate(preprocessed_stream):
                    print(i, tr, len(tr))
                sys.exit(0)
        return obs.Stream(preprocessed_stream)
    else:
        return data_preprocessor(stream)

def _premerge(stream, verbose=False):
    """Clean-up stream before calling merge.
    """
    # first, make a list of all stations in stream
    stations = []
    for tr in stream:
        stations.append(tr.stats.station)
    stations = list(set(stations))
    # second, make a list of all channel types for each station
    for station in stations:
        st = stream.select(station=station)
        channels = []
        for tr in st:
            channels.append(tr.stats.channel[:-1])
        channels = list(set(channels))
        # third, for each channel type, make a list of all sampling
        # rates and detect anomalies if there are more than one single
        # sampling rate
        sampling_rates = []
        for cha in channels:
            st_cha = st.select(channel=f"{cha}*")
            for tr in st_cha:
                sampling_rates.append(tr.stats.sampling_rate)
        unique_sampling_rates, sampling_rates_counts = np.unique(
            sampling_rates, return_counts=True
        )
        # if more than one sampling rate, remove the traces with the least
        # represented sampling rate
        if len(unique_sampling_rates) > 1:
            if sampling_rates_counts[unique_sampling_rates.argmax()] >= 3:
                ref_sampling_rate = unique_sampling_rates.max()
            else:
                ref_sampling_rate = unique_sampling_rates[sampling_rates_counts.argmax()]
            for tr in stream:
                if tr.stats.sampling_rate != ref_sampling_rate:
                    if verbose:
                        print(f"Removing {tr.id} because not desired sampling rate "
                              f"({tr.stats.sampling_rate} vs {ref_sampling_rate})"  )
                    stream.remove(tr)
    return stream, stations


def _preprocess_stream(
    stream,
    freqmin=None,
    freqmax=None,
    target_SR=None,
    remove_response=False,
    remove_sensitivity=False,
    plot_resp=False,
    target_duration=None,
    target_starttime=None,
    target_endtime=None,
    minimum_length=0.75,
    minimum_chunk_duration=600.0,
    verbose=True,
    SR_decimals=1,
    decimation_method="simple",
    unit="VEL",
    **kwargs,
):
    """
    See `preprocess_stream`.
    """
    if isinstance(stream, obs.Trace):
        # user gave a single trace instead of a stream
        stream = obs.Stream(stream)
    preprocessed_stream = obs.Stream()
    if len(stream) == 0:
        if verbose:
            print("Input data is empty!")
        return preprocessed_stream
    if (target_duration is None) and (
        (target_starttime is not None) and target_endtime is not None
    ):
        target_duration = target_endtime - target_starttime
    # find errors in sampling rate metadata
    # first, round sampling rates that may be misrepresented in floating point numbers
    for tr in stream:
        tr.stats.sampling_rate = np.round(tr.stats.sampling_rate, decimals=SR_decimals)
        # and remove the short chunks now to gain time
        # (only works if the traces do not come with masked arrays)
        t1 = udt(tr.stats.starttime.timestamp)
        t2 = udt(tr.stats.endtime.timestamp)
        if t2 - t1 < minimum_chunk_duration:
            # don't include this chunk
            stream.remove(tr)
    if len(stream) == 0:
        if verbose:
            print("Removed all traces because they were too short.")
        return preprocessed_stream
    stream, stations = _premerge(stream, verbose=verbose)
    ## second, make a list of all stations in stream
    #stations = []
    #for tr in stream:
    #    stations.append(tr.stats.station)
    #stations = list(set(stations))
    ## third, make a list of all channel types for each station
    #for station in stations:
    #    st = stream.select(station=station)
    #    channels = []
    #    for tr in st:
    #        channels.append(tr.stats.channel[:-1])
    #    channels = list(set(channels))
    #    # fourth, for each channel type, make a list of all sampling
    #    # rates and detect anomalies if there are more than one single
    #    # sampling rate
    #    sampling_rates = []
    #    for cha in channels:
    #        st_cha = st.select(channel=f"{cha}*")
    #        for tr in st_cha:
    #            sampling_rates.append(tr.stats.sampling_rate)
    #    unique_sampling_rates, sampling_rates_counts = np.unique(
    #        sampling_rates, return_counts=True
    #    )
    #    # if more than one sampling rate, remove the traces with the least
    #    # represented sampling rate
    #    if len(unique_sampling_rates) > 1:
    #        if sampling_rates_counts[unique_sampling_rates.argmax()] >= 3:
    #            ref_sampling_rate = unique_sampling_rates.max()
    #        else:
    #            ref_sampling_rate = unique_sampling_rates[sampling_rates_counts.argmax()]
    #        for tr in stream:
    #            if tr.stats.sampling_rate != ref_sampling_rate:
    #                if verbose:
    #                    print(f"Removing {tr.id} because not desired sampling rate "
    #                          f"({tr.stats.sampling_rate} vs {ref_sampling_rate})"  )
    #                stream.remove(tr)
    # start by cleaning the gaps if there are any
    # start with a simple merge to unite data from same channels into unique
    # trace but without losing information on gaps
    stream.merge()
    for tr in stream:
        trace_id = tr.id
        if np.isnan(tr.data.max()):
            if verbose:
                print(f"Problem with {tr.id} (detected NaNs)!")
            continue
        T1 = udt(tr.stats.starttime.timestamp)
        T2 = udt(tr.stats.endtime.timestamp)
        trace_duration = T2 - T1
        if trace_duration < minimum_length * target_duration:
            # don't include this trace
            if verbose:
                print(f"Too much gap duration on {trace_id}.")
            continue
        # split will lose information about start and end times
        # if the start or the end is masked
        tr = tr.split()
        for chunk in tr:
            t1 = udt(chunk.stats.starttime.timestamp)
            t2 = udt(chunk.stats.endtime.timestamp)
            if t2 - t1 < minimum_chunk_duration:
                # don't include this chunk
                tr.remove(chunk)
        if len(tr) == 0:
            # all chunks were too short
            if verbose:
                print(f"All chunks within {trace_id} were too short.")
            continue
        # measure gap duration
        gap_duration = target_duration - trace_duration
        for gap in tr.get_gaps():
            gap_duration += gap[6]
        if (target_duration is not None) and (
            gap_duration > minimum_length * target_duration
        ):
            if verbose:
                print(f"Too much gap duration on {trace_id}.")
            continue
        tr.detrend("constant")
        tr.detrend("linear")
        tr.taper(0.05, type="cosine")
        # it's now safe to fill the gaps with zeros
        tr.merge(fill_value=0.0)[0]
        tr.trim(starttime=T1, endtime=T2, pad=True, fill_value=0.0)
        preprocessed_stream += tr
    # if the trace came as separated segments without masked
    # elements, it is necessary to merge the stream
    preprocessed_stream = preprocessed_stream.merge(fill_value=0.0)

    # delete the original data to save memory
    del stream
    # resample if necessary:
    for tr in preprocessed_stream:
        if target_SR is None:
            continue
        sr_ratio = tr.stats.sampling_rate / target_SR
        if sr_ratio > 1:
            tr.data = lowpass_chebyshev_II(
                tr.data,
                0.49 * target_SR,
                tr.stats.sampling_rate,
                order=10,
                min_attenuation_dB=40.0,
                zerophase=True,
            )
            if (
                np.round(sr_ratio, decimals=0) == sr_ratio
            ) and decimation_method == "simple":
                # tr's sampling rate is an integer
                # multiple of target_SR
                # do not re-filter the data
                tr.decimate(int(sr_ratio), no_filter=True)
            elif decimation_method == "fourier":
                tr.resample(target_SR, no_filter=True)
            else:
                tr.resample(target_SR, no_filter=True)
        elif sr_ratio < 1:
            if verbose:
                print("Sampling rate is too high on {tr.id}.")
                print(tr)
            preprocessed_stream.remove(tr)
            continue
        else:
            pass
    # remove response if requested
    if remove_response:
        for tr in preprocessed_stream:
            if not hasattr(tr.stats, "response"):
                print(f"Could not find the instrument response for {tr.id}.")
                continue
            T_max = tr.stats.npts * tr.stats.delta
            T_min = tr.stats.delta
            f_min = 1.0 / T_max
            f_max = 1.0 / (2.0 * T_min)
            pre_filt = [f_min, 3.0 * f_min, 0.90 * f_max, 0.97 * f_max]
            tr.remove_response(pre_filt=pre_filt, output=unit, plot=plot_resp)
    elif remove_sensitivity:
        for tr in preprocessed_stream:
            if not hasattr(tr.stats, "response"):
                print(f"Could not find the instrument response for {tr.id}.")
                continue
            tr.remove_sensitivity()
    # filter
    preprocessed_stream.detrend("constant")
    preprocessed_stream.detrend("linear")
    preprocessed_stream.taper(0.02, type="cosine")
    if freqmin is None and freqmax is None:
        # no filtering
        pass
    elif freqmin is None:
        # lowpass filtering
        preprocessed_stream.filter("lowpass", freq=freqmax, zerophase=True)
    elif freqmax is None:
        # highpass filtering
        preprocessed_stream.filter("highpass", freq=freqmin, zerophase=True)
    else:
        # bandpass filtering
        preprocessed_stream.filter(
            "bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True
        )
    # adjust duration
    if target_starttime is not None:
        preprocessed_stream.trim(starttime=target_starttime, pad=True, fill_value=0.0)
    if target_endtime is not None:
        preprocessed_stream.trim(endtime=target_endtime, pad=True, fill_value=0.0)
    if target_duration is not None:
        for i in range(len(preprocessed_stream)):
            n_samples = sec_to_samp(
                target_duration, sr=preprocessed_stream[i].stats.sampling_rate
            )
            preprocessed_stream[i].data = preprocessed_stream[i].data[:n_samples]
    return preprocessed_stream


# -------------------------------------------------
#       Loading travel-time data
# -------------------------------------------------


def get_moveout_array(tts, stations, phases):
    """Format the travel times into a numpy.ndarray.

    Parameters
    -----------
    tts: dictionary
        Output of `load_travel_times`.
    stations: list of strings
        List of station names. Determine the order in which travel times are
        written in the output array.
    phases: list of strings
        List of seismic phases. Determine the order in which travel times are
        writtem in the output array.

    Returns
    ---------
    moveout_arr: (n_sources, n_stations, n_phases) numpy.ndarray
        Numpy array of travel times, in the order specified by `stations` and
        `phases`. At this stage, moveout_arr should still be in units of
        seconds.
    """
    n_stations = len(stations)
    n_phases = len(phases)
    moveout_arr = np.array([tts[ph][sta] for sta in stations for ph in phases]).T
    return moveout_arr.reshape(-1, n_stations, n_phases)


def load_travel_times(
    path, phases, source_indexes=None, return_coords=False, stations=None
):
    """Load the travel times from `path`.

    Parameters
    ------------
    path: string
        Path to the file with the travel times.
    phases: list of strings
        List of the seismic phases to load.
    source_indexes: (n_sources,) int numpy.ndarray, default to None
        If not None, this is used to select a subset of sources from the grid.
    return_coords: boolean, default to False
        If True, also return the source coordinates.
    stations: list of strings, default to None
        If not None, only read the travel times for stations in `stations`.

    Returns
    ---------
    tts: dictionary
        Dictionary with one field per phase. `tts[phase_n]` is itself
        a dictionary with one field per station.
    source_coords: dictionary, optional
        Returned only if `return_coords` is True. This is a dictionary with
        three (n_sources,) numpy.ndarray: `source_coords['latitude']`,
        `source_coords['longitude']`, `source_coords['depth']`.
    """
    tts = pd.DataFrame(columns=phases, index=stations)
    with h5.File(path, mode="r") as f:
        grid_shape = f["source_coordinates"]["depth"].shape
        for ph in phases:
            for sta in f[f"tt_{ph}"].keys():
                if stations is not None and sta not in stations:
                    continue
                # flatten the lon/lat/dep grid as we work with
                # flat source indexes
                if source_indexes is not None:
                    # select a subset of the source grid
                    source_indexes_unravelled = np.unravel_index(
                        source_indexes, grid_shape
                    )
                    selection = np.zeros(grid_shape, dtype=bool)
                    selection[source_indexes_unravelled] = True
                    tts.loc[sta, ph] = f[f"tt_{ph}"][sta][selection].flatten()
                else:
                    tts.loc[sta, ph] = f[f"tt_{ph}"][sta][()].flatten()
        if return_coords:
            if source_indexes is not None:
                source_indexes_unravelled = np.unravel_index(source_indexes, grid_shape)
                selection = np.zeros(grid_shape, dtype=bool)
                selection[source_indexes_unravelled] = True
                source_coords = pd.DataFrame(
                    columns=["longitude", "latitude", "depth"], index=source_indexes
                )
                for coord in f["source_coordinates"].keys():
                    source_coords.loc[source_indexes, coord] = f["source_coordinates"][
                        coord
                    ][selection].flatten()
            else:
                source_coords = pd.DataFrame(
                    columns=["longitude", "latitude", "depth"],
                    index=np.arange(np.prod(grid_shape)),
                )
                for coord in f["source_coordinates"].keys():
                    source_coords[coord] = f["source_coordinates"][coord][()].flatten()
    if return_coords:
        return tts, source_coords
    else:
        return tts


# -------------------------------------------------
#             Stacking routines
# -------------------------------------------------


def SVDWF(
    matrix,
    expl_var=0.4,
    max_singular_values=5,
    freqmin=cfg.MIN_FREQ_HZ,
    freqmax=cfg.MAX_FREQ_HZ,
    sampling_rate=cfg.SAMPLING_RATE_HZ,
    wiener_filter_colsize=None,
):
    """
    Implementation of the Singular Value Decomposition Wiener Filter (SVDWF)
    described in Moreau et al 2017.

    Parameters
    ----------
    matrix: (n x m) numpy array
        n is the number of events, m is the number of time samples
        per event.
    n_singular_values: scalar float
        Number of singular values to retain in the
        SVD decomposition of matrix.
    max_freq: scalar float, default to cfg.MAX_FREQ_HZ
        The maximum frequency of the data, or maximum target
        frequency, is used to determined the size in the
        time axis of the Wiener filter.

    Returns
    --------
    filtered_data: (n x m) numpy array
        The matrix filtered through the SVD procedure.
    """
    from scipy.linalg import svd
    from scipy.signal import wiener
    import matplotlib.pyplot as plt

    try:
        U, S, Vt = svd(matrix, full_matrices=False)
    except Exception as e:
        print(e)
        print("Problem while computing the svd...!")
        return np.random.normal(loc=0.0, scale=1.0, size=matrix.shape)
    if wiener_filter_colsize is None:
        wiener_filter_colsize = U.shape[0]
    # wiener_filter = [wiener_filter_colsize, int(cfg.SAMPLING_RATE_HZ/max_freq)]
    wiener_filter = [wiener_filter_colsize, 1]
    filtered_data = np.zeros((U.shape[0], Vt.shape[1]), dtype=np.float32)
    # select the number of singular values
    # in order to explain 100xn_singular_values%
    # of the variance of the matrix
    var = np.cumsum(S**2)
    if var[-1] == 0.0:
        # only zeros in matrix
        return filtered_data
    var /= var[-1]
    n_singular_values = np.min(np.where(var >= expl_var)[0]) + 1
    n_singular_values = min(max_singular_values, n_singular_values)
    for n in range(min(U.shape[0], n_singular_values)):
        s_n = np.zeros(S.size, dtype=np.float32)
        s_n[n] = S[n]
        projection_n = np.dot(U, np.dot(np.diag(s_n), Vt))
        if wiener_filter[0] == 1 and wiener_filter[1] == 1:
            # no wiener filtering
            filtered_projection = projection_n
        else:
            # the following application of Wiener filtering is questionable: because each projection in this loop is a projection
            # onto a vector space with one dimension, all the waveforms are colinear: they just differ by an amplitude factor (but same shape).
            filtered_projection = wiener(
                projection_n,
                # mysize=[max(2, int(U.shape[0]/10)), int(cfg.SAMPLING_RATE_HZ/freqmax)]
                mysize=wiener_filter,
            )
        # filtered_projection = projection_n
        if np.isnan(filtered_projection.max()):
            continue
        filtered_data += filtered_projection
    if wiener_filter[0] == 1 and wiener_filter[1] == 1:
        # no wiener filtering
        pass
    else:
        filtered_data = wiener(filtered_data, mysize=wiener_filter)
    # remove nans or infs
    filtered_data[np.isnan(filtered_data)] = 0.0
    filtered_data[np.isinf(filtered_data)] = 0.0
    # SVD adds noise in the low and the high frequencies
    # refiltering the SVD-filtered data seems necessary
    filtered_data = bandpass_filter(
        filtered_data,
        filter_order=4,
        freqmin=freqmin,
        freqmax=freqmax,
        f_Nyq=sampling_rate / 2.0,
    )
    return filtered_data


def fetch_detection_waveforms(
    tid,
    db_path_T,
    db_path_M,
    db_path=cfg.INPUT_PATH,
    best_CC=False,
    max_n_events=0,
    norm_rms=True,
    ordering="correlation_coefficients",
    flip_order=True,
    selection=None,
    return_event_ids=False,
    unique_events=False,
    catalog=None,
):
    from itertools import groupby
    from operator import itemgetter

    if catalog is None:
        cat = dataset.Catalog(f"multiplets{tid}catalog.h5", db_path_M)
        cat.read_data()
    else:
        cat = catalog
    CC = np.sort(cat.correlation_coefficients.copy())
    if max_n_events > 0:
        max_n_events = min(max_n_events, len(CC))
        CC_thres = CC[-max_n_events]
    elif best_CC:
        if len(CC) > 300:
            CC_thres = CC[-100]
        elif len(CC) > 70:
            CC_thres = CC[int(7.0 / 10.0 * len(CC))]  # the best 30%
        elif len(CC) > 30:
            CC_thres = np.median(CC)  # the best 50%
        elif len(CC) > 10:
            CC_thres = np.percentile(CC, 33.0)  # the best 66% detections
        else:
            CC_thres = 0.0
    else:
        CC_thres = -1.0
    if selection is None:
        selection = cat.correlation_coefficients >= CC_thres
        if unique_events:
            selection = selection & cat.unique_events
    if (np.sum(selection) == 0) and return_event_ids:
        return np.empty(0), np.empty(0), np.empty(0)
    elif np.sum(selection) == 0:
        return np.empty(0), np.empty(0)
    else:
        pass
    filenames = cat.filenames[selection].astype("U")
    indices = cat.indices[selection]
    CCs = cat.correlation_coefficients[selection]
    event_ids = np.arange(len(cat.origin_times), dtype=np.int32)[selection]
    detection_waveforms = []
    t1 = give_time()
    for filename, rows in groupby(zip(filenames, indices), itemgetter(0)):
        full_filename = os.path.join(db_path, db_path_M, filename + "wav.h5")
        with h5.File(full_filename, mode="r") as f:
            for row in rows:
                idx = row[1]
                detection_waveforms.append(f[str(tid)]["waveforms"][idx, ...])
    detection_waveforms = np.stack(detection_waveforms, axis=0)
    if norm_rms:
        norm = np.std(detection_waveforms, axis=(2, 3))[..., np.newaxis, np.newaxis]
        norm[norm == 0.0] = 1.0
        detection_waveforms /= norm
    n_detections = detection_waveforms.shape[0]
    t2 = give_time()
    print("{:.2f} s to retrieve the waveforms.".format(t2 - t1))
    if ordering is not None:
        # use the requested attribute to order the detections
        if not hasattr(cat, ordering):
            print(
                f"The catalog does not have the {ordering} attribute, "
                "return by chronological order."
            )
        else:
            order = np.argsort(getattr(cat, ordering)[selection])
            if flip_order:
                order = order[::-1]
            detection_waveforms = detection_waveforms[order, ...]
            CCs = CCs[order]
            event_ids = event_ids[order]
    if return_event_ids:
        return detection_waveforms, CCs, event_ids
    else:
        return detection_waveforms, CCs


def fetch_detection_waveforms_refilter(
    tid,
    db_path_T,
    db_path_M,
    net,
    db_path=cfg.INPUT_PATH,
    best_CC=False,
    max_n_events=0,
    norm_rms=True,
    freqmin=0.5,
    freqmax=12.0,
    target_SR=50.0,
    integrate=False,
    t0="detection_time",
    ordering="correlation_coefficients",
    flip_order=True,
    **preprocess_kwargs,
):
    # sys.path.append(os.path.join(cfg.base, 'earthquake_location_eb'))
    # import relocation_utils
    from . import event_extraction

    cat = dataset.Catalog(f"multiplets{tid}catalog.h5", db_path_M)
    cat.read_data()
    CC = np.sort(cat.correlation_coefficients.copy())

    T = dataset.Template(f"template{tid}", db_path_T)
    if t0 == "detection_time":
        correction_time = T.reference_absolute_time - cfg.BUFFER_EXTRACTED_EVENTS_SEC
    elif t0 == "origin_time":
        correction_time = 0.0
    else:
        print("t0 should either be detection_time or origin_time")
    # ------------------------------
    CC = np.sort(CC)
    if max_n_events > 0:
        max_n_events = min(max_n_events, len(CC))
        CC_thres = CC[-max_n_events]
    elif best_CC:
        if len(CC) > 300:
            CC_thres = CC[-100]
        elif len(CC) > 70:
            CC_thres = CC[int(7.0 / 10.0 * len(CC))]  # the best 30%
        elif len(CC) > 30:
            CC_thres = np.median(CC)  # the best 50%
        elif len(CC) > 10:
            CC_thres = np.percentile(CC, 33.0)  # the best 66% detections
        else:
            CC_thres = 0.0
    else:
        CC_thres = -1.0
    selection = cat.correlation_coefficients >= CC_thres
    CCs = cat.correlation_coefficients[selection]
    OTs = cat.origin_times[selection]
    detection_waveforms = []
    t1 = give_time()
    for ot in OTs:
        # the OT in the h5 files correspond to the
        # beginning of the windows that were extracted
        # during the matched-filter search
        print("Extracting event from {}".format(udt(ot)))
        event = event_extraction.extract_event_parallel(
            ot + correction_time,
            net,
            duration=cfg.multiplet_len,
            offset_start=0.0,
            folder="raw",
            attach_response=preprocess_kwargs.get("attach_response", False),
        )
        if integrate:
            event.integrate()
        filtered_ev = event_extraction.preprocess_event(
            event,
            freqmin=freqmin,
            freqmax=freqmax,
            target_SR=target_SR,
            target_duration=cfg.multiplet_len,
            **preprocess_kwargs,
        )
        if len(filtered_ev) > 0:
            detection_waveforms.append(get_np_array(filtered_ev, net, verbose=False))
        else:
            detection_waveforms.append(
                np.zeros(
                    (
                        len(net.stations),
                        len(net.components),
                        sec_to_samp(cfg.multiplet_len, sr=target_SR),
                    ),
                    dtype=np.float32,
                )
            )
    detection_waveforms = np.stack(detection_waveforms, axis=0)
    if norm_rms:
        # one normalization factor for each 3-comp seismogram
        norm = np.std(detection_waveforms, axis=(2, 3))[..., np.newaxis, np.newaxis]
        norm[norm == 0.0] = 1.0
        detection_waveforms /= norm
    n_detections = detection_waveforms.shape[0]
    t2 = give_time()
    print("{:.2f} s to retrieve the waveforms.".format(t2 - t1))
    if ordering is not None:
        # use the requested attribute to order the detections
        if not hasattr(cat, ordering):
            print(
                f"The catalog does not have the {ordering} attribute, "
                "return by chronological order."
            )
        else:
            order = np.argsort(getattr(cat, ordering)[selection])
            if flip_order:
                order = order[::-1]
            detection_waveforms = detection_waveforms[order, ...]
            CCs = CCs[order]
    return detection_waveforms, CCs


def SVDWF_multiplets(
    tid,
    db_path=cfg.INPUT_PATH,
    db_path_M="matched_filter_1",
    db_path_T="template_db_1",
    best=False,
    norm_rms=True,
    max_singular_values=5,
    expl_var=0.4,
    freqmin=cfg.MIN_FREQ_HZ,
    freqmax=cfg.MAX_FREQ_HZ,
    sampling_rate=cfg.SAMPLING_RATE_HZ,
    wiener_filter_colsize=None,
    attach_raw_data=False,
    detection_waveforms=None,
):
    """
    Parameters
    -----------
    tid: scalar integer,
        Template id.
    db_path: string, default to cfg.INPUT_PATH
        Root path of the database.
    db_path_M: string, default to 'matched_filter_1'
        Name of the folder where matched-filtering results
        are stored.
    db_path_T: string, default to 'template_db_1'
        Name of the folder where template files are stored.
    best: boolean, default to False
        If True, only use the detections with higher
        correlation coefficients.
    norm_rms: boolean, default to True
        If True, individual event are RMS-normalized
        before stacking.
    max_singular_values: scalar integer, default to 5
        Disregarding how many singular values are needed
        to reconstruct 100xexp_var% of the variance of
        the detection matrix, the number of singular values
        will not be larger than max_singular_values.
    max_freq: scalar float, default to cfg.MAX_FREQ_HZ
        The maximum frequency of the data, or maximum target
        frequency, is used to determined the size in the
        time axis of the Wiener filter.
    wiener_filter_colsize: scalar integer, default to None,
        Size of the wiener filter in the vertical direction
        (i.e. in the observation axis). If set to None,
        then it will be equal to the number of rows.
    attach_raw_data: boolean, default to False.
        If True, the data extracted during the matched-filter
        search are returned.

    Returns
    --------
    S: obspy Stream,
        Return the SVDWF-processed detections traces in the
        format of an obspy Stream. Stacked traces can be
        found in the obspy Traces, and the filtered (and raw,
        if attach_raw_data is True) data matrix is returned as
        an attribute. See also all the useful metadata in
        the attributes.
    """

    # -----------------------------------------------------------------------------------------------
    T = dataset.Template("template{:d}".format(tid), db_path_T, db_path=db_path)
    # -----------------------------------------------------------------------------------------------
    files_all = glob.glob(os.path.join(db_path, db_path_M, "*multiplets_*meta.h5"))
    files = []
    # ------------------------------
    stack = dataset.Stack(
        T.network_stations, T.channels, sampling_rate=sampling_rate, tid=tid
    )
    n_stations = len(stack.stations)
    n_components = len(stack.components)
    stack.latitude = T.latitude
    stack.longitude = T.longitude
    stack.depth = T.depth
    # ------------------------------
    if detection_waveforms is None:
        detection_waveforms, CCs = fetch_detection_waveforms(
            tid, db_path_T, db_path_M, best_CC=best, norm_rms=norm_rms, db_path=db_path
        )
    else:
        # provided by the user
        pass
    print("{:d} events.".format(detection_waveforms.shape[0]))
    filtered_data = np.zeros_like(detection_waveforms)
    for s in range(n_stations):
        for c in range(n_components):
            filtered_data[:, s, c, :] = SVDWF(
                detection_waveforms[:, s, c, :],
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
                        stack.stations[s], s, stack.components[c], c
                    )
                )
    stacked_waveforms = np.mean(filtered_data, axis=0)
    norm = np.max(stacked_waveforms, axis=-1)[..., np.newaxis]
    norm[norm == 0.0] = 1.0
    stacked_waveforms /= norm
    stack.add_data(stacked_waveforms)
    stack.data = filtered_data
    if attach_raw_data:
        stack.raw_data = detection_waveforms
    stack.n_detections = detection_waveforms.shape[0]
    try:
        stack.correlation_coefficients = CCs
    except:
        stack.correlation_coefficients = "N/A"
    return stack


# ------------------------------------
#      hierarchical clustering
# ------------------------------------


def extract_colors_from_tree(dendogram, labels, color_singleton):
    """
    Routine to build the map from cluster ids to dendogram colors.

    Parameters
    ----------
    dendogram: dendogram from scipy.hierarchy.dendogram.
    labels: (n_samples) numpy array,
        Labels, or cluster ids, returned by
        scipy.hierarchy.fcluster. For each of the initial
        n_samples samples, this function returns its
        cluster membership. labels[i] = k, means that
        data point i belongs to cluster k.
    color_singleton: string,
        Color given to the singleton when calling
        scipy.hierarchy.dendogram.

    Returns
    ---------
    cluster_colors: dictionary,
        Map between cluster id and plotting color.
        cluster_colors[k] = 'C0' means that cluster k
        is colored in 'C0'.
    """
    from itertools import groupby

    # --------------------
    list_summary = []
    for color, group in groupby(dendogram["color_list"]):
        if color == color_singleton:
            continue
        elements = []
        for el in group:
            elements.append(el)
        list_summary.append([color, len(elements)])
    leaf_colors = {}
    leaf_count = 0
    cluster_count = 0
    while True:
        if leaf_count == len(dendogram["leaves"]):
            break
        ev_clusterid = labels[dendogram["leaves"][leaf_count]]
        ev_cluster_size = np.sum(labels == ev_clusterid)
        if ev_cluster_size == 1:
            # next leaf is of color "color_singleton"
            leaf_colors[dendogram["leaves"][leaf_count]] = color_singleton
            leaf_count += 1
            continue
        color = list_summary[cluster_count][0]
        n_branches = list_summary[cluster_count][1]
        for j in range(leaf_count, leaf_count + n_branches + 1):
            leaf_colors[dendogram["leaves"][j]] = color
            leaf_count += 1
        cluster_count += 1
    # leaf_colors should match what is plotted on the dendogram
    # we are mostly interested in the color of each cluster
    cluster_colors = {}
    for i in range(len(leaf_colors)):
        ev_id = dendogram["leaves"][i]
        cl_id = labels[ev_id]
        cluster_colors[cl_id] = leaf_colors[ev_id]
    return cluster_colors


def find_template_clusters(
    TpGroup,
    method="single",
    metric="correlation",
    criterion="distance",
    clustering_threshold=0.33,
    color_singleton="dimgray",
    ax_dendogram=None,
):
    """
    Find non-overlapping groups of similar templates
    with the hierarchical clustering package from scipy.
    """
    from scipy.spatial.distance import squareform
    from scipy.cluster import hierarchy

    # first, transform the CC matrix into a condensed matrix
    np.fill_diagonal(TpGroup.intertp_cc.values, 1.0)
    corr_dist = squareform(1.0 - TpGroup.intertp_cc.values)
    if corr_dist.min() < -1.0e-6:
        print("Prob with FMF")
    else:
        # avoid tiny negative values because of numerical imprecision
        corr_dist[corr_dist < 0.0] = 0.0
    # link the events
    Z = hierarchy.linkage(
        corr_dist, method=method, metric=metric, optimal_ordering=True
    )
    # get cluster labels
    labels = hierarchy.fcluster(Z, clustering_threshold, criterion=criterion)
    cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    if ax_dendogram is not None:
        # plot dendogram
        dendogram = hierarchy.dendrogram(
            Z,
            count_sort=True,
            above_threshold_color=color_singleton,
            color_threshold=clustering_threshold,
            ax=ax_dendogram,
        )
        # get cluster colors from the dendogram
        cluster_colors = extract_colors_from_tree(dendogram, labels, color_singleton)
        ## count all singleton clusters as one, for plotting purposes
        # n_clusters = np.sum(cluster_sizes > 1)
        # if np.sum(cluster_sizes == 1) > 0:
        #        n_clusters += 1
        #        sort_by_size = np.argsort(cluster_sizes)
        return labels, cluster_ids, cluster_sizes, dendogram, cluster_colors
    else:
        return labels, cluster_ids, cluster_sizes


# -------------------------------------------------
#           Convert and round times
# -------------------------------------------------


def round_time(t, sr=cfg.SAMPLING_RATE_HZ):
    """
    Parameters
    -----------
    t: scalar float,
        Time, in seconds, to be rounded so that the number
        of meaningful decimals is consistent with the precision
        allowed by the sampling rate.
    sr: scalar float, default to cfg.SAMPLING_RATE_HZ,
        Sampling rate of the data. It is used to
        round the time.

    Returns
    --------
    t: scalar float,
        Rounded time.
    """
    # convert t to samples
    t_samp = np.int64(t * sr)
    # get it back to seconds
    t = np.float64(t_samp) / sr
    return t


def sec_to_samp(t, sr=cfg.SAMPLING_RATE_HZ, epsilon=0.2):
    """Convert seconds to samples taking into account rounding errors.

    Parameters
    -----------
    """
    # we add epsilon so that we fall onto the right
    # integer number even if there is a small precision
    # error in the floating point number
    sign = np.sign(t)
    t_samp_float = abs(t * sr) + epsilon
    # round and restore sign
    t_samp_int = np.int64(sign * np.int64(t_samp_float))
    return t_samp_int


def time_range(
    start_time,
    end_time,
    dt_sec,
    unit="ms",
    unit_value={"ms": 1.0e3, "us": 1.0e6, "ns": 1.0e9},
):
    """Compute a range of datetime64.

    Parameters
    ------------
    start_time: string or datetime
        Start of the time range.
    end_time: string or datetime
        End of the time range.
    dt_sec: scalar float
        Time step, in seconds, of the time range.
    unit: string, default to 'ms'
        Unit in which dt_sec is converted in order to reach an integer number.
    unit_value: dictionary, optional
        Dictionary with the value of 1 second in different units.

    Returns
    ---------
    time_range: (n_samples,) numpy.ndarray of numpy.datetime64
        The time range computed from the input parameters.
    """
    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    dt = np.timedelta64(int(dt_sec * unit_value[unit]), unit)
    return np.arange(start_time, end_time, dt)


# -------------------------------------------------
#                    Regression
# -------------------------------------------------


def linear_regression(x, y):
    """
    cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

    Returns
    -------
    a: slope
    b: intercept
    r_val: correlation coefficient, usually
           people use the coefficient of determination
           R**2 = r_val**2 to measure the quality of
           the fit
    p_val: two-sided p-value for a hypothesis test whose null
           hypothesis is that the slope is zero
    std_err: standard error of the estimated slope
    """
    from scipy.stats import linregress

    a, b, r_val, p_val, std_err = linregress(x, y)
    return a, b, r_val, p_val, std_err


def weighted_linear_regression(X, Y, W=None):
    """
    Parameters
    -----------
    X: (n,) numpy array or list
    Y: (n,) numpy array or list
    W: default to None, (n,) numpy array or list

    Returns
    --------
    best_slope: scalar float,
        Best slope from the least square formula
    best_intercept: scalar float,
        Best intercept from the least square formula
    std_err: scalar float,
        Error on the slope
    """
    X = np.asarray(X)
    if W is None:
        W = np.ones(X.size)
    W_sum = W.sum()
    x_mean = np.sum(W * X) / W_sum
    y_mean = np.sum(W * Y) / W_sum
    x_var = np.sum(W * (X - x_mean) ** 2)
    xy_cov = np.sum(W * (X - x_mean) * (Y - y_mean))
    best_slope = xy_cov / x_var
    best_intercept = y_mean - best_slope * x_mean
    # errors in best_slope and best_intercept
    estimate = best_intercept + best_slope * X
    s2 = sum(estimate - Y) ** 2 / (Y.size - 2)
    s2_intercept = s2 * (1.0 / X.size + x_mean**2 / ((X.size - 1) * x_var))
    s2_slope = s2 * (1.0 / ((X.size - 1) * x_var))
    return best_slope, best_intercept, np.sqrt(s2_slope)


# -------------------------------------------------
#             Others
# -------------------------------------------------

def cov_mat_intersection(cov_mat, axis1=0, axis2=1):
    """Compute intersection between covariance matrix and plane.

    Note that we assume the following coordinate system:
    - X: westward
    - Y: southward
    - Z: upward

    Parameters
    ----------
    cov_mat : numpy.ndarray
        The (3x3) covariance matrix returned by Event.relocate(method='NLLoc').
    axis1 : integer, optional
        Index of the first axis defining the intersecting plane.
    axis2 : integer, optional
        Index of the second axis defining the intersecting plane.

    Returns
    -------
    max_unc : float
        Maximum uncertainty, in km, of the intersected covariance matrix.
    min_unc : float
        Minimum uncertainty, in km, of the intersected covariance matrix.
    az_max : float
        "Azimuth", that is, angle from `axis2` of maximum uncertainty.
    az_min : float
        "Azimuth", that is, angle from `axis2` of minimum uncertainty.
    """
    # X: west, Y: south, Z: upward
    s_68_3df = 3.52
    s_68_2df = 2.28
    # eigendecomposition of restricted matrix
    indexes = np.array([axis1, axis2])
    w, v = np.linalg.eigh(cov_mat[indexes, :][:, indexes])
    semi_axis_length = np.sqrt(s_68_2df * w)
    max_unc = np.max(semi_axis_length)
    min_unc = np.min(semi_axis_length)
    max_dir = v[:, w.argmax()]
    min_dir = v[:, w.argmin()]
    # "azimuth" is angle between `axis2` (`max_dir[1]`) and ellipse's semi-axis
    az_max = np.arctan2(max_dir[0], max_dir[1]) * 180.0 / np.pi
    az_min = (az_max + 90.) % 360.
    return max_unc, min_unc, az_max, az_min


def compute_distances(
    source_longitudes,
    source_latitudes,
    source_depths,
    receiver_longitudes,
    receiver_latitudes,
    receiver_depths,
    return_epicentral_distances=False,
):
    """
    Fast distance computation between all source points and all receivers.

    This function uses `cartopy.geodesic.Geodesic` to compute pair-wise distances
    between source points and receivers. It computes both hypocentral distances
    and, if specified, epicentral distances.

    Parameters
    ----------
    source_longitudes : numpy.ndarray or list
        Longitudes, in decimal degrees, of the source points.
    source_latitudes : numpy.ndarray or list
        Latitudes, in decimal degrees, of the source points.
    source_depths : numpy.ndarray or list
        Depths, in kilometers, of the source points.
    receiver_longitudes : numpy.ndarray or list
        Longitudes, in decimal degrees, of the receivers.
    receiver_latitudes : numpy.ndarray or list
        Latitudes, in decimal degrees, of the receivers.
    receiver_depths : numpy.ndarray or list
        Depths, in kilometers, of the receivers. Negative depths indicate
        receivers located at the surface.
    return_epicentral_distances : bool, optional
        Flag indicating whether to return epicentral distances in addition to
        hypocentral distances. Default is False.

    Returns
    -------
    hypocentral_distances : numpy.ndarray
        Array of hypocentral distances between source points and receivers.
        The shape of the array is (n_sources, n_receivers).
    epicentral_distances : numpy.ndarray, optional
        Array of epicentral distances between source points and receivers.
        This array is returned only if `return_epicentral_distances` is True.
        The shape of the array is (n_sources, n_receivers).
    """
    from cartopy.geodesic import Geodesic

    # convert types if necessary
    if isinstance(source_longitudes, list):
        source_longitudes = np.asarray(source_longitudes)
    if isinstance(source_latitudes, list):
        source_latitudes = np.asarray(source_latitudes)
    if isinstance(source_depths, list):
        source_depths = np.asarray(source_depths)

    # initialize distance array
    hypocentral_distances = np.zeros(
        (len(source_latitudes), len(receiver_latitudes)), dtype=np.float32
    )
    epicentral_distances = np.zeros(
        (len(source_latitudes), len(receiver_latitudes)), dtype=np.float32
    )
    # initialize the Geodesic instance
    G = Geodesic()
    for s in range(len(receiver_latitudes)):
        epi_distances = G.inverse(
            np.array([[receiver_longitudes[s], receiver_latitudes[s]]]),
            np.hstack(
                (source_longitudes[:, np.newaxis], source_latitudes[:, np.newaxis])
            ),
        )
        epicentral_distances[:, s] = np.asarray(epi_distances)[:, 0].squeeze() / 1000.0
        hypocentral_distances[:, s] = np.sqrt(
            epicentral_distances[:, s] ** 2 + (source_depths - receiver_depths[s]) ** 2
        )
    if return_epicentral_distances:
        return hypocentral_distances, epicentral_distances
    else:
        return hypocentral_distances


def event_count(
    event_timings_str,
    start_date,
    end_date,
    freq="1D",
    offset=0.0,
    trim_start=True,
    trim_end=False,
    mode="end",
):
    """
    Parameters
    ----------
    event_timings_str: list of array of str
        Timings of the events given as strings of characters.
    start_date: str
        Starting date of the event count time series.
    end_date: str
        End date of the event count time series.
    freq: str, default to '1D'
        Desired frequency of the event count time series.
        Default is one day.
    offset: float, default to 0.
        Fraction of the frequency used for defining
        the beginning of each bin. For example, offset=0.5
        with freq='1D' will return daily event counts
        from noon to noon.
    mode: str, default to 'end'
        Can be 'end' or 'beginning'. This string defines whether
        the seismicity counted between time 1 and time 2 is
        indexed at time 2 ('end') or time 1 ('beginning').

    Returns
    -------
    event_count: Pandas Series
        Pandas Series with temporal indexes defined
        by freq and base, and values given by the
        event count.
        To get a numpy array from this Pandas Series,
        use: event_count.values
    """
    import pandas as pd

    start_date = pd.to_datetime(start_date.replace(",", "-"))
    end_date = pd.to_datetime(end_date.replace(",", "-"))
    offset_str = "{}{}".format(offset, freq[-1])
    event_occurrence = pd.Series(
        data=np.ones(len(event_timings_str), dtype=np.int32),
        index=pd.to_datetime(np.asarray(event_timings_str).astype("U")).astype(
            "datetime64[ns]"
        ),
    )
    # trick to force a good match between initial indexes and new indexes
    event_occurrence[start_date] = 0
    event_occurrence[end_date] = 0
    if mode == "end":
        # note: we use mode='end' so that the number of events
        # counted at time t is the event count between t-dt and t
        # this is consistent with the timing convention of pandas diff()
        # namely: diff(t) = x(t)-x(t-dt)
        event_count = event_occurrence.groupby(
            pd.Grouper(freq=freq, offset=offset_str, label="right")
        ).agg("sum")
    elif mode == "beginning":
        event_count = event_occurrence.groupby(
            pd.Grouper(freq=freq, offset=offset_str, label="left")
        ).agg("sum")
    else:
        print("mode should be end or beginning")
        return
    if event_count.index[0] > pd.Timestamp(start_date):
        event_count[event_count.index[0] - pd.Timedelta(freq)] = 0
    if event_count.index[-1] < pd.Timestamp(end_date):
        event_count[event_count.index[-1] + pd.Timedelta(freq)] = 0
    if trim_start or offset == 0.0:
        event_count = event_count[event_count.index >= start_date]
    if trim_end or offset == 0.0:
        if offset > 0.0:
            stop_date = pd.to_datetime(end_date) + pd.Timedelta(freq)
        else:
            stop_date = end_date
        event_count = event_count[event_count.index <= stop_date]
    # force the manually added items to be well
    # located in time
    event_count.sort_index(inplace=True)
    return event_count


def get_np_array(
    stream,
    stations,
    components=["N", "E", "Z"],
    priority="HH",
    n_samples=None,
    component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
    verbose=True,
):
    """Fetch data from Obspy Stream and returns an ndarray.

    Parameters
    -----------
    stream: Obspy Stream instance
        The Obspy Stream instance with the waveform time series.
    stations: list of strings
        Names of the stations to include in the output array. Define the order
        of the station axis.
    components: list of strings, default to ['N','E','Z']
        Names of the components to include in the output array. Define the order
        of the component axis.
    component_aliases: dictionary, optional
        Sometimes, components might be named differently than N, E, Z. This
        dictionary tells the function which alternative component names can be
        associated with each "canonical" component. For example,
        `component_aliases['N'] = ['N', '1']` means that the function will also
        check the '1' component in case the 'N' component doesn't exist.
    priority: string, default to 'HH'
        When a station has multiple instruments, this string tells which
        channel to use in priority.
    n_samples: scalar int, default to None
        Duration, in samples, of the output numpy.ndarray. Select the
        `n_samples` first samples of each trace. If None, take `n_samples` as
        the length of the first trace.
    verbose: boolean, default to True
        If True, print extra output in case the target data cannot be fetched.

    Returns
    ---------
    data: (n_stations, n_components, n_samples) numpy.ndarray
        The waveform time series formatted as an numpy.ndarray.
    """
    if len(stream) == 0:
        print("The input data stream is empty!")
        return
    if n_samples is None:
        n_samples = stream[0].data.size
    data = np.zeros((len(stations), len(components), n_samples), dtype=np.float32)
    for s, sta in enumerate(stations):
        for c, cp in enumerate(components):
            for cp_alias in component_aliases[cp]:
                channel = stream.select(station=sta, component=cp_alias)
                if len(channel) > 0:
                    # succesfully retrieved data
                    break
            if len(channel) > 0:
                try:
                    # try selecting the preferred channel if it exists
                    cha = channel.select(channel=f"{priority}{cp_alias}")[0]
                    # data[s, c, :] = channel.select(
                    #        channel=f'{priority}{cp_alias}')[0].data[:n_samples]
                except IndexError:
                    cha = channel[0]
                    # data[s, c, :] = channel[0].data[:n_samples]
                if len(cha.data) < n_samples:
                    length_diff = n_samples - len(cha.data)
                    data[s, c, :] = np.hstack(
                        (cha.data, np.zeros(length_diff, dtype=np.float32))
                    )
                else:
                    data[s, c, :] = cha.data[:n_samples]
    return data


def max_norm(X):
    max_ = np.abs(X).max()
    if max_ != 0.0:
        return X / max_
    else:
        return X


def running_mad(time_series, window, n_mad=10.0, overlap=0.75):
    from scipy.stats import median_abs_deviation as scimad

    # calculate n_windows given window
    # and overlap
    shift = int((1.0 - overlap) * window)
    n_windows = int((len(time_series) - window) // shift) + 1
    mad_ = np.zeros(n_windows + 2, dtype=np.float32)
    med_ = np.zeros(n_windows + 2, dtype=np.float32)
    time = np.zeros(n_windows + 2, dtype=np.float32)
    for i in range(1, n_windows + 1):
        i1 = i * shift
        i2 = min(len(time_series), i1 + window)
        sliding_window = time_series[i1:i2]
        # non_zero = cnr_window != 0
        # if sum(non_zero) < 3:
        #    # won't be possible to calculate median
        #    # and mad on that few samples
        #    continue
        # med_[i] = np.median(cnr_window[non_zero])
        # mad_[i] = scimad(cnr_window[non_zero])
        med_[i] = np.median(sliding_window)
        mad_[i] = scimad(sliding_window)
        time[i] = (i1 + i2) / 2.0
    # add boundary cases manually
    time[0] = 0.0
    mad_[0] = mad_[1]
    med_[0] = med_[1]
    time[-1] = len(time_series)
    mad_[-1] = mad_[-2]
    med_[-1] = med_[-2]
    running_stat = med_ + n_mad * mad_
    interpolator = interp1d(
        time,
        running_stat,
        kind="slinear",
        fill_value=(running_stat[0], running_stat[-1]),
        bounds_error=False,
    )
    full_time = np.arange(0, len(time_series))
    running_stat = interpolator(full_time)
    return running_stat

def two_point_epicentral_distance(lon_1, lat_1, lon_2, lat_2):
    """Compute the distance between two points.


    Parameters
    -----------
    lon_1: scalar, float
        Longitude of Point 1.
    lat_1: scalar, float
        Latitude of Point 1.
    lon_2: scalar, float
        Longitude of Point 2.
    lat_2: scalar, float
        Latitude of Point 2.

    Returns
    ---------
    dist: scalar, float
        Distance between Point 1 and Point 2 in kilometers.
    """
    from obspy.geodetics.base import calc_vincenty_inverse

    dist, az, baz = calc_vincenty_inverse(lat_1, lon_1, lat_2, lon_2)
    dist /= 1000.0  # from m to km
    return dist

def two_point_distance(lon_1, lat_1, depth_1, lon_2, lat_2, depth_2):
    """Compute the distance between two points.


    Parameters
    -----------
    lon_1: scalar, float
        Longitude of Point 1.
    lat_1: scalar, float
        Latitude of Point 1.
    depth_1: scalar, float
        Depth of Point 1 (in km).
    lon_2: scalar, float
        Longitude of Point 2.
    lat_2: scalar, float
        Latitude of Point 2.
    depth_2: scalar, float
        Depth of Point 2 (in km).

    Returns
    ---------
    dist: scalar, float
        Distance between Point 1 and Point 2 in kilometers.
    """

    from obspy.geodetics.base import calc_vincenty_inverse

    dist, az, baz = calc_vincenty_inverse(lat_1, lon_1, lat_2, lon_2)
    dist /= 1000.0  # from m to km
    dist = np.sqrt(dist**2 + (depth_1 - depth_2) ** 2)
    return dist


def donefun():
    """
    Super useful function.
    """
    print(
        """⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⢀⡤⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⢀⡏⠀⠀⠈⠳⣄⠀⠀⠀⠀⠀⣀⠴⠋⠉⠉⡆⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠈⠉⠉⠙⠓⠚⠁⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⢀⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣄⠀⠀⠀⠀
    ⠀⠀⠀⠀⡞⠀⠀⠀⠀⠀⠶⠀⠀⠀⠀⠀⠀⠦⠀⠀⠀⠀⠀⠸⡆⠀⠀⠀
    ⢠⣤⣶⣾⣧⣤⣤⣀⡀⠀⠀⠀⠀⠈⠀⠀⠀⢀⡤⠴⠶⠤⢤⡀⣧⣀⣀⠀
    ⠻⠶⣾⠁⠀⠀⠀⠀⠙⣆⠀⠀⠀⠀⠀⠀⣰⠋⠀⠀⠀⠀⠀⢹⣿⣭⣽⠇
    ⠀⠀⠙⠤⠴⢤⡤⠤⠤⠋⠉⠉⠉⠉⠉⠉⠉⠳⠖⠦⠤⠶⠦⠞⠁⠀⠀⠀
                ALL DONE!⠀⠀⠀⠀
    """
    )


def write_lock_file(path, check=False, flush=False):
    if check:
        assert os.path.isfile(path) is False, f"Lock file {path} already exists!"
    if not flush:
        open(path, "w").close()
    else:
        f = open(path, "w")
        f.flush()
        os.fsync(f.fileno())
        f.close()


def read_write_waiting_list(func, path, unit_wait_time=0.2):
    """Read/write queue to avoid conflicts between jobs."""
    from time import sleep

    path_no_ext, _ = os.path.splitext(path)
    while True:
        try:
            path_lock = path_no_ext + "_lock"
            path_wait = path_no_ext + "_waiting_list"
            sleep(unit_wait_time * np.random.random())
            waiting_list_position = len(glob.glob(path_wait + "*"))
            waiting_list_ticket = f"{path_wait}{waiting_list_position:d}"
            while True:
                try:
                    # take place in waiting list by creating empty file
                    write_lock_file(waiting_list_ticket, check=True)
                    break
                except AssertionError:
                    # several jobs trying to create the same ticket?
                    # randomness will solve the conflict
                    waiting_list_position += np.random.randint(5)
                    waiting_list_ticket = f"{path_wait}{waiting_list_position:d}"
            next_place_ticket = f"{path_wait}{waiting_list_position-1:d}"
            # print(f"1: Created {os.path.basename(waiting_list_ticket)}")
            while True:
                # sleep(unit_wait_time * np.random.random())
                if waiting_list_position == 0:
                    # is first in the waiting list!
                    if not os.path.isfile(path_lock):
                        # first, create lock file
                        write_lock_file(path_lock, flush=True)
                        # print(f"2: Created {os.path.basename(path_lock)}")
                        # then, free the waiting list position #0
                        pathlib.Path(waiting_list_ticket).unlink()
                        # print(f"2: Deleted {os.path.basename(waiting_list_ticket)}")
                        # now the process can proceed with the reading or writing
                        break
                    else:
                        # wait a bit
                        sleep(unit_wait_time * np.random.random())
                elif not os.path.isfile(next_place_ticket):
                    # front place in the waiting list was freed!
                    # first, create new ticket at the new position
                    write_lock_file(next_place_ticket)
                    # print(f"3: Created {os.path.basename(next_place_ticket)}")
                    # then, free previous place in the waiting list
                    pathlib.Path(waiting_list_ticket).unlink()
                    # print(f"3: Deleted {os.path.basename(waiting_list_ticket)}")
                    # update place in waiting list
                    waiting_list_position -= 1
                    # update ticket names
                    waiting_list_ticket = f"{path_wait}{waiting_list_position:d}"
                    next_place_ticket = f"{path_wait}{waiting_list_position-1:d}"
                    # and wait for its turn
                    sleep(unit_wait_time * np.random.random())
                else:
                    # waiting list didn't change, just wait
                    sleep(unit_wait_time * np.random.random())
            # start reading/writing
            try:
                func(path)
            except Exception as e:
                pathlib.Path(path_lock).unlink()
                raise (e)
            pathlib.Path(path_lock).unlink()
            break
            # done!
        except FileNotFoundError:
            print(
                "4: Concurent error, reset queue "
                f"(last ticket was {waiting_list_ticket})"
            )
            if waiting_list_position == 0:
                print(os.path.isfile(path_lock))
                pathlib.Path(path_lock).unlink()
            continue

# =======================================================
#         routines for automatic picking
# =======================================================


def normalize_batch(seismogram, normalization_window_sample=3000, overlap=0.50):
    """Apply Z-score normalization in running windows.

    Following Zhu et al. 2019, this function applied Z-score
    normalization in running windows with length `normalization_window_sample`.

    Parameters
    -----------
    seismogram : numpy.ndarray
        Three-component seismograms. `seismogram` has shape
        (num_traces, num_channels=3, num_time_samples).
    normalization_window_sample : integer, optional
        The window length, in samples, over which normalization is applied.
        Default is 3000 (like in Zhu et al. 2019).

    Returns
    --------
    normalized_seismogram : numpy.ndarray
        Normalized seismogram with same shape as `seismogram`.
    """
    from scipy.interpolate import interp1d

    #shift = normalization_window_sample // 2
    shift = int((1. - overlap) * normalization_window_sample)
    num_stations, num_channels, num_time_samples = seismogram.shape

    # std in sliding windows
    seismogram_pad = np.pad(
        seismogram, ((0, 0), (0, 0), (shift, shift)), mode="reflect"
    )
    # time = np.arange(0, num_time_samples, shift, dtype=np.int32)
    seismogram_view = np.lib.stride_tricks.sliding_window_view(
        seismogram_pad, normalization_window_sample, axis=-1
    )[:, :, ::shift, :]
    sliding_std = np.std(seismogram_view, axis=-1)
    sliding_mean = np.mean(seismogram_view, axis=-1)

    # time at centers of sliding windows
    num_sliding_windows = seismogram_view.shape[2]
    time = np.linspace(shift, num_time_samples - shift, num_sliding_windows)

    sliding_std[:, :, -1], sliding_mean[:, :, -1] = (
        sliding_std[:, :, -2],
        sliding_mean[:, :, -2],
    )
    sliding_std[:, :, 0], sliding_mean[:, :, 0] = (
        sliding_std[:, :, 1],
        sliding_mean[:, :, 1],
    )
    sliding_std[sliding_std == 0] = 1

    # normalize data with sliding std and mean
    t_interp = np.arange(num_time_samples)
    std_interp = np.stack(
            tuple(np.interp(
                t_interp, time, sld_std, left=sld_std[0], right=sld_std[-1]
                )
                for sld_std in sliding_std.reshape(-1, sliding_std.shape[-1])
                ),
            axis=0
            ).reshape(
                    sliding_std.shape[:-1] + (len(t_interp),)
                    )
    mean_interp = np.stack(
            tuple(np.interp(
                t_interp, time, m_std, left=m_std[0], right=m_std[-1]
                )
                for m_std in sliding_mean.reshape(-1, sliding_mean.shape[-1])
                ),
            axis=0
            ).reshape(
                    sliding_mean.shape[:-1] + (len(t_interp),)
                    )

    seismogram = (seismogram - mean_interp) / std_interp

    return seismogram

def trigger_picks(
        probability,
        threshold,
        minimum_peak_distance_samp=int(1. * cfg.SAMPLING_RATE_HZ)
        ):
    """
    Parameters
    ----------
    probability : 1D array_like
    threshold : float
    minimum_peak_distance_samp : integer

    Returns
    -------
    probability_at_peak : 1D array_like
    peak_indexes : 1D array_like
    """
    peak_indexes = _detect_peaks(
            probability, mph=threshold, mpd=minimum_peak_distance_samp
            )
    return (
            np.atleast_1d(probability[peak_indexes]),
            np.atleast_1d(peak_indexes)
            )

def get_picks(
        picks,
        buffer_length=int(2. * cfg.SAMPLING_RATE_HZ),
        prior_knowledge=None,
        search_win_samp=int(4. * cfg.SAMPLING_RATE_HZ)
        ):

    """Select a single P- and S-pick on each 3-comp seismogram.
    
    Parameters
    ----------
    picks: dictionary
        Dictionary returned by `automatic_picking`.
    buffer_length: scalar int, optional
        Picks that are before this buffer length, in samples, are discarded.
    prior_knowledge: pandas.DataFrame, optional
        If given, picks that are closer to the a priori pick
        (for example, given by a preliminary location) will be given
        a larger weight and will be more likely to be selected. In practice,
        pick probabilities are multiplied by gaussian weights and the highest
        modified pick probability is selected.
    search_win_samp: scalar int, optional
        Standard deviation, in samples, used in the gaussian weights.
    """
    for st in picks["P_picks"].keys():
        if prior_knowledge is not None and st in prior_knowledge.index:
            prior_P = prior_knowledge.loc[st, "P"]
            prior_S = prior_knowledge.loc[st, "S"]
        else:
            prior_P, prior_S = None, None
        #for n in range(len(picks["P_picks"][st])):
        # ----------------
        # remove picks from the buffer length
        valid_P_picks = picks["P_picks"][st] > int(buffer_length)
        valid_S_picks = picks["S_picks"][st] > int(buffer_length)
        picks["P_picks"][st] = picks["P_picks"][st][valid_P_picks]
        picks["S_picks"][st] = picks["S_picks"][st][valid_S_picks]
        picks["P_proba"][st] = picks["P_proba"][st][valid_P_picks]
        picks["S_proba"][st] = picks["S_proba"][st][valid_S_picks]
        search_S_pick = True
        search_P_pick = True
        if len(picks["S_picks"][st]) == 0:
            # if no valid S pick: fill in with nan
            picks["S_picks"][st] = np.nan
            picks["S_proba"][st] = np.nan
            search_S_pick = False
        if len(picks["P_picks"][st]) == 0:
            # if no valid P pick: fill in with nan
            picks["P_picks"][st] = np.nan
            picks["P_proba"][st] = np.nan
            search_P_pick = False
        if search_S_pick:
            if prior_S is None:
                # take only the highest probability trigger
                best_S_trigger = picks["S_proba"][st].argmax()
            else:
                # use a priori picks
                tapered_S_probas = (
                        picks["S_proba"][st]
                        *
                        np.exp(
                            -(picks["S_picks"][st] - prior_S)**2/(2.*search_win_samp**2)
                            )
                        )
                best_S_trigger = tapered_S_probas.argmax()
                ## don't keep if too far from a priori
                #if abs(picks["S_picks"][st][best_S_trigger] - prior_S) > 4 * search_win_samp:
                #    best_S_trigger = np.nan
            if np.isnan(best_S_trigger):
                picks["S_picks"][st] = np.nan 
                picks["S_proba"][st] = np.nan 
            else:
                picks["S_picks"][st] = picks["S_picks"][st][best_S_trigger]
                picks["S_proba"][st] = picks["S_proba"][st][best_S_trigger]
            # update P picks: keep only those that are before the best S pick
            if search_P_pick:
                valid_P_picks = picks["P_picks"][st] < picks["S_picks"][st]
                picks["P_picks"][st] = picks["P_picks"][st][valid_P_picks]
                picks["P_proba"][st] = picks["P_proba"][st][valid_P_picks]
                if len(picks["P_picks"][st]) == 0:
                    # if no valid P pick: fill in with nan
                    picks["P_picks"][st] = np.nan
                    picks["P_proba"][st] = np.nan
                    search_P_pick = False
        if search_P_pick:
            if prior_P is None:
                # take only the highest probability trigger
                best_P_trigger = picks["P_proba"][st].argmax()
            else:
                # use a priori picks
                tapered_P_probas = (
                        picks["P_proba"][st]
                        *
                        np.exp(
                            -(picks["P_picks"][st] - prior_P)**2/(2.*search_win_samp**2)
                            )
                        )
                best_P_trigger = tapered_P_probas.argmax()
                ## don't keep if too far from a priori
                #if abs(picks["P_picks"][st][best_P_trigger] - prior_P) > 4 * search_win_samp:
                #    best_P_trigger = np.nan
            if np.isnan(best_P_trigger):
                picks["P_picks"][st] = np.nan 
                picks["P_proba"][st] = np.nan 
            else:
                picks["P_picks"][st] = picks["P_picks"][st][best_P_trigger]
                picks["P_proba"][st] = picks["P_proba"][st][best_P_trigger]
        # convert picks to float to allow NaNs
        picks["P_picks"][st] = np.atleast_1d(np.float32(picks["P_picks"][st]))
        picks["S_picks"][st] = np.atleast_1d(np.float32(picks["S_picks"][st]))
        picks["P_proba"][st] = np.atleast_1d(np.float32(picks["P_proba"][st]))
        picks["S_proba"][st] = np.atleast_1d(np.float32(picks["S_proba"][st]))
    return picks

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

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1]:http://nbviewer.ipython.org/github/demotu/BMC/blob/master/
        notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype("float64")
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True
            )
        ]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

