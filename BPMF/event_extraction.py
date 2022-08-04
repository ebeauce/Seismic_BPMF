import os
import sys

sys.path.append("/nobackup1/ebeauce/DANA/")
import automatic_detection as autodet

import numpy as np
import matplotlib.pyplot as plt
import glob

import obspy as obs
from obspy.core import UTCDateTime as udt

from time import time as give_time

from .config import cfg


def align_traces(event, travel_times, stations, time_before_P, duration):
    for s, sta in enumerate(stations):
        for tr in event.select(station=sta):
            starttime = tr.stats.starttime + travel_times[s] - time_before_P
            endtime = starttime + duration
            tr.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=None)
    return event


def extract_event(origin_time, net, duration=60.0, offset_start=0.0, folder="raw"):
    """
    Event extractor: sequential version.
    """

    n_stations = len(net.stations)
    n_components = len(net.components)
    # obspy stream object
    S = obs.Stream()
    ot = udt(origin_time)
    # make sure ot falls onto a sample
    ot = udt(float(int(ot.timestamp * cfg.sampling_rate)) / cfg.sampling_rate)
    t1 = give_time()
    for s in range(n_stations):
        for c in range(n_components):
            filename = os.path.join(
                cfg.input_path,
                str(ot.year),
                "continuous{:03d}".format(ot.julday),
                "{}/*.{}*{}*".format(folder, net.stations[s], net.components[c]),
            )
            file = glob.glob(filename)
            if len(file) > 0:
                file = file[0]
            else:
                # print('No data for {}.{}'.format(net.stations[s], net.components[c]))
                continue
            try:
                tr = obs.read(
                    file,
                    starttime=ot + offset_start,
                    endtime=ot + offset_start + duration + 1,
                )[0]
                tr.trim(
                    starttime=ot + offset_start,
                    endtime=ot + offset_start + duration + 1,
                    pad=True,
                    fill_value=0.0,
                )
            except Exception as e:
                print("Error when trying to read {}".format(file))
                continue
            # print(tr)
            S += tr
    t2 = give_time()
    print("{:.2f} s to extract the waveforms from {}.".format(t2 - t1, ot))
    return S


def extract_event_parallel(
    origin_time,
    net,
    duration=60.0,
    offset_start=0.0,
    folder="raw",
    attach_response=False,
    verbose=True,
    check_compression=False,
):
    """
    Event extractor: parallelized version.
    """
    import itertools
    import concurrent.futures

    n_stations = len(net.stations)
    n_components = len(net.components)
    ot = udt(origin_time)
    # make sure ot falls onto a sample
    ot = udt(float(int(ot.timestamp * cfg.sampling_rate)) / cfg.sampling_rate)
    # path to data
    data_path = os.path.join(
        cfg.input_path, str(ot.year), "continuous{:03d}".format(ot.julday), folder
    )
    target_starttime = ot + offset_start
    target_endtime = target_starttime + duration + 1.0
    t1 = give_time()
    # get all names
    files = generate_file_list(data_path, net)
    # read all traces in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                load_data,
                f,
                target_starttime,
                target_endtime,
                attach_response,
                verbose,
                check_compression,
            )
            for f in iter(files)
        ]
        results = [fut.result() for fut in futures]
        traces = obs.Stream([tr for tr in results if tr is not None])
    t2 = give_time()
    print("{:.2f} s to extract the waveforms from {}.".format(t2 - t1, ot))
    return traces


def extract_event_realigned(
    origin_time,
    net,
    travel_times,
    duration=20.0,
    offset_start=0.0,
    folder="raw",
    attach_response=False,
    verbose=True,
    check_compression=False,
):
    """
    Event extractor: parallelized version.
    """
    import itertools
    import concurrent.futures

    n_stations = len(net.stations)
    n_components = len(net.components)
    ot = udt(origin_time)
    # make sure ot falls onto a sample
    ot = udt(float(int(ot.timestamp * cfg.sampling_rate)) / cfg.sampling_rate)
    # path to data
    data_path = os.path.join(
        cfg.input_path, str(ot.year), "continuous{:03d}".format(ot.julday), folder
    )
    t1 = give_time()
    # get all names
    files, tts = generate_file_list(data_path, net, travel_times=travel_times)
    target_starttimes = [(ot + offset_start + tts[s]) for s in range(len(tts))]
    target_endtimes = [(time + duration) for time in target_starttimes]
    # read all traces in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(load_data, *f, attach_response, verbose, check_compression)
            for f in iter(zip(files, target_starttimes, target_endtimes))
        ]
        results = [fut.result() for fut in futures]
        traces = obs.Stream([tr for tr in results if tr is not None])
    t2 = give_time()
    print("{:.2f} s to extract the waveforms from {}.".format(t2 - t1, ot))
    return traces


def load_data(
    filename,
    target_starttime,
    target_endtime,
    attach_response=False,
    verbose=True,
    check_compression=True,
):
    try:
        trace = obs.read(
            filename,
            starttime=target_starttime,
            endtime=target_endtime,
            check_compression=check_compression,
        )[0]
        # if necessary: fill trace with masked elements at
        # the beginning and at the end
        trace = trace.trim(
            starttime=target_starttime,
            endtime=target_endtime,
            fill_value=None,
            pad=True,
        )
        if attach_response:
            # attach the instrument response for later
            # have the possibility to easily correct it
            # assume that the files are organized in the
            # usual obspyDMT convention
            base = os.path.dirname(os.path.dirname(filename))
            response_file = os.path.join(base, "resp", "STXML." + trace.id)
            inv = obs.read_inventory(response_file)
            trace.attach_response(inv)
        return trace
    except Exception as e:
        if verbose:
            print(e)
            print("Error when trying to read {}".format(filename))


def generate_file_list(data_path, net, travel_times=None):
    files = []
    if travel_times is not None:
        tts = []
        if travel_times.ndim == 1:
            # add a component axis
            travel_times = np.repeat(travel_times, len(net.components)).reshape(
                len(net.stations), len(net.components)
            )
    for s in range(len(net.stations)):
        for c in range(len(net.components)):
            target_files = "*{}*{}*".format(net.stations[s], net.components[c])
            new_files = glob.glob(os.path.join(data_path, target_files))
            files.extend(new_files)
            if travel_times is not None:
                tts.extend([travel_times[s, c]] * len(new_files))
    if travel_times is not None:
        return files, tts
    else:
        return files


def preprocess_event(
    event,
    freqmin=None,
    freqmax=None,
    target_SR=None,
    remove_response=False,
    remove_sensitivity=False,
    plot_resp=False,
    target_duration=None,
    verbose=True,
    unit="VEL",
    **kwargs,
):
    preprocessed_event = obs.Stream()
    if len(event) == 0:
        print("Input data is empty!")
        return preprocessed_event
    # start by cleaning the gaps if there are any
    for tr in event:
        if np.isnan(tr.data.max()):
            print(f"Problem with {tr.id} (detected NaNs)!")
            continue
        # split will lose information about start and end times
        # if the start or the end is masked
        t1 = udt(tr.stats.starttime.timestamp)
        t2 = udt(tr.stats.endtime.timestamp)
        tr = tr.split()
        tr.detrend("constant")
        tr.detrend("linear")
        tr.taper(0.05, type="cosine")
        # it's now safe to fill the gaps with zeros
        tr.merge(fill_value=0.0)[0]
        tr.trim(starttime=t1, endtime=t2, pad=True, fill_value=0.0)
        preprocessed_event += tr
    # delete the original data to save memory
    del event
    # resample if necessary:
    for tr in preprocessed_event:
        if target_SR is None:
            continue
        sr_ratio = tr.stats.sampling_rate / target_SR
        if sr_ratio > 1:
            tr.data = autodet.utils.lowpass_chebyshev_II(
                tr.data,
                0.49 * target_SR,
                tr.stats.sampling_rate,
                order=10,
                min_attenuation_dB=40.0,
                zerophase=True,
            )
            if np.round(sr_ratio, decimals=0) == sr_ratio:
                # tr's sampling rate is an integer
                # multiple of target_SR
                # do not re-filter the data
                tr.decimate(int(sr_ratio), no_filter=True)
            else:
                tr.resample(target_SR, no_filter=True)
        elif sr_ratio < 1:
            if verbose:
                print("Sampling rate is too high!")
                print(tr)
            preprocessed_event.remove(tr)
            continue
        else:
            pass
    if target_duration is not None:
        for i in range(len(preprocessed_event)):
            n_samples = autodet.utils.sec_to_samp(
                target_duration, sr=preprocessed_event[i].stats.sampling_rate
            )
            preprocessed_event[i].data = preprocessed_event[i].data[:n_samples]
    # remove response if requested
    if remove_response:
        for tr in preprocessed_event:
            # assume that the instrument response
            # is already attached to the trace
            T_max = tr.stats.npts * tr.stats.delta
            T_min = tr.stats.delta
            f_min = 1.0 / T_max
            f_max = 1.0 / (2.0 * T_min)
            pre_filt = [f_min, 3.0 * f_min, 0.90 * f_max, 0.97 * f_max]
            tr.remove_response(pre_filt=pre_filt, output=unit, plot=plot_resp)
    elif remove_sensitivity:
        for tr in preprocessed_event:
            tr.remove_sensitivity()
    # filter
    preprocessed_event.detrend("constant")
    preprocessed_event.detrend("linear")
    preprocessed_event.taper(0.02, type="cosine")
    if freqmin is None and freqmax is None:
        # no filtering
        pass
    elif freqmin is None:
        # lowpass filtering
        preprocessed_event.filter("lowpass", freq=freqmax, zerophase=True)
    elif freqmax is None:
        # highpass filtering
        preprocessed_event.filter("highpass", freq=freqmin, zerophase=True)
    else:
        # bandpass filtering
        preprocessed_event.filter(
            "bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True
        )
    return preprocessed_event
