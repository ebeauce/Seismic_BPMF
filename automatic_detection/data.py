import os

from .config import cfg

import numpy as np
import h5py as h5
import glob

from obspy.core import UTCDateTime as udt
import obspy as obs

from time import time as give_time

import itertools
import concurrent.futures

# this is only an example with one function to read data
# custom functions should be added to this module to adapt
# to the architecture you are using to store your data

def replace_zeros_with_white_noise(data):
    #delta = (data.max() - data.min())/2.
    #zeros = np.abs(data) < delta / 1.e20
    zeros = np.abs(data) < 1.e-10
    if zeros.sum() == data.size:
        return np.random.normal(loc=0., scale=1.0, size=data.size).astype(np.float32)
    std = np.std(data[~zeros])
    data[zeros] = np.random.normal(loc=0., scale=std, size=zeros.sum()).astype(np.float32)
    return data

def fill_incomplete_trace(trace, target_starttime, target_endtime):
    dt_before = trace.stats.starttime.timestamp - target_starttime.timestamp
    tr_before = obs.Trace(data=np.zeros(int(dt_before*trace.stats.sampling_rate), dtype=trace.data.dtype))
    tr_before.stats.sampling_rate = trace.stats.sampling_rate
    tr_before.stats.starttime = target_starttime
    tr_before.id = trace.id
    # ---------------------
    dt_after = target_endtime.timestamp - trace.stats.endtime.timestamp
    tr_after = obs.Trace(data=np.zeros(int(dt_after*trace.stats.sampling_rate), dtype=trace.data.dtype))
    tr_after.stats.sampling_rate = trace.stats.sampling_rate
    tr_after.stats.starttime = trace.stats.endtime
    tr_after.id = trace.id
    # ---------------------
    S = obs.Stream()
    S += tr_before
    S += trace
    S += tr_after
    S.merge(fill_value='interpolate')
    return S[0]


def ReadData(date, net, folder='processed'):
    """
    ReadData(date, net)
    """
    date = udt(date)
    # ----------------------------
    # load the table
    table = {}
    with h5.File(cfg.input_path + 'table.h5', mode='r') as f:
        for key in f.keys():
            table[key] = f[key][()]
    # ----------------------------
    n_stations = len(net.stations)
    n_components = len(net.components)
    n_samples = int(3600. * 24. * cfg.sampling_rate)
    waveforms = np.zeros((n_stations, n_components, n_samples), dtype=np.float32)
    # -------------------------
    time_looking_for_file = 0.
    time_loading_data = 0.
    time_adjusting_data = 0.
    for s in range(n_stations):
        for c in range(n_components):
            filename = cfg.input_path\
                       + table[date.strftime('%Y,%m,%d')]\
                       + '/{}/*{}*{}*'.format(folder, net.stations[s], net.components[c])
            t1 = give_time()
            file = glob.glob(filename)
            t2 = give_time()
            time_looking_for_file += (t2-t1)
            if len(file) == 0:
                # no data
                continue
            t1 = give_time()
            trace = obs.read(file[0])[0]
            t2 = give_time()
            time_loading_data += (t2-t1)
            target_endtime = date + 3600.*24. + trace.stats.delta
            trace = trace.slice(starttime=date, endtime=target_endtime)
            if trace.stats.starttime.timestamp != date.timestamp or\
               trace.data.size < n_samples:
                t1 = give_time()
                trace = fill_incomplete_trace(trace, date, target_endtime)
                t2 = give_time()
                time_adjusting_data += (t2-t1)
            waveforms[s, c, :] = trace.data[:n_samples]
    # --------------------------
    data = {}
    data['waveforms'] = waveforms
    data['metadata'] = {}
    data['metadata']['sampling_rate'] = cfg.sampling_rate
    data['metadata']['networks'] = net.networks
    data['metadata']['stations'] = net.stations
    data['metadata']['components'] = net.components
    data['metadata']['date'] = date
    #print('{:.2f} s to look for data files'.format(time_looking_for_file))
    #print('{:.2f} s to load data'.format(time_loading_data))
    #print('{:.2f} s to reslice the data traces'.format(time_adjusting_data))
    return data

def ReadData_parallel(date, net, priority='HH', duration=24.*3600.,
                      replace_zeros=False, return_traces=False,
                      folder='processed_2_12', verbose=True,
                      remove_response=False, remove_sensitivity=False,
                      check_compression=False, headonly=False,
                      check_sampling_rate=True):
    """
    Parallelize reading operations with ThreadPoolExecutor
    from the concurrent package.

    Parameters
    -----------
    date: string,
        Date of the target day from which to read data.
    net: Network object,
    priority: string, default to 'HH'
        When several instruments are found with the same
        station code name, this string defines which
        instrument is loaded.
    duration: float, default to 24*3600
        The duration in seconds of the extracted data.
    replace_zeros: boolean, default to False
        If True, zeros in the data will be replaced by
        white noise with same std as the non-zero data.
        This is useful for template matching.
    return_traces: boolean, default to False
        If True, return obspy traces in addition to the
        numpy array waveforms.
    folder: string, default to 'processed_2_12'
        Subfolder name from which to read data.
    verbose: boolean, default to True
        If true, print the warning messages.
    remove_response: boolean, default to False
        If True, remove the instrumental response from the traces.
    remove_sensitivity: boolean, default to False
        If True, remove the instrument sensitivty from the traces.
    check_compression: boolean, default to False
        If True, check the compression style when reading
        the traces. It slows down the reading a bit, and it
        shouldn't be necessary on preprocessed traces.
    headonly: boolean, default to False
        If True, only the headers of the miniseed/sac files.
    check_sampling_rate: boolean, default to True
        If True, check the traces' sampling rate against the
        expected sampling rate from the parameter file.
        This should be set to False when reading raw data with
        several possible sampling rates.

    Returns
    ----------
    data: dictionary,
        Python dictionary with the data and metadata requested.
    """
    date = udt(date)
    # ----------------------------
    # data loading kwargs
    loading_kwargs = {}
    loading_kwargs['replace_zeros'] = replace_zeros
    loading_kwargs['remove_response'] = remove_response
    loading_kwargs['remove_sensitivity'] = remove_sensitivity
    loading_kwargs['check_compression'] = check_compression
    loading_kwargs['headonly'] = headonly
    loading_kwargs['check_sampling_rate'] = check_sampling_rate
    loading_kwargs['verbose'] = verbose
    # ----------------------------
    target_starttime = date
    target_endtime = date + duration + int(1./cfg.sampling_rate+1)
    # ----------------------------
    n_stations = len(net.stations)
    n_components = len(net.components)
    n_samples = int(duration*cfg.sampling_rate)
    waveforms = np.zeros((n_stations, n_components, n_samples), dtype=np.float32)
    # -------------------------
    data_availability = np.ones((n_stations, n_components), dtype=np.bool)
    data_path = os.path.join(cfg.input_path, str(date.year),
                             'continuous{:03d}'.format(date.julday),
                             folder, '')
    t1 = give_time()
    # get all names
    files = generate_file_list(data_path, net)
    # read all traces in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_data, f, target_starttime,
                                   target_endtime, **loading_kwargs)\
                     for f in iter(files)]
        results = [fut.result() for fut in futures]
        traces = obs.Stream([tr for tr in results if tr is not None])
    t2 = give_time()
    #print('{:.2f}s to load the traces.'.format(t2-t1))
    # initialize output dictionary
    data = {}
    t1 = give_time()
    if return_traces:
        data['traces'] = obs.Stream()
    for s in range(n_stations):
        for c in range(n_components):
            trace = traces.select(station=net.stations[s],
                                  component=net.components[c])
            if len(trace) == 0:
                # no data
                data_availability[s, c] = False
                continue
            elif len(trace) > 1:
                if verbose:
                    print('More than one trace were found:')
                    print(trace)
                if trace[0].stats.npts < trace[1].stats.npts:
                    print('The first trace might not be the best choice!!')
                trace = trace[0]
            else:
                trace = trace[0]
            if not headonly:
                waveforms[s, c, :] = trace.data[:n_samples]
            if return_traces:
                data['traces'] += trace
    t2 = give_time()
    #print('{:.2f}s to preprocess the data.'.format(t2-t1))
    # --------------------------
    data['waveforms'] = waveforms
    data['metadata'] = {}
    data['metadata']['sampling_rate'] = cfg.sampling_rate
    data['metadata']['networks'] = net.networks
    data['metadata']['stations'] = net.stations
    data['metadata']['components'] = net.components
    data['metadata']['date'] = date
    data['metadata']['availability'] = data_availability
    return data

#def load_data(filename,
#              target_starttime,
#              target_endtime,
#              replace_zeros=False):
#    trace = obs.read(filename, format='mseed', check_compression=False)
#    if len(trace) == 0:
#        return
#    trace = trace[0]
#    if trace.stats.sampling_rate != cfg.sampling_rate:
#        print('Warning! The trace has not the expected sampling rate:')
#        print(trace)
#        return
#    # if necessary: fill trace with zeros at
#    # the beginning and at the end
#    trace = trace.trim(starttime=target_starttime,
#                       endtime=target_endtime,
#                       fill_value=0,
#                       pad=True)
#    if replace_zeros:
#        #print('Before {}: {:d} zeros'.format(trace.id, np.sum(trace.data == 0.)))
#        trace.data = replace_zeros_with_white_noise(trace.data) 
#        #print('After {}: {:d} zeros'.format(trace.id, np.sum(trace.data == 0.)))
#    return trace

def load_data(filename, target_starttime, target_endtime,
              remove_response=False, remove_sensitivity=False,
              verbose=True, replace_zeros=False,
              check_compression=False, headonly=False,
              check_sampling_rate=True):
    try:
        trace = obs.read(filename, starttime=target_starttime,
                         endtime=target_endtime,
                         check_compression=check_compression,
                         headonly=headonly)[0]
        if (trace.stats.sampling_rate != cfg.sampling_rate) and check_sampling_rate:
            print('Warning! The trace has not the expected sampling rate:')
            print(trace)
            return
        # if necessary: fill trace with zeros at
        # the beginning and at the end
        trace = trace.trim(starttime=target_starttime,
                           endtime=target_endtime,
                           fill_value=0, pad=True)
        if replace_zeros:
            trace.data = replace_zeros_with_white_noise(trace.data) 
        if remove_response or remove_sensitivity:
            # attach the instrument response for later
            # have the possibility to easily correct it
            # assume that the files are organized in the
            # usual obspyDMT convention
            base = os.path.dirname(os.path.dirname(filename))
            response_file = os.path.join(base, 'resp', 'STXML.'+trace.id)
            inv = obs.read_inventory(response_file)
            trace.attach_response(inv)
        # remove response if requested
        if remove_response:
            # assume that the instrument response
            # is already attached to the trace
            T_max = trace.stats.npts*trace.stats.delta
            T_min = trace.stats.delta
            f_min = 1./T_max
            f_max = 1./(2.*T_min)
            pre_filt = [f_min, 3.*f_min, 0.95*f_max, f_max]
            trace.remove_response(pre_filt=pre_filt, output='VEL')
        elif remove_sensitivity:
            trace.remove_sensitivity()
        return trace
    except Exception as e:
        if verbose:
            print(e)
            print('Error when trying to read {}'.format(filename))
        return

def generate_file_list(data_path,
                       net):
    files = []
    for s in range(len(net.stations)):
        for c in range(len(net.components)):
            #target_files = '{}*{}*{}*'.format(net.networks[s],
            #                                  net.stations[s],
            #                                  net.components[c])
            target_files = '*{}*{}*'.format(net.stations[s],
                                            net.components[c])
            files.extend(glob.glob(os.path.join(data_path,
                                                target_files)))
    return files
