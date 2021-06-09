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
        return np.random.normal(loc=0., scale=1.0, size=data.size)
    std = np.std(data[~zeros])
    data[zeros] = np.random.normal(loc=0., scale=std, size=zeros.sum())
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

def ReadData_per_year(date,
                      net,
                      priority='HH',
                      folder='processed'):
    date = udt(date)
    # ----------------------------
    n_stations = len(net.stations)
    n_components = len(net.components)
    n_samples = int(3600. * 24. * cfg.sampling_rate)
    waveforms = np.zeros((n_stations, n_components, n_samples), dtype=np.float32)
    # -------------------------
    time_looking_for_file = 0.
    time_loading_data = 0.
    time_adjusting_data = 0.
    # -------------------------

    data_availability = np.ones((n_stations, n_components), dtype=np.bool)
    for s in range(n_stations):
        for c in range(n_components):
            filename = os.path.join(cfg.input_path,
                                    str(date.year),
                                    'continuous{:03d}'.format(date.julday),
                                    '{}/{}.{}*{}*'.format(folder, net.networks[s], net.stations[s], net.components[c]))
            t1 = give_time()
            file = glob.glob(filename)
            t2 = give_time()
            time_looking_for_file += (t2-t1)
            if len(file) == 0:
                # no data
                data_availability[s, c] = False
                continue
            t1 = give_time()
            trace = obs.read(file[0])[0]
            if trace.stats.sampling_rate != cfg.sampling_rate:
                print('Warning! The trace has not the expected sampling rate:')
                print(trace)
            if len(file) > 1:
                # there is more than one instrument
                trace = obs.Stream(trace)
                for i in range(1, len(file)):
                    trace += obs.read(file[i])
                print('Several traces were loaded:')
                print(trace)
                print('Priority to {}*'.format(priority))
                trace = trace.select(channel='{}{}'.format(priority, net.components[c]))[0]
                print(trace)
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
            #print('------------------------')
            #print(trace)
            #print(trace.data)
            waveforms[s, c, :] = replace_zeros_with_white_noise(trace.data[:n_samples])
            #print(waveforms[s, c, :])
    # --------------------------
    data = {}
    data['waveforms'] = waveforms
    data['metadata'] = {}
    data['metadata']['sampling_rate'] = cfg.sampling_rate
    data['metadata']['networks'] = net.networks
    data['metadata']['stations'] = net.stations
    data['metadata']['components'] = net.components
    data['metadata']['date'] = date
    data['metadata']['availability'] = data_availability
    return data

def ReadData_per_year_parallel(date,
                      net,
                      priority='HH',
                      folder='processed'):
    date = udt(date)
    # ----------------------------
    n_stations = len(net.stations)
    n_components = len(net.components)
    n_samples = int(3600. * 24. * cfg.sampling_rate)
    waveforms = np.zeros((n_stations, n_components, n_samples), dtype=np.float32)
    # -------------------------

    data_availability = np.ones((n_stations, n_components), dtype=np.bool)
    data_path = os.path.join(cfg.input_path,
                             str(date.year),
                             'continuous{:03d}'.format(date.julday),
                             folder,
                             '')
    t1 = give_time()
    # get all file names
    files = glob.glob(os.path.join(data_path, '*'))
    # read all traces in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(obs.read, f) for f in iter(files)]
        results = [fut.result() for fut in futures]
        traces = obs.Stream([st[0] for st in results])
    t2 = give_time()
    print('{:.2f}s to load the traces.'.format(t2-t1))
    t1 = give_time()
    for s in range(n_stations):
        for c in range(n_components):
            trace = traces.select(network=net.networks[s],
                                  station=net.stations[s],
                                  component=net.components[c])
            if len(trace) == 0:
                # no data
                data_availability[s, c] = False
                continue
            t1 = give_time()
            if trace[0].stats.sampling_rate != cfg.sampling_rate:
                print('Warning! The trace has not the expected sampling rate:')
                print(trace)
            if len(trace) > 1:
                print('Several traces were loaded:')
                print(trace)
                print('Priority to {}*'.format(priority))
                trace = trace.select(channel='{}{}'.format(priority, net.components[c]))[0]
                print(trace)
            else:
                trace = trace[0]
            t2 = give_time()
            time_loading_data += (t2-t1)
            target_endtime = date + 3600.*24. + trace.stats.delta
            trace = trace.slice(starttime=date, endtime=target_endtime)
            if trace.stats.starttime.timestamp != date.timestamp or\
               trace.data.size < n_samples:
                t1 = give_time()
                trace = fill_incomplete_trace(trace, date, target_endtime)
                t2 = give_time()
            waveforms[s, c, :] = replace_zeros_with_white_noise(trace.data[:n_samples])
    t2 = give_time()
    print('{:.2f}s to preprocess the data.'.format(t2-t1))
    # --------------------------
    data = {}
    data['waveforms'] = waveforms
    data['metadata'] = {}
    data['metadata']['sampling_rate'] = cfg.sampling_rate
    data['metadata']['networks'] = net.networks
    data['metadata']['stations'] = net.stations
    data['metadata']['components'] = net.components
    data['metadata']['date'] = date
    data['metadata']['availability'] = data_availability
    return data

def load_data(filename,
              target_starttime,
              target_endtime):
    trace = obs.read(filename)
    if len(trace) == 0:
        return
    trace = trace[0]
    if trace.stats.sampling_rate != cfg.sampling_rate:
        print('Warning! The trace has not the expected sampling rate:')
        print(trace)
        return
    # if necessary: fill trace with zeros at
    # the beginning and at the end
    trace = trace.trim(starttime=target_starttime,
                       endtime=target_endtime,
                       fill_value=0)
    return replace_zeros_with_white_noise(trace.data[:n_samples])

def generate_file_list(data_path,
                       net):
    files = []
    for s in range(len(net.stations)):
        for c in range(len(net.components)):
            target_files = '{}*{}*{}*'.format(net.networks[s],
                                              net.stations[s],
                                              net.components[c])
            files.append(glob.glob(os.path.join(data_path,
                                                target_files)))
