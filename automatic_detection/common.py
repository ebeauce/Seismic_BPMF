import numpy as np
import datetime as dt
from .config import cfg


def to_samples(value, sampling_rate=None):
    if sampling_rate is None:
        sampling_rate = cfg.sampling_rate
    value = np.asarray(np.around(value * sampling_rate), dtype=np.int32)
    return value


def to_seconds(value, sampling_rate=None):
    if sampling_rate is None:
        sampling_rate = cfg.sampling_rate
    value = np.asarray(np.float64(value) / sampling_rate, dtype=np.float64)
    return np.around(value, 2)


def smooth(signal, smooth_win):
    win = np.repeat(1, smooth_win)
    return np.convolve(signal, win, 'same')


def rms(signal):
    return np.sqrt(np.sum(signal ** 2) / signal.size)


def mad(signal):
    return np.median(np.abs(signal - np.median(signal)))


def bandstr(band_list):
   return '{0[0]:.1f}_{0[1]:.1f}'.format(band_list)


def bandlist(band_str):
    return [float(band_str[:3]), float(band_str[-3:])]


def datetime2matlab(dtime):
    mdn = dtime + dt.timedelta(days = 366)
    frac_seconds = (dtime - dt.datetime(dtime.year, dtime.month, dtime.day,
                                        0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dtime.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds

