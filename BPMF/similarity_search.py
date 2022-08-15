import os

from .config import cfg
from . import utils
from . import dataset
from . import clib

import numpy as np
import fast_matched_filter as fmf
import matplotlib.pyplot as plt

from obspy.core import UTCDateTime as udt
from obspy import Stream

from functools import partial
from scipy.stats import kurtosis
import concurrent.futures

from time import time as give_time


class MatchedFilter(object):
    """Class for running a matched filter search and detecting earthquakes."""

    def __init__(
        self,
        template_group,
        min_channels=6,
        max_kurto=100.0,
        remove_edges=True,
        normalize=True,
        max_CC_threshold=0.80,
        n_network_chunks=1,
        threshold_type="rms",
        step=cfg.matched_filter_step,
        max_memory=None,
        max_workers=None,
    ):
        """Instanciate a MatchedFilter object.

        Parameters
        ------------
        template_group: `dataset.TemplateGroup` instance
            The `dataset.TemplateGroup` instance with all the templates to search
            for in the data.
        remove_edges : boolean, default to True
            If True, remove the detections occurring at the beginning and end of the
            data time series. The duration of the edges are set by
            `BPMF.cfg.data_buffer`.
        min_channels : scalar int, default to 6
            Minimum number of channels to consider the CCs valid.
        max_kurto : scalar float, default to 100
            Maximum kurtosis allowed on the CC distribution. Above this threshold,
            it is likely that something went wrong on that day.
            Warning! A higher number of stations will tend to improve the SNR
            in the CC time series, and thus to increase the kurtosis values of
            fine CC distribution. If not sure, set this value to something very large
            (e.g. 10000).
        normalize: boolean, default to True
            If True, normalize the traces by their RMS when given as input to FMF.
            This is recommended to avoid problems with low numbers.
        n_network_chunks: scalar int, default to 1
            Increase this number is your GPU(s) encounters memory issues.
        threshold_type : string, default to 'rms'
            Either 'rms' or 'mad'. Determines whether the
            detection threshold uses the rms or the mad of the correlation
            coefficient time series.
        step: scalar float, default to `cfg.matched_filter_step`
            Step, in seconds, of the matched filter search. That is, time
            interval between two sliding windows.
        max_memory: scalar float, default to None
            If not None, `max_memory` is the maximum memory, in Gb, to not
            exceed during the matched-filter search. The MatchedFilter
            instance will compute the maximum number of templates to use in one
            run to not exceed this memory threshold.
        max_workers: scalar int or None, default to None
            Controls the maximum number of threads created when finding
            detections of new events in the CC time series. If None, use one CPU.
        """
        self.template_group = template_group
        self.min_channels = min_channels
        self.max_kurto = max_kurto
        self.remove_edges = remove_edges
        self.normalize = normalize
        self.max_CC_threshold = max_CC_threshold
        self.n_network_chunks = n_network_chunks
        self.threshold_type = threshold_type.lower()
        self.step_sec = step
        self.max_memory = max_memory
        if max_workers is None:
            max_workers = 1
        self.max_workers = max_workers

    # properties
    @property
    def components(self):
        return self.template_group.components

    @property
    def stations(self):
        return self.template_group.stations

    @property
    def memory_cc_time_series(self):
        if not hasattr(self, "data"):
            return 0.0
        else:
            # 1 float32 = 4 bytes
            nbytes = 4 * int(self.data.duration / self.step_sec)
            nGbytes = nbytes / (1024.0**3)
            return nGbytes

    def set_data(self, data):
        """Attribute `dataset.Data` instance to `self`.

        Parameters
        ------------
        data: `dataset.Data` instance
            The `dataset.Data` instance to read the continuous data to scan.
        """
        self.data = data
        self.data_arr = self.data.get_np_array(
            self.template_group.stations, components=self.template_group.components
        )
        if self.normalize:
            norm = np.std(self.data_arr, axis=-1, keepdims=True)
            norm[norm == 0.0] = 1.0
            self.data_arr /= norm

    def select_cc_indexes(self, cc_t, threshold, search_win):
        """Select the peaks in the CC time series.

        Parameters
        ------------
        cc_t: (n_corr,) numpy.ndarray
            The CC time series for one template.
        threshold: (n_corr,) numpy.ndarray or scalar
            The detection threshold.
        search_win: scalar int
            The minimum inter-event time, in units of correlation step.

        Returns
        --------
        cc_idx: (n_detections,) numpy.ndarray
            The list of all selected CC indexes. They give the timings of the
            detected events.
        """
        sr = self.data.sr
        step = utils.sec_to_samp(self.step_sec, sr=sr)
        cc_detections = cc_t > threshold
        cc_idx = np.where(cc_detections)[0]

        cc_idx = list(cc_idx)
        n_rm = 0
        for i in range(1, len(cc_idx)):
            if (cc_idx[i - n_rm] - cc_idx[i - n_rm - 1]) < search_win:
                if cc_t[cc_idx[i - n_rm]] > cc_t[cc_idx[i - n_rm - 1]]:
                    # keep (i-n_rm)-th detection
                    cc_idx.remove(cc_idx[i - n_rm - 1])
                else:
                    # keep (i-n_rm-1)-th detection
                    cc_idx.remove(cc_idx[i - n_rm])
                n_rm += 1
        cc_idx = np.asarray(cc_idx)

        # go back to regular sampling space
        detection_indexes = cc_idx * step
        if self.remove_edges:
            # remove detections from buffer
            limit = utils.sec_to_samp(cfg.data_buffer, sr=sr)
            idx = detection_indexes >= limit
            cc_idx = cc_idx[idx]
            detection_indexes = detection_indexes[idx]

            limit = utils.sec_to_samp(self.data.duration + cfg.data_buffer, sr=sr)
            idx = detection_indexes < limit
            cc_idx = cc_idx[idx]
        return cc_idx

    def compute_cc_time_series(self, weight_type="simple", device="cpu", tids=None):
        """Compute the CC time series (step 1 of matched-filter search).

        Parameters
        ----------
        weight_type: string, default to 'simple'
            Either of 'simple (default), 'distance', or 'snr'.
        device : string, default to 'cpu'
            Either 'cpu' or 'gpu'.
        tids: list of `tid`, default to None
            If None, run the matched filter search with all templates in
            `self.template_group`. If not None, the matched-filter search is run
            with a limited number of templates (convenient to not overflow the
            device's memory).
        """
        if not hasattr(self, "data"):
            print("You should call `self.set_data(data)` first.")
            return

        if tids is None:
            select_tts = np.arange(self.template_group.n_templates, dtype=np.int32)
        else:
            select_tts = self.template_group.tindexes.loc[tids]

        # parameters
        nt, ns, nc, Nsamp = self.template_group.waveforms_arr[select_tts, ...].shape
        step = utils.sec_to_samp(self.step_sec, sr=self.data.sr)
        n_stations, n_components, n_samples_data = self.data_arr.shape
        if weight_type == "simple":
            # equal weights to all channels
            weights_arr = np.float32(
                self.template_group.network_to_template_map[select_tts, ...]
            )
            norm = np.sum(weights_arr, axis=(1, 2), keepdims=True)
            norm[norm == 0.0] = 1.0
            weights_arr /= norm
            # insufficient data
            invalid = np.sum((weights_arr != 0.0), axis=(1, 2)) < self.min_channels
            weights_arr[invalid] = 0.0
        self.weights_arr = weights_arr
        # ----------------------------------------------
        #   compute the CC time series: run FMF
        CC_SUMS = []
        L = ns // self.n_network_chunks + 1
        for i in range(self.n_network_chunks):
            # to be memory friendly, we subdivide the network into n_network_chunks
            # and the resulting correlation coefficients are then manually stacked
            # in a separate loop
            id1 = i * L
            id2 = (i + 1) * L
            if id2 > ns:
                id2 = ns
            cc_sums = fmf.matched_filter(
                self.template_group.waveforms_arr[select_tts, id1:id2, :, :],
                self.template_group.moveouts_arr[select_tts, id1:id2, :],
                weights_arr[:, id1:id2, :],
                self.data_arr[id1:id2, ...],
                step,
                arch=device,
            )
            CC_SUMS.append(cc_sums)
        cc_sums = CC_SUMS[0]
        for i in range(1, self.n_network_chunks):
            # stack the correlation coefficients
            cc_sums += CC_SUMS[i]

        cc_sums[np.isnan(cc_sums)] = 0.0

        self.cc = {}
        for t, tid in enumerate(self.template_group.tids[select_tts]):
            self.cc[tid] = cc_sums[t, ...]

    def find_detections(
        self,
        minimum_interevent_time,
        threshold_window_dur=600.0,
        sanity_check=True,
        verbose=0,
    ):
        """Analyze the composite network response to find detections.

        Parameters
        -----------
        minimum_interevent_time: scalar, float
            The shortest duration, in seconds, allowed between two
            consecutive detections.
        threshold_window_dur: scalar float, default to 600
            Duration, in seconds, of the sliding window used in the computation
            of the time-dependent detection threshold.
        sanity_check: boolean, default to True
            If True, check that the kurtosis of the CC time series is not above
            `self.max_kurto`. Set to False to speed up the computation.
        verbose: scalar int, default to 0
            If > 0, print some messages.

        Returns
        -----------
        detections: dictionary,
            Dictionary where `detections[tid]` is a list of `dataset.Event` for
            all events detected with template `tid`.
        """

        self.minimum_interevent_time = minimum_interevent_time
        self.threshold_window_dur = threshold_window_dur
        self.sanity_check = sanity_check
        self.white_noise = np.random.normal(size=self.data.n_samples).astype("float32")
        sr = self.data.sr
        detections = {}
        tids = list(self.cc.keys())
        # for t, tid in enumerate(tids):
        #    # t: index in this run's loop
        #    # tt: index in the self.template_group.templates list
        #    # tid: template id
        #    detections_t, tid = self._find_detections_t(t, tid)
        # detections[tid] = detections_t
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(tids), self.max_workers)
        ) as executor:
            output = list(executor.map(self._find_detections_t, range(len(tids)), tids))
        detections.update({output[i][1]: output[i][0] for i in range(len(output))})
        if verbose > 0:
            for tid in tids:
                print(f"Template {tid} detected {len(detections[tid]):d} events.")
        return detections

    def _find_detections_t(self, t, tid):
        """ """
        # t: index in this run's loop
        # tt: index in the self.template_group.templates list
        # tid: template id
        tt = self.template_group.tindexes.loc[tid]

        step = utils.sec_to_samp(self.step_sec, sr=self.data.sr)
        minimum_interevent_time = utils.sec_to_samp(
            self.minimum_interevent_time, sr=self.data.sr
        )

        cc_t = self.cc[tid]
        weights_t = self.weights_arr[t, ...]
        valid = cc_t != 0.0
        if np.sum(valid) == 0:
            # fix the threshold to some value so
            # that there won't be any detections
            threshold = 10.0
        else:
            threshold = time_dependent_threshold(
                cc_t,
                utils.sec_to_samp(self.threshold_window_dur, sr=self.data.sr),
                threshold_type=self.threshold_type,
                white_noise=self.white_noise,
            )
            # saturate threshold as requested by the user
            threshold = np.minimum(self.max_CC_threshold * np.sum(weights_t), threshold)
        if self.sanity_check:
            # ------------------
            # sanity check: the cc time series should approximately
            # be normal, therefore, the kurtosis should be around 0
            # strong deviations from 0 kurtosis arise when some of the
            # data are unavailable during the period of time being processed
            if kurtosis(cc_t) > self.max_kurto:
                # 20 is already a very conservative threshold
                print("Kurtosis too large! Set the CCs to zero. ()")
                cc_t = np.zeros(cc_t.size, dtype=np.float32)
        # select the peaks in the CC time series
        # ----------------------------------------------------------
        # only keep highest correlation coefficient for grouped detections
        # --- different phases correlating with one another can typically
        # --- produce high enough CCs to trigger a detection
        # --- therefore, we use the time difference between the earliest
        # --- phase and the latest phase, on each station, as a proxy for
        # --- the allowed inter-event time
        d_mv = np.max(self.template_group.moveouts_arr[tt, ...], axis=-1) - np.min(
            self.template_group.moveouts_arr[tt, ...], axis=-1
        )
        # take the median across stations
        d_mv = int(np.median(d_mv)) + 1
        search_win = min(
            10 * minimum_interevent_time, max(d_mv, minimum_interevent_time)
        )
        search_win /= step  # time in correlation steps units
        cc_idx = self.select_cc_indexes(cc_t, threshold, search_win)
        detection_indexes = cc_idx * step
        # ----------------------------------------------------------
        n_detections = len(detection_indexes)

        # extract waveforms
        detections_t = []
        data_path, data_filename = os.path.split(self.data.where)
        # give template's attributes to each detection
        template = self.template_group.templates[tt]
        # make sure stations and mv are consistent
        stations = template.moveouts.index.values.astype("U")
        latitude = template.latitude
        longitude = template.longitude
        depth = template.depth
        mv = template.moveouts.values
        phases = template.phases
        for i in range(len(detection_indexes)):
            event = Stream()
            ot_i = self.data.date + detection_indexes[i] / self.data.sr
            event = dataset.Event(
                ot_i,
                mv,
                stations,
                phases,
                data_filename,
                data_path,
                latitude=latitude,
                longitude=longitude,
                depth=depth,
                sampling_rate=self.data.sr,
                data_reader=self.data.data_reader
            )
            aux_data = {}
            aux_data["cc"] = cc_t[cc_idx[i]]
            aux_data["n_threshold"] = cc_t[cc_idx[i]] / threshold[cc_idx[i]]
            aux_data["tid"] = tid
            event.set_aux_data(aux_data)
            detections_t.append(event)
        return detections_t, tid

    def run_matched_filter_search(
        self,
        minimum_interevent_time,
        weight_type="simple",
        device="cpu",
        threshold_window_dur=600.0,
        sanity_check=True,
        verbose=0,
    ):
        """Run the matched-filter search.

        If `self.max_memory` is specified, divide the task into chunks so that
        the device memory is not exceeded.

        Parameters
        ----------
        minimum_interevent_time: scalar float
            The shortest duration, in seconds, allowed between two
            consecutive detections.
        weight_type: string, default to 'simple'
            Either of 'simple (default), 'distance', or 'snr'.
        device : string, default to 'cpu'
            Either 'cpu' or 'gpu'.
        threshold_window_dur: scalar float, default to 600
            Duration, in seconds, of the sliding window used in the computation
            of the time-dependent detection threshold.
        sanity_check: boolean, default to True
            If True, check that the kurtosis of the CC time series is not above
            `self.max_kurto`. Set to False to speed up the computation.
        verbose: scalar int, default to 0
            If > 0, print some messages.

        Returns
        -----------
        detections: dictionary,
            Dictionary where `detections[tid]` is a list of `dataset.Event` for
            all events detected with template `tid`.
        """
        if self.max_memory is not None:
            n_tp_chunk = int(self.max_memory / self.memory_cc_time_series)
        else:
            n_tp_chunk = self.template_group.n_templates
        n_parts = self.template_group.n_templates // n_tp_chunk + 1 * int(
            self.template_group.n_templates % n_tp_chunk > 0
        )
        detections = {}
        duration_fmf = 0.0
        duration_det = 0.0
        for n in range(n_parts):
            tt1 = n * n_tp_chunk
            tt2 = (n + 1) * n_tp_chunk
            if tt2 > self.template_group.n_templates:
                tt2 = self.template_group.n_templates
            tids_chunk = self.template_group.tids[tt1:tt2]
            t1_fmf = give_time()
            self.compute_cc_time_series(
                weight_type=weight_type, device=device, tids=tids_chunk
            )
            t2_fmf = give_time()
            duration_fmf += t2_fmf - t1_fmf
            t1_det = give_time()
            detections_chunk = self.find_detections(
                minimum_interevent_time,
                threshold_window_dur=threshold_window_dur,
                sanity_check=sanity_check,
                verbose=verbose,
            )
            detections.update(detections_chunk)
            t2_det = give_time()
            duration_det += t2_det - t1_det
        if verbose > -1:
            print(f"Total time spent on computing CCs: {duration_fmf:.2f}sec")
            print(f"Total time spent on finding detections: {duration_det:.2f}sec")
        return detections

    # -------------------------------------------
    #       Plotting methods
    # -------------------------------------------
    def plot_cc(self, tid, ax=None, detection=None, figsize=(20, 7)):
        """Plot the composite network response.

        Parameters
        -----------
        tid: string or scalar int
            The id of the template that produces the CCs.
        ax: plt.Axes, default to None
            If not None, plot in this axis.
        detection: dataset.Event, default to None
            A dataset.Event instance of a given detection.
        figsize: tuple, default to (20, 7)
            Width and height of the figure, in inches.

        Returns
        --------
        fig: plt.Figure
            The Figure instance produced by this method.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if ax is None:
            # plot the composite network response
            fig = plt.figure("composite_network_response", figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        if tid not in self.cc.keys():
            print(
                f"{tid} is not in `self.cc.keys()`. Re-run"
                " `self.compute_cc_time_series` properly."
            )
            return

        sr = self.data.sr
        step = utils.sec_to_samp(self.step_sec, sr=sr)
        cc_t = self.cc[tid]
        weights_t = self.weights_arr[self.template_group.tindexes.loc[tid], ...]

        detection_threshold = time_dependent_threshold(
            cc_t,
            utils.sec_to_samp(self.threshold_window_dur, sr=sr),
            threshold_type=self.threshold_type,
        )
        # saturate threshold as requested by the user
        detection_threshold = np.minimum(
            self.max_CC_threshold * np.sum(weights_t), detection_threshold
        )
        cc_idx = self.select_cc_indexes(
            cc_t,
            detection_threshold,
            utils.sec_to_samp(self.minimum_interevent_time, sr=sr) // step,
        )

        time_indexes = np.arange(len(self.data.time), dtype=np.int32)[::step][
            : len(cc_t)
        ]
        ax.plot(self.data.time[time_indexes], cc_t)
        ax.plot(
            self.data.time[time_indexes],
            detection_threshold,
            color="C3",
            ls="--",
            label="Detection Threshold",
        )
        if len(cc_idx) > 0:
            ax.plot(
                self.data.time[cc_idx * step],
                cc_t[cc_idx * step],
                marker="o",
                ls="",
                color="C3",
            )
        ax.legend(loc="upper right")
        ax.set_xlabel("Time of the day")
        ax.set_ylabel("Network CC")

        ax.set_xlim(self.data.time.min(), self.data.time.max())
        # ax.set_ylim(-0.1*(detection_threshold.max() - cnr.min()), 1.2*detection_threshold.max())
        # ax.set_ylim(-0.1*(detection_threshold.max() - cnr.min()), 1.2*detection_threshold.max())

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax.legend(loc="upper right")

        if detection is not None:
            ot = np.datetime64(detection.origin_time)
            ax.annotate(
                "detection",
                (ot, detection.aux_data["cc"]),
                (
                    ot + np.timedelta64(15, "m"),
                    min(ax.get_ylim()[1], 2.0 * detection.aux_data["cc"]),
                ),
                arrowprops={"width": 2, "headwidth": 5, "color": "k"},
            )
        return fig

    def plot_detection(
        self,
        detection,
        figsize=(20, 20),
        component_aliases={"N": ["N", "1"], "E": ["E", "2"], "Z": ["Z"]},
    ):
        """Plot a detection and the composite network response.

        Parameters
        -----------
        detection: dataset.Event
            A dataset.Event instance of a given detection.
        figsize: tuple, default to (20, 20)
            Widht and height of the figure, in inches.
        component_aliases: Dictionary, optional
            Sometimes, components might be named differently than N, E, Z. This
            dictionary tells the function which alternative component names can be
            associated with each "canonical" component. For example,
            `component_aliases['N'] = ['N', '1']` means that the function will also
            check the '1' component in case the 'N' component doesn't exist.

        Returns
        -------
        fig: plt.Figure
            The Figure instance produced by this method.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        tid = detection.aux_data["tid"]

        if tid not in self.cc.keys():
            print(
                f"{tid} is not in `self.cc.keys()`. Re-run"
                " `self.compute_cc_time_series` properly."
            )
            return

        tt = self.template_group.tindexes[tid]
        sr = self.data.sr
        fig = plt.figure(f"detection_{detection.origin_time}", figsize=figsize)
        grid = fig.add_gridspec(
            nrows=len(detection.stations) + 2,
            ncols=len(detection.components),
            hspace=0.35,
        )
        start_times, end_times = [], []
        wav_axes = []
        ax_cc = fig.add_subplot(grid[:2, :])
        self.plot_cc(tid, ax=ax_cc, detection=detection)
        ax_cc.set_ylim(-1.1, 1.1)
        for s, sta in enumerate(detection.stations):
            for c, cp in enumerate(detection.components):
                ax = fig.add_subplot(grid[2 + s, c])
                for cp_alias in component_aliases[cp]:
                    tr = detection.traces.select(station=sta, component=cp_alias)
                    if len(tr) > 0:
                        # succesfully retrieved data
                        break
                if len(tr) == 0:
                    continue
                else:
                    tr = tr[0]
                time = utils.time_range(
                    tr.stats.starttime, tr.stats.endtime, 1.0 / sr, unit="ms"
                )
                start_times.append(time[0])
                end_times.append(time[-1])
                ax.plot(
                    time[: detection.n_samples],
                    utils.max_norm(tr.data[: detection.n_samples]),
                    color="k",
                )
                # plot the template waveforms
                tr_tp = self.template_group.templates[tt].traces.select(
                    station=sta, component=cp_alias
                )
                if len(tr_tp) > 0:
                    tr_tp = tr_tp[0]
                    ax.plot(
                        time[: detection.n_samples],
                        utils.max_norm(tr_tp.data[: detection.n_samples]),
                        color="C3",
                        lw=0.50,
                    )
                # plot the theoretical pick
                phase = detection.aux_data[f"phase_on_comp{cp_alias}"].upper()
                offset_ph = detection.aux_data[f"offset_{phase}"]
                ax.axvline(
                    time[0] + np.timedelta64(int(1000.0 * offset_ph), "ms"), color="C3"
                )
                ax.text(0.05, 0.05, f"{sta}.{cp_alias}", transform=ax.transAxes)
                wav_axes.append(ax)
        for ax in wav_axes:
            ax.set_xlim(min(start_times), max(end_times))
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
            )
        plt.subplots_adjust(top=0.95, bottom=0.06, right=0.98, left=0.06)
        return fig


def time_dependent_threshold(
    time_series, sliding_window, overlap=0.66, threshold_type="rms", white_noise=None
):
    """
    Time dependent detection threshold.

    Parameters
    -----------
    time_series: (n_correlations) array_like
        The array of correlation coefficients calculated by
        FMF (float 32).
    sliding_window: scalar integer
        The size of the sliding window, in samples, used
        to calculate the time dependent central tendancy
        and deviation of the time series.
    overlap: scalar float, default to 0.75
    threshold_type: string, default to 'rms'
        Either rms or mad, depending on which measure
        of deviation you want to use.
    white_noise: `numpy.ndarray` or None, default to None
        If not None, `white_noise` is a vector of random values sampled from the
        standard normal distribution. It is used to fill zeros in the CC time
        series. If None, a random vector is generated from scratch.

    Returns
    ----------
    threshold: (n_correlations) array_like
        Returns the time dependent threshold, with same
        size as the input time series.
    """
    threshold_type = threshold_type.lower()
    n_samples = len(time_series)
    shift = int((1.0 - overlap) * sliding_window)
    last_valid_win = n_samples - sliding_window + 1
    time = np.hstack(
        (np.arange(n_samples, dtype=np.int32)[0:last_valid_win:shift], last_valid_win)
    )
    zeros = time_series == 0.0
    if white_noise is None:
        white_noise = np.random.normal(size=np.sum(zeros)).astype("float32")
    if threshold_type == "rms":
        default_center = time_series[~zeros].mean()
        default_deviation = np.std(time_series[~zeros])
        time_series[zeros] = (
            white_noise[: np.sum(zeros)] * default_deviation + default_center
        )
        time_series_win = np.lib.stride_tricks.sliding_window_view(
            time_series, sliding_window
        )[::shift, :]
        last_win = time_series[time[-1] :]
        center = np.hstack((np.mean(time_series_win, axis=-1), np.mean(last_win)))
        deviation = np.hstack((np.std(time_series_win, axis=-1), np.std(last_win)))
    elif threshold_type == "mad":
        default_center = np.median(time_series[~zeros])
        default_deviation = np.median(np.abs(time_series[~zeros] - default_center))
        time_series[zeros] = (
            white_noise[: np.sum(zeros)] * default_deviation + default_center
        )
        time_series_win = np.lib.stride_tricks.sliding_window_view(
            time_series, sliding_window
        )[::shift, :]
        last_win = time_series[time[-1] :]
        center = np.hstack((np.median(time_series_win, axis=-1), np.median(last_win)))
        deviation = np.hstack(
            (
                np.median(np.abs(time_series_win - center[:-1, np.newaxis]), axis=-1),
                np.median(np.abs(last_win - center[-1]), axis=-1),
            )
        )
    threshold = center + cfg.matched_filter_threshold * deviation
    threshold = np.hstack((threshold[0], threshold, threshold[-1]))
    time = np.hstack((0.0, time, len(time_series)))
    threshold = np.interp(np.arange(n_samples, dtype=np.int32), time, threshold)
    return threshold
