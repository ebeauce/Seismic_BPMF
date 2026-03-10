import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scisig
import warnings

from scipy.interpolate import interp1d
from obspy import Stream


class Spectrum:
    """
    Class for handling spectral data and calculations.$a
    """

    def __init__(self, event=None, frequency_bands=None):
        """
        Parameters
        ----------
        event : str or None, optional
            The event associated with the spectrum. Default is None.
        frequency_bands : list or None, optional
            The frequency bands for the spectrum. Default is None.

        Attributes
        ----------
        event : str or None
            The event associated with the spectrum.
        frequency_bands : list or None
            The frequency bands for the spectrum.
        """
        self.event = event
        if frequency_bands is not None:
            self.set_frequency_bands(frequency_bands)
        self.correction_flags = {}

    def set_Q_model(self, Q, frequencies):
        """
        Set the attenuation Q model for P and S phases.

        Parameters
        ----------
        Q : array-like
            Array of attenuation Q values.
        frequencies : array-like
            Array of corresponding frequencies. These are used to later keep
            track of which frequencies were used for the Q model.

        Returns
        -------
        None
            The Q model is stored in the `Q0` attribute. The frequencies are
            stored in the `Q0_frequencies` attribute. These are later used to
            computed the Q-model and attenuation factor at arbitrary frequencies.

        """
        self.Q0 = np.asarray(Q)
        self.Q0_frequencies = np.asarray(frequencies)
        interpolator = interp1d(
            frequencies, Q, kind="linear", fill_value=(Q[0], Q[-1]), bounds_error=False
        )
        self.Q = np.asarray(interpolator(self.frequencies))

    def update_Q_model(self):
        """
        Interpolate the Q-model at the current `self.frequencies`.
        """
        interpolator = interp1d(
            self.Q0_frequencies,
            self.Q0,
            kind="linear",
            fill_value=(self.Q0[0], self.Q0[-1]),
            bounds_error=False,
        )
        self.Q = interpolator(self.frequencies)

    def update_attenuation_factor(self):
        """
        Compute attenuation factor at the current `self.frequencies`.
        """
        self.update_Q_model()

        for sta in self.event.source_receiver_dist.index:
            r_m = 1000.0 * self.event.source_receiver_dist.loc[sta]
            tt_s = self.event.arrival_times.loc[sta, "S_tt_sec"]
            tt_p = self.event.arrival_times.loc[sta, "P_tt_sec"]
            self.attenuation_factor.loc[sta, f"attenuation_S"] = np.exp(
                np.pi * tt_s * np.asarray(self.frequencies) / self.Q
            )
            self.attenuation_factor.loc[sta, f"attenuation_P"] = np.exp(
                np.pi * tt_s * np.asarray(self.frequencies) / self.Q
            )

    def compute_correction_factor(
        self,
        rho_source,
        rho_receiver,
        vp_source,
        vp_receiver,
        vs_source,
        vs_receiver,
        radiation_S=np.sqrt(2.0 / 5.0),
        radiation_P=np.sqrt(4.0 / 15.0),
    ):
        """
        Compute the correction factor and attenuation factor for a seismic event.

        Parameters
        ----------
        rho_source : float
            Density of the source medium, in kg/m3.
        rho_receiver : float
            Density of the receiver medium, in kg/m3.
        vp_source : float
            P-wave velocity of the source medium, in m/s.
        vp_receiver : float
            P-wave velocity of the receiver medium, in m/s.
        vs_source : float
            S-wave velocity of the source medium, in m/s.
        vs_receiver : float
            S-wave velocity of the receiver medium, in m/s.
        radiation_S : float, optional
            Radiation coefficient for S-wave. Default is sqrt(2/5).
        radiation_P : float, optional
            Radiation coefficient for P-wave. Default is sqrt(4/15).

        Returns
        -------
        None
            The correction factor and attenuation factor are stored in the
            object's attributes `geometrical_factor` and `attenuation_factor`.

        Notes
        -----
        This method requires the object to have an attached `BPMF.dataset.Event`
        instance and for the instance to have called the `set_source_receiver_dist(network)` method.
        """
        from scipy.interpolate import interp1d

        if not hasattr(self, "event"):
            print("Attach the BPMF.dataset.Event instance first.")
            return
        if not hasattr(self.event, "_source_receiver_dist"):
            print("Call event.set_source_receiver_dist(network) first.")
            return
        stations = np.sort(self.event.source_receiver_dist.index)
        geometrical_factor = pd.DataFrame(index=stations)
        attenuation_factor = pd.DataFrame(
            index=stations, columns=["attenuation_P", "attenuation_S"], dtype=object
        )
        if hasattr(self, "Q"):
            self.update_Q_model()
        for sta in stations:
            r_m = 1000.0 * self.event.source_receiver_dist.loc[sta]
            tt_s = self.event.arrival_times.loc[sta, "S_tt_sec"]
            corr_s = (
                4.0
                * np.pi
                * np.sqrt(rho_receiver)
                * np.sqrt(rho_source)
                * np.sqrt(vs_receiver)
                * vs_source ** (5.0 / 2.0)
                * r_m
                / radiation_S
            )
            geometrical_factor.loc[sta, f"geometry_S"] = corr_s
            if hasattr(self, "Q"):
                attenuation_factor.loc[sta, f"attenuation_S"] = np.exp(
                    np.pi * tt_s * np.asarray(self.frequencies) / self.Q
                )
            else:
                attenuation_factor.loc[sta, f"attenuation_S"] = None

            tt_p = self.event.arrival_times.loc[sta, "P_tt_sec"]
            corr_p = (
                4.0
                * np.pi
                * np.sqrt(rho_receiver)
                * np.sqrt(rho_source)
                * np.sqrt(vp_receiver)
                * vp_source ** (5.0 / 2.0)
                * r_m
                / radiation_P
            )
            geometrical_factor.loc[sta, f"geometry_P"] = corr_p
            if hasattr(self, "Q"):
                attenuation_factor.loc[sta, f"attenuation_P"] = np.exp(
                    np.pi * tt_p * np.asarray(self.frequencies) / self.Q
                )
            else:
                attenuation_factor.loc[sta, f"attenuation_P"] = None
        self.geometrical_factor = geometrical_factor
        self.attenuation_factor = attenuation_factor

    def correct_geometrical_spreading(self):
        """Correct all spectra for geometrical spreading."""
        if not hasattr(self, "geometrical_factor"):
            warnings.warn("You need to use compute_correction_factor first.")
            return
        for phase in self.phases:
            if phase == "noise":
                continue
            if not phase in self.correction_flags:
                # initialize flag
                self.correction_flags[phase] = {}
                self.correction_flags[phase][f"geometry_{phase}"] = False
            if self.correction_flags[phase].get(f"geometry_{phase}", False):
                print(
                    f"Geometrical spreading was already corrected for for {phase} spectrum"
                )
                continue
            for trid in getattr(self, f"{phase}_spectrum"):
                sta = trid.split(".")[1]
                _geom_corr = self.geometrical_factor.loc[
                    sta, f"geometry_{phase.upper()}"
                ]
                getattr(self, f"{phase}_spectrum")[trid]["spectrum"] *= _geom_corr
            # update flag
            self.correction_flags[phase][f"geometry_{phase}"] = True

    def correct_attenuation(self):
        """Correct all spectra for attenuation."""
        if not hasattr(self, "attenuation_factor"):
            warnings.warn("You need to use compute_correction_factor first.")
            return
        self.update_Q_model()
        self.update_attenuation_factor()
        for phase in self.phases:
            if phase == "noise":
                continue
            if not phase in self.correction_flags:
                # initialize flag
                self.correction_flags[phase] = {}
                self.correction_flags[phase][f"attenuation_{phase}"] = False
            if self.correction_flags[phase].get(f"attenuation_{phase}", False):
                print(f"Attenuation was already corrected for for {phase} spectrum")
                continue
            for trid in getattr(self, f"{phase}_spectrum"):
                sta = trid.split(".")[1]
                _att_corr = self.attenuation_factor.loc[
                    sta, f"attenuation_{phase.upper()}"
                ]
                getattr(self, f"{phase}_spectrum")[trid]["spectrum"] *= _att_corr
            # update flag
            self.correction_flags[phase][f"attenuation_{phase}"] = True

    def compute_network_average_spectrum(
        self,
        phase,
        snr_threshold,
        average_log=True,
        min_num_valid_channels_per_freq_bin=0,
        max_relative_distance_err_pct=25.0,
        reduce="mean",
        verbose=0,
    ):
        """
        Compute the network average spectrum for a given phase.

        Parameters
        ----------
        phase : str
            Phase of the seismic event. Should be either 'p' or 's'.
        snr_threshold : float
            Signal-to-noise ratio threshold for valid channels.
        average_log : bool, optional
            Flag indicating whether to average the logarithm of the spectra. Default is True.
        min_num_valid_channels_per_freq_bin : int, optional
            Minimum number of valid channels required per frequency bin. Default is 0.
        max_relative_distance_err_pct : float, optional
            Maximum relative distance error percentage for a valid channel. Default is 25.0.
        verbose : int, optional
            Verbosity level. Set to 0 for no output. Default is 0.

        Returns
        -------
        None
            The average spectrum and related information are stored in the object's attributes.

        Notes
        -----
        This method requires the object to have already computed the spectrum for the specified phase
        and to have set the target frequencies using the `set_target_frequencies` method.
        If `correct_propagation` is set to True, the method also requires the object to have computed
        the correction factor using the `compute_correction_factor` method.
        """
        phase = phase.lower()
        assert phase in ["p", "s"], "phase should be 'p' or 's'"
        assert phase in self.phases, f"You need to compute the {phase} spectrum first"
        assert reduce in ["median", "mean"], "reduce should be 'mean' or 'median'"
        assert hasattr(
            self, "frequencies"
        ), "You need to use set_target_frequencies first"
        average_spectrum = np.ma.zeros(len(self.frequencies), dtype=np.float64)
        masked_spectra = []
        signal_spectrum = getattr(self, f"{phase}_spectrum")
        snr_spectrum = getattr(self, f"snr_{phase}_spectrum")
        for trid in signal_spectrum:
            if (
                signal_spectrum[trid]["relative_distance_err_pct"]
                > max_relative_distance_err_pct
            ):
                if verbose > 0:
                    print(
                        f"Source-receiver distance relative error is too high: "
                        f"{signal_spectrum[trid]['relative_distance_err_pct']:.2f}"
                    )
                # the location uncertainty implies too much error
                # on this station, skip it
                continue
            mask = snr_spectrum[trid]["snr"] < snr_threshold
            amplitude_spectrum = signal_spectrum[trid]["spectrum"].copy()
            masked_spectra.append(
                np.ma.masked_array(data=amplitude_spectrum, mask=mask)
            )
        if len(masked_spectra) == 0:
            # there seems to be cases when to spectra were in signal_spectrum??
            if verbose > 0:
                print(f"No spectra found in {phase}_spectrum")
            self.average_spectra = []
            return
        # it looks like we need this explicit definition of the mask
        # otherwise the mask can be converted to a single boolean when
        # all elements are False
        masked_spectra = np.ma.masked_array(
            data=np.stack([arr.data for arr in masked_spectra], axis=0),
            mask=np.stack([arr.mask for arr in masked_spectra], axis=0),
        )
        # count the number of channels that satisfied the SNR criterion
        num_valid_channels = np.sum(~masked_spectra.mask, axis=0)
        # discard the frequency bins for which the minimum number
        # of valid channels was not achieved
        discarded_freq_bins = num_valid_channels < min_num_valid_channels_per_freq_bin
        masked_spectra.mask[:, discarded_freq_bins] = True
        # compute average spectrum without masked elements
        mask = masked_spectra.mask.copy()
        if average_log:
            log10_masked_spectra = np.ma.log10(masked_spectra)
            # another mysterious feature of numpy....
            # need to use exp to propagate mask correctly
            if reduce == "mean":
                average_spectrum = np.exp(
                    np.ma.mean(log10_masked_spectra, axis=0) * np.log(10.0)
                )
            elif reduce == "median":
                average_spectrum = np.exp(
                    np.ma.median(log10_masked_spectra, axis=0) * np.log(10.0)
                )
            std_spectrum = np.ma.std(log10_masked_spectra, axis=0)
        else:
            if reduce == "mean":
                average_spectrum = np.ma.mean(masked_spectra, axis=0)
            elif reduce == "median":
                average_spectrum = np.ma.median(masked_spectra, axis=0)
            std_spectrum = np.ma.std(masked_spectra, axis=0)

        setattr(
            self,
            f"average_{phase}_spectrum",
            {
                "spectrum": average_spectrum,
                "std": std_spectrum,
                "num_valid_channels": num_valid_channels,
                "spectra": masked_spectra,
                "freq": self.frequencies,
                "snr_threshold": snr_threshold,
            },
        )

        if not hasattr(self, "average_spectra"):
            self.average_spectra = [phase]
        else:
            self.average_spectra.append(phase)
            self.average_spectra = list(set(self.average_spectra))

    def compute_multi_band_spectrum(self, traces, phase, buffer_seconds, **kwargs):
        """
        Compute spectrum from the maximum amplitude in multiple 1-octave frequency bands.

        Parameters
        ----------
        traces : list
            List of seismic traces.
        phase : str
            Phase of the seismic event. Should be 'noise', 'p', or 's'.
        buffer_seconds : float
            Buffer duration in seconds to remove from the beginning and end of the trace.
        **kwargs
            Additional keyword arguments for filtering.

        Returns
        -------
        None
            The computed spectrum is stored in the object's attribute `{phase}_spectrum`.

        Notes
        -----
        - The attribute `frequency_bands` is required for this method.
        - The spectrum is computed by finding the maximum amplitude in each 1-octave frequency band.
        - The resulting spectrum and frequency values are stored in the `spectrum` dictionary.
        - The relative distance error percentage is calculated and stored for each trace.
        - The attribute `{phase.lower()}_spectrum` is updated with the computed spectrum.
        - If the attribute `phases` exists, the phase is appended to the list. Otherwise, a new list is created.
        """
        assert phase.lower() in (
            "noise",
            "p",
            "s",
        ), "phase should be 'noise', 'p' or 's'."
        assert hasattr(
            self, "frequency_bands"
        ), "Attribute `frequency_bands` is required for this method."
        num_freqs = len(self.frequency_bands)
        spectrum = {}
        dev_mode = kwargs.get("dev_mode", False)
        for tr in traces:
            nyq = tr.stats.sampling_rate / 2.0
            buffer_samples = int(buffer_seconds * tr.stats.sampling_rate)
            spectrum[tr.id] = {}
            spectrum[tr.id]["spectrum"] = np.zeros(num_freqs, dtype=np.float64)
            spectrum[tr.id]["freq"] = np.zeros(num_freqs, dtype=np.float32)
            if dev_mode:
                spectrum[tr.id]["filtered_traces"] = {}
            for i, band in enumerate(self.frequency_bands):
                bandwidth = (
                    self.frequency_bands[band][1] - self.frequency_bands[band][0]
                )
                center_freq = 0.5 * (
                    self.frequency_bands[band][0] + self.frequency_bands[band][1]
                )
                spectrum[tr.id]["freq"][i] = center_freq
                if self.frequency_bands[band][1] >= nyq:
                    # cannot use this frequency band on this channel
                    continue
                tr_band = tr.copy()
                # preprocess before filtering
                tr_band.detrend("constant")
                tr_band.detrend("linear")
                tr_band.taper(0.25, max_length=buffer_seconds, type="cosine")
                # filter
                tr_band.filter(
                    "bandpass",
                    freqmin=self.frequency_bands[band][0],
                    freqmax=self.frequency_bands[band][1],
                    corners=kwargs.get("corners", 4),
                    zerophase=True,
                )
                trimmed_tr = tr_band.data[buffer_samples:-buffer_samples]
                if dev_mode:
                    tr_band.trim(
                        starttime=tr_band.stats.starttime + buffer_seconds,
                        endtime=tr_band.stats.endtime - buffer_seconds,
                    )
                    spectrum[tr.id]["filtered_traces"][str(band)] = tr_band
                if len(trimmed_tr) == 0:
                    # gap in data?
                    continue
                max_amp = np.max(np.abs(trimmed_tr)) / bandwidth
                spectrum[tr.id]["spectrum"][i] = max_amp
            # print(spectrum[tr.id]["spectrum"])
            max_err = np.sqrt(self.event.hmax_unc**2 + self.event.vmax_unc**2)
            spectrum[tr.id]["relative_distance_err_pct"] = 100.0 * (
                max_err / self.event.source_receiver_dist.loc[tr.stats.station]
            )
        setattr(self, f"{phase.lower()}_spectrum", spectrum)
        if hasattr(self, "phases"):
            self.phases.append(phase)
        else:
            self.phases = [phase]

    def compute_spectrum(self, traces, phase, taper=None, **taper_kwargs):
        """
        Compute spectrum using the Fast Fourier Transform (FFT) on the input traces.

        Parameters
        ----------
        traces : list
            List of seismic traces.
        phase : str
            Phase of the seismic event. Should be 'noise', 'p', or 's'.
        taper : callable or None, optional
            Tapering function to apply to the traces before computing the spectrum. Default is None.
        **taper_kwargs
            Additional keyword arguments for the tapering function.

        Returns
        -------
        None
            The computed spectrum is stored in the object's attribute `{phase}_spectrum`.

        Notes
        -----
        - The spectrum is computed using the FFT on the input traces.
        - The computed spectrum and frequency values are stored in the `spectrum` dictionary.
        - The relative distance error percentage is calculated and stored for each trace.
        - The attribute `{phase.lower()}_spectrum` is updated with the computed spectrum.
        - If the attribute `phases` exists, the phase is appended to the list. Otherwise, a new list is created.
        """
        assert phase.lower() in (
            "noise",
            "p",
            "s",
        ), "phase should be 'noise', 'p' or 's'."
        if taper is None:
            if hasattr(scisig, "windows"):
                taper = scisig.windows.tukey
            else:
                taper = scisig.tukey
            taper_kwargs.setdefault("alpha", 0.05)
        spectrum = {}
        for tr in traces:
            spectrum[tr.id] = {}
            spectrum[tr.id]["spectrum"] = (
                np.fft.rfft(tr.data * taper(tr.stats.npts, **taper_kwargs))
                * tr.stats.delta
            )
            spectrum[tr.id]["freq"] = np.fft.rfftfreq(tr.stats.npts, d=tr.stats.delta)
            max_err = np.sqrt(self.event.hmax_unc**2 + self.event.vmax_unc**2)
            spectrum[tr.id]["relative_distance_err_pct"] = 100.0 * (
                max_err / self.event.source_receiver_dist.loc[tr.stats.station]
            )
        setattr(self, f"{phase.lower()}_spectrum", spectrum)
        if hasattr(self, "phases"):
            self.phases.append(phase)
        else:
            self.phases = [phase]

    def compute_signal_to_noise_ratio(self, phase):
        """
        Compute the signal-to-noise ratio (SNR) for the specified phase.

        Parameters
        ----------
        phase : str
            Phase of the seismic event. Should be 'p' or 's'.

        Returns
        -------
        None
            The computed SNR values are stored in the object's attribute `snr_{phase}_spectrum`.

        Raises
        ------
        AssertionError
            - If `phase` is not 'p' or 's'.
            - If the {phase} spectrum has not been computed.
            - If the noise spectrum has not been computed.

        Notes
        -----
        - The SNR is calculated as the modulus of the signal spectrum divided by the modulus of the noise spectrum.
        - The SNR values and corresponding frequencies are stored in the `snr` dictionary.
        - The attribute `snr_{phase}_spectrum` is updated with the computed SNR values.
        """
        phase = phase.lower()
        assert phase in ["p", "s"], "phase should be 'p' or 's'"
        assert phase in self.phases, f"You need to compute the {phase} spectrum first"
        assert "noise" in self.phases, f"You need to compute the noise spectrum first"
        signal_spectrum = getattr(self, f"{phase}_spectrum")
        noise_spectrum = getattr(self, f"noise_spectrum")
        snr = {}
        for trid in signal_spectrum:
            snr[trid] = {}
            snr_ = np.zeros(len(signal_spectrum[trid]["spectrum"]), dtype=np.float64)
            if trid in noise_spectrum:
                signal = np.abs(signal_spectrum[trid]["spectrum"])
                noise = np.abs(noise_spectrum[trid]["spectrum"])
                zero_by_zero = (signal == 0.0) & (noise == 0.0)
                snr_[~zero_by_zero] = signal[~zero_by_zero] / noise[~zero_by_zero]
            else:
                # no noise spectrum, probably because of gap
                pass
            snr[trid]["snr"] = snr_
            snr[trid]["freq"] = signal_spectrum[trid]["freq"]
        setattr(self, f"snr_{phase}_spectrum", snr)

    def integrate(self, phase, average=True):
        """
        Integrate the spectrum for a specific phase.

        Parameters
        ----------
        phase : str
            Phase name. Should be 'p' or 's'.
        average : bool, optional
            Specifies whether to integrate the average spectrum (if True) or individual spectra (if False).
            Default is True.

        Returns
        -------
        None
            The spectrum is integrated in-place.

        Raises
        ------
        AssertionError
            If the specified phase spectrum has not been computed.
        """
        phase = phase.lower()
        if average:
            assert (
                phase in self.average_spectra
            ), f"You need to compute the average {phase} spectrum first."
            getattr(self, f"average_{phase}_spectrum")["spectrum"] = (
                getattr(self, f"average_{phase}_spectrum")["spectrum"]
                / getattr(self, f"average_{phase}_spectrum")["freq"]
            )
        else:
            assert (
                phase in self.phases
            ), f"You need to compute the {phase} spectrum first."
            spectrum = getattr(self, f"{phase}_spectrum")
            for trid in spectrum:
                spectrum[trid]["spectrum"] /= spectrum[trid]["freq"]

    def differentiate(self, phase, average=True):
        """
        Apply differentiation to the spectrum of the specified phase.

        Parameters
        ----------
        phase : str
            Phase of the seismic event. Should be 'p' or 's'.
        average : bool, optional
            Flag indicating whether to differentiate the average spectrum (True) or individual spectra (False).
            Defaults to True.

        Returns
        -------
        None
            The spectrum is modified in-place by multiplying it with the corresponding frequency values.

        Raises
        ------
        AssertionError
            - If `average` is True and the average {phase} spectrum has not been computed.
            - If `average` is False and the {phase} spectrum has not been computed.
        """
        phase = phase.lower()
        if average:
            assert (
                phase in self.average_spectra
            ), f"You need to compute the average {phase} spectrum first."
            getattr(self, f"average_{phase}_spectrum")["spectrum"] = (
                getattr(self, f"average_{phase}_spectrum")["spectrum"]
                * getattr(self, f"average_{phase}_spectrum")["freq"]
            )
        else:
            assert (
                phase in self.phases
            ), f"You need to compute the {phase} spectrum first."
            spectrum = getattr(self, f"{phase}_spectrum")
            for trid in spectrum:
                spectrum[trid]["spectrum"] *= spectrum[trid]["freq"]

    def fit_average_spectrum(
        self,
        phase,
        model="brune",
        log=True,
        min_fraction_valid_points_below_fc=0.10,
        min_fraction_valid_points=0.50,
        weighted=False,
        **kwargs,
    ):
        """
        Fit the average displacement spectrum with a specified model.

        Parameters
        ----------
        phase : str
            Phase of the seismic event. Should be 'p' or 's'.
        model : str, optional
            Model to use for fitting the spectrum. Default is 'brune'.
        log : bool, optional
            Flag indicating whether to fit the logarithm of the spectrum. Default is True.
        min_fraction_valid_points_below_fc : float, optional
            Minimum fraction of valid points required below the corner frequency. Default is 0.10.
        min_fraction_valid_points : float, optional
            Minimum fraction of valid points required overall. Default is 0.50.
        weighted : bool, optional
            Flag indicating whether to apply weighted fitting using sigmoid weights. Default is False.
        **kwargs
            Additional keyword arguments to be passed to the curve fitting function.

        Returns
        -------
        None
            The fitting results are stored as attributes of the object.

        Raises
        ------
        AssertionError
            If the average {phase} spectrum has not been computed.

        Notes
        -----
        - If the spectrum is below the SNR threshold everywhere, the fitting cannot be performed.
        - If there are not enough valid points or valid points below the corner frequency,
          the fitting is considered unsuccessful.
        """
        from scipy.optimize import curve_fit
        from functools import partial

        phase = phase.lower()
        assert (
            phase in self.average_spectra
        ), f"You need to compute the average {phase} spectrum first."
        spectrum = getattr(self, f"average_{phase}_spectrum")
        if np.sum(~spectrum["spectrum"].mask) == 0:
            print("Spectrum is below SNR threshold everywhere, cannot fit it.")
            self.inversion_success = False
            return
        valid_fraction = np.sum(~spectrum["spectrum"].mask) / float(
            len(spectrum["spectrum"])
        )
        if valid_fraction < min_fraction_valid_points:
            print(f"Not enough valid points! (Only {100.*valid_fraction:.2f}%)")
            self.inversion_success = False
            return
        omega0_first_guess = spectrum["spectrum"].data[~spectrum["spectrum"].mask][0]
        fc_first_guess = fc_circular_crack(moment_to_magnitude(omega0_first_guess))
        standardized_num_valid_channels = (
            spectrum["num_valid_channels"] - spectrum["num_valid_channels"].mean()
        ) / spectrum["num_valid_channels"].mean()
        sigmoid_weights = 1.0 / (1.0 + np.exp(-standardized_num_valid_channels))
        if model == "brune":
            mod = brune
        elif model == "boatwright":
            mod = boatwright
        if log:
            obs = np.log10(spectrum["spectrum"].data)
        else:
            obs = spectrum["spectrum"].data
        # misfit = (
        #        lambda f, omega0, fc:
        #        weights * (obs - mod(f, omega0, fc, log=log))
        #        )
        mod = partial(mod, log=log)
        y = obs[~spectrum["spectrum"].mask]
        x = spectrum["freq"][~spectrum["spectrum"].mask]
        if weighted:
            inverse_weights = 1.0 / sigmoid_weights[~spectrum["spectrum"].mask]
        else:
            inverse_weights = None
        p0 = np.array([omega0_first_guess, fc_first_guess])
        bounds = (np.array([0.0, 0.0]), np.array([np.inf, 1.0e3 * fc_first_guess]))
        try:
            popt, pcov = curve_fit(
                mod, x, y, p0=p0, bounds=bounds, sigma=inverse_weights, **kwargs
            )
            self.inversion_success = True
        except (RuntimeError, ValueError):
            print("Inversion (scipy.optimize.cuve_fit) failed.")
            self.inversion_success = False
            return
        # check whether the low-frequency plateau was well constrained
        npts_valid_below_fc = np.sum(x < popt[1])
        fraction_valid_points_below_fc = float(npts_valid_below_fc) / float(
            len(spectrum["freq"])
        )
        # print(f"Fraction of valid points below fc is: {fraction_valid_points_below_fc:.2f}")
        if fraction_valid_points_below_fc < min_fraction_valid_points_below_fc:
            self.inversion_success = False
            print(
                "Not enough valid points below corner frequency "
                f"(only {100.*fraction_valid_points_below_fc:.1f}%)"
            )
            return
        perr = np.sqrt(np.diag(pcov))
        self.M0 = popt[0]
        self.fc = popt[1]
        self.Mw = moment_to_magnitude(self.M0)
        self.M0_err = perr[0]
        self.fc_err = perr[1]
        self.model = model

    def resample(self, new_frequencies, phase):
        """
        Resample the spectrum to new frequencies.

        Parameters
        ----------
        new_frequencies : array-like
            New frequency values to resample the spectrum to.
        phase : str or list
            Phase(s) of the seismic event to resample. Can be a single phase or a list of phases.

        Returns
        -------
        None
            The spectrum is resampled and updated in-place.
        """
        if isinstance(phase, str):
            phase = [phase]
        for ph in phase:
            ph = ph.lower()
            if not hasattr(self, f"{ph}_spectrum"):
                print(f"Attribute {ph}_spectrum does not exist.")
                continue
            spectrum = getattr(self, f"{ph}_spectrum")
            for trid in spectrum:
                spectrum[trid]["spectrum"] = np.interp(
                    new_frequencies,
                    spectrum[trid]["freq"],
                    np.abs(spectrum[trid]["spectrum"]),
                )
                # set to zero any frequency bins that were extrapolated
                # need the 0.99 in case new_frequencies have some rounding errors
                outside_bandwidth = new_frequencies >= 0.99 * np.max(
                    spectrum[trid]["freq"]
                )
                spectrum[trid]["spectrum"][outside_bandwidth] = 0.0
                spectrum[trid]["freq"] = new_frequencies

    def set_frequency_bands(self, frequency_bands):
        """
        Set the frequency bands for spectrum analysis.

        Parameters
        ----------
        frequency_bands : dict
            Dictionary specifying the frequency bands.
            The keys are the names of the bands, and the values are tuples of the form (freqmin, freqmax).
            freqmin and freqmax represent the minimum and maximum frequencies of each band, respectively.

        Returns
        -------
        None
            The frequency bands are set and stored as an attribute in the object.
            The center frequencies of each band are also calculated and stored in the `frequencies` attribute.
        """
        self.frequency_bands = frequency_bands
        self.frequencies = np.zeros(len(self.frequency_bands), dtype=np.float32)
        for i, band in enumerate(self.frequency_bands):
            self.frequencies[i] = 0.5 * (
                self.frequency_bands[band][0] + self.frequency_bands[band][1]
            )
        # make sure this is sorted!!
        freq_bands = list(self.frequency_bands.keys())
        sorted_idx = np.argsort(self.frequencies)
        self.frequencies = self.frequencies[sorted_idx]
        self.frequency_bands = {
            freq_bands[i]: self.frequency_bands[freq_bands[i]] for i in sorted_idx
        }

    def set_target_frequencies(self, freq_min, freq_max, num_points):
        """
        Set the target frequencies for spectrum analysis.

        Parameters
        ----------
        freq_min : float
            Minimum frequency.
        freq_max : float
            Maximum frequency.
        num_points : int
            Number of target frequencies to generate.

        Returns
        -------
        None
            The target frequencies are set and stored as an attribute in the object.
        """
        self.frequencies = np.logspace(
            np.log10(freq_min), np.log10(freq_max), num_points
        )

    def _spectra_pd(self, phase):
        _spec = getattr(self, f"{phase}_spectrum")
        return pd.DataFrame(
            columns=list(_spec.keys()),
            index=self.frequencies,
            data=np.stack(
                [_spec[trid]["spectrum"] for trid in _spec],
                axis=1,
            ),
        )

    def _snr_spectra_pd(self, phase):
        _spec = getattr(self, f"snr_{phase}_spectrum")
        return pd.DataFrame(
            columns=list(_spec.keys()),
            index=self.frequencies,
            data=np.stack(
                [_spec[trid]["snr"] for trid in _spec],
                axis=1,
            ),
        )

    def plot_average_spectrum(
        self,
        phase,
        figname="spectrum",
        figtitle="",
        figsize=(10, 10),
        colors={"noise": "dimgrey", "s": "black", "p": "C3"},
        linestyle={"noise": "--", "s": "-", "p": "-"},
        plot_fit=False,
        plot_std=False,
        plot_num_valid_channels=False,
    ):
        """
        Plot the average spectrum for a given phase.

        Parameters
        ----------
        phase : str or list of str
            The phase(s) for which to plot the average spectrum.
        figname : str, optional
            The name of the figure. Default is "spectrum".
        figtitle : str, optional
            The title of the figure. Default is an empty string.
        figsize : tuple, optional
            The size of the figure in inches. Default is (10, 10).
        colors : dict, optional
            A dictionary specifying the colors for different phases.
            Default is {"noise": "dimgrey", "s": "black", "p": "C3"}.
        linestyle : dict, optional
            A dictionary specifying the line styles for different phases.
            Default is {"noise": "--", "s": "-", "p": "-"}.
        plot_fit : bool, optional
            Whether to plot the fitted model spectrum. Default is False.
        plot_std : bool, optional
            Whether to plot the standard deviation range of the average spectrum. Default is False.
        plot_num_valid_channels : bool, optional
            Whether to plot the number of channels above the signal-to-noise ratio (SNR) threshold. Default is False.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import fnmatch
        from matplotlib.ticker import MaxNLocator

        if isinstance(phase, str):
            phase = [phase]
        fig, ax = plt.subplots(num=figname, figsize=figsize)
        ax.set_title(figtitle)
        for ph in phase:
            ph = ph.lower()
            if not hasattr(self, f"average_{ph}_spectrum"):
                print(f"Attribute average_{ph}_spectrum does not exist.")
                continue
            spectrum = getattr(self, f"average_{ph}_spectrum")
            fft = spectrum["spectrum"]
            freq = spectrum["freq"]
            amplitude_spec = np.abs(fft)
            ax.plot(
                freq,
                amplitude_spec,
                color=colors[ph],
                ls=linestyle[ph],
                label=f"Average {ph} spectrum",
            )
            if plot_std:
                lower_amp = 10.0 ** (np.log10(amplitude_spec) - spectrum["std"])
                upper_amp = 10.0 ** (np.log10(amplitude_spec) + spectrum["std"])
                ax.fill_between(
                    freq, lower_amp, upper_amp, color=colors[ph], alpha=0.33
                )
            if plot_num_valid_channels:
                axb = ax.twinx()
                axb.plot(
                    freq,
                    spectrum["num_valid_channels"],
                    marker="o",
                    ls="",
                    color=colors[ph],
                    label="Number of channels above SNR threshold",
                )
                axb.set_ylabel("Number of channels above SNR threshold")
                ylim = axb.get_ylim()
                axb.set_ylim(max(0, ylim[0] - 1), ylim[1] + 1)
                axb.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_zorder(axb.get_zorder() + 1)
                ax.patch.set_visible(False)
                axb.legend(loc="upper right")
        if plot_fit and hasattr(self, "M0"):
            if self.model == "brune":
                model = brune
            elif self.model == "boatwright":
                model = boatwright
            fit = model(freq, self.M0, self.fc)
            label = (
                f"{self.model.capitalize()} model:\n"
                r"$M_w=$"
                f"{self.Mw:.1f}, "
                r"$f_c$="
                f"{self.fc:.2f}Hz"
            )
            ax.plot(freq, fit, color=colors[ph], ls="--", label=label)
        ax.legend(loc="lower left")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude spectrum ([input units/Hz])")
        ax.loglog()
        return fig

    def plot_spectrum(
        self,
        phase,
        station=None,
        component=None,
        figname="spectrum",
        figsize=(10, 10),
        correct_propagation=False,
        plot_snr=False,
        colors={"noise": "dimgrey", "s": "black", "p": "C3"},
        linestyle={"noise": "--", "s": "-", "p": "-"},
    ):
        """
        Plot the spectrum for the specified phase(s) and trace(s).

        Parameters
        ----------
        phase : str or list of str
            The phase(s) for which to plot the spectrum.
        station : str or None, optional
            The station code. If None, all stations will be plotted. Default is None.
        component : str or None, optional
            The component code. If None, all components will be plotted. Default is None.
        figname : str, optional
            The name of the figure. Default is "spectrum".
        figsize : tuple, optional
            The size of the figure in inches. Default is (10, 10).
        correct_propagation : bool, optional
            Whether to correct the spectrum for propagation effects. Default is False.
        plot_snr : bool, optional
            Whether to plot the signal-to-noise ratio (SNR) spectrum if available. Default is False.
        colors : dict, optional
            A dictionary specifying the colors for different phases. Default is {"noise": "dimgrey", "s": "black", "p": "C3"}.
        linestyle : dict, optional
            A dictionary specifying the line styles for different phases. Default is {"noise": "--", "s": "-", "p": "-"}.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import fnmatch

        if isinstance(phase, str):
            phase = [phase]
        fig, ax = plt.subplots(num=figname, figsize=figsize)
        for ph in phase:
            ph = ph.lower()
            if not hasattr(self, f"{ph}_spectrum"):
                print(f"Attribute {ph}_spectrum does not exist.")
                continue
            spectrum = getattr(self, f"{ph}_spectrum")
            trace_ids = list(spectrum.keys())
            if station is None:
                station = "*"
            if component is None:
                component = "*"
            target_tr_id = f"*.{station}.*.*{component}"
            selected_ids = fnmatch.filter(trace_ids, target_tr_id)
            for trid in selected_ids:
                fft = spectrum[trid]["spectrum"]
                freq = spectrum[trid]["freq"]
                amplitude_spec = np.abs(fft)
                if correct_propagation and ph in ["p", "s"]:
                    # sta = trid.split(".")[1]
                    amplitude_spec *= self.geometrical_factor.loc[
                        sta, f"geometry_{ph.upper()}"
                    ]
                ax.plot(
                    freq,
                    amplitude_spec,
                    color=colors[ph],
                    ls=linestyle[ph],
                    label=f"{ph} spectrum: {trid}",
                )
                if plot_snr:
                    if not hasattr(self, f"snr_{ph}_spectrum"):
                        # print(f"Attribute snr_{ph}_spectrum does not exist.")
                        continue
                    snr_spectrum = getattr(self, f"snr_{ph}_spectrum")
                    ax.plot(
                        snr_spectrum[trid]["freq"],
                        snr_spectrum[trid]["snr"],
                        color=colors[ph],
                        ls=linestyle["noise"],
                        label=f"{ph} snr: {trid}",
                    )
        plt.subplots_adjust(right=0.85, bottom=0.20)
        ax.legend(bbox_to_anchor=(1.01, 1.00), loc="upper left", handlelength=0.9)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude spectrum ([input units/Hz])")
        ax.loglog()
        return fig


# classic models of earthquake displacement far-field spectrum


def brune(freqs, omega0, fc, log=False):
    """Brune model."""
    if log:
        return np.log10(omega0) - np.log10(1.0 + (freqs / fc) ** 2)
    else:
        return omega0 / (1.0 + (freqs / fc) ** 2)


def boatwright(freqs, omega0, fc, log=False):
    """Boatwright model."""
    if log:
        return np.log10(omega0) - 0.5 * np.log10(1.0 + (freqs / fc) ** 4)
    else:
        return omega0 / np.sqrt(1.0 + (freqs / fc) ** 4)


def magnitude_to_moment(Mw):
    """Convert moment magnitude to seismic moment [N.m]."""
    return 10.0 ** (3.0 / 2.0 * Mw + 9.1)


def moment_to_magnitude(M0):
    """Convert seismic moment [N.m] to moment magnitude."""
    return 2.0 / 3.0 * (np.log10(M0) - 9.1)


def fc_circular_crack(
    Mw, stress_drop_Pa=1.0e6, phase="p", vs_m_per_s=3500.0, vr_vs_ratio=0.9
):
    """
    Compute the corner frequency assuming a circular crack model (Eshelby).

    Parameters
    ----------
    Mw : float
        Moment magnitude of the earthquake.
    stress_drop_Pa : float, optional
        Stress drop in Pascals. Default is 1.0e6.
    phase : str, optional
        Seismic phase. Valid values are 'p' for P-wave and 's' for S-wave.
        Default is 'p'.
    vs_m_per_s : float, optional
        Shear wave velocity in meters per second. Default is 3500.0.
    vr_vs_ratio : float, optional
        Ratio of rupture velocity to shear wave velocity. Default is 0.9.

    Returns
    -------
    corner_frequency : float
        Corner frequency in Hertz.

    Raises
    ------
    AssertionError
        If phase is not 'p' or 's'.
    """
    phase = phase.lower()
    assert phase in ["p", "s"], "phase should 'p' or 's'."
    M0 = magnitude_to_moment(Mw)
    crack_radius = np.power((7.0 / 16.0) * (M0 / stress_drop_Pa), 1.0 / 3.0)
    if phase == "p":
        constant = 2.23
    elif phase == "s":
        constant = 1.47
    vr = vr_vs_ratio * vs_m_per_s
    corner_frequency = (constant * vr) / (2.0 * np.pi * crack_radius)
    return corner_frequency


def stress_drop_circular_crack(Mw, fc, phase="p", vs_m_per_s=3500.0, vr_vs_ratio=0.9):
    """
    Compute the stress drop assuming a circular crack model (Eshelby).

    Parameters
    ----------
    Mw : float
        Moment magnitude of the earthquake.
    fc : float
        Corner frequency in Hertz.
    phase : str, optional
        Seismic phase. Valid values are 'p' for P-wave and 's' for S-wave.
        Default is 'p'.
    vs_m_per_s : float, optional
        Shear wave velocity in meters per second. Default is 3500.0.
    vr_vs_ratio : float, optional
        Ratio of rupture velocity to shear wave velocity. Default is 0.9.

    Returns
    -------
    stress_drop : float
        Stress drop in Pascals.

    Raises
    ------
    AssertionError
        If phase is not 'p' or 's'.
    """
    phase = phase.lower()
    assert phase in ["p", "s"], "phase should 'p' or 's'."
    M0 = magnitude_to_moment(Mw)
    if phase == "p":
        constant = 2.23
    elif phase == "s":
        constant = 1.47
    vr = vr_vs_ratio * vs_m_per_s
    crack_radius = constant * vr / (2.0 * np.pi * fc)
    stress_drop = 7.0 / 16.0 * M0 / crack_radius**3
    return stress_drop


def _snr_based_weights(snr, snr_threshold, weight_max=3.0, max_num_bad_measurements=6):
    # allow some numerical noise (?)
    snr_clipped = np.minimum(snr, 1.001 * snr_threshold)
    # linear function of snr
    weights = snr_clipped
    # clip weights
    weights = np.minimum(weights, weight_max)
    # print("Before", weights)
    if np.sum(snr >= snr_threshold) >= max_num_bad_measurements:
        # set weights of bad measurements to 0
        weights[snr < snr_threshold] = 0.0
    else:
        ordered_indexes = np.argsort(snr)
        # set to 0 all but the `max_num_bad_measurements` least bad meas.
        weights[ordered_indexes[:-max_num_bad_measurements]] = 0.0
    # print("After", weights)
    return weights


def approximate_moment_magnitude(
    spectrum,
    snr_threshold=10.0,
    num_averaging_bands=1,
    low_snr_freq_min_hz=2.0,
    magnitude_log_moment_scaling=2.0 / 3.0,
    phases=None,
    snr_based_weights=_snr_based_weights,
):
    if phases is None:
        phases = spectrum.phases

    corr_disp_spectra = {}
    snr_spectra = {}
    approx_moment = {}
    for ph in phases:
        # 1) store multi-band, propagation-corrected displacement
        # spectra in a pandas.DataFrame
        corr_disp_spectra[ph] = spectrum._spectra_pd(ph)
        # 2) store snr in a pandas.DataFrame
        snr_spectra[ph] = spectrum._snr_spectra_pd(ph)
        # 3) select peaks from best lowest frequency band
        #    to calculate the approximate moment magnitude
        geo_corrected_peaks = pd.Series(
            index=list(corr_disp_spectra[ph].columns), dtype=np.float64
        )
        num_cha = len(corr_disp_spectra[ph].columns)
        _peak_snr = np.zeros(num_cha, dtype=np.float32)

        for j, idx in enumerate(geo_corrected_peaks.index):
            # process station by station
            station = idx.split(".")[1]
            # fetch relevant disp spectra and snr
            multi_band_peaks = corr_disp_spectra[ph].loc[:, idx]
            multi_band_snr = snr_spectra[ph].loc[:, idx]
            # find frequency bands that satisfy snr criterion
            valid_bands = multi_band_snr.loc[multi_band_snr > snr_threshold].index
            if len(valid_bands) > 0:
                # peak amplitude is taken from the lowest-frequency,
                # valid frequency band (reflect physical seismic moment)
                valid_bands = np.sort(valid_bands)
                selected_bands = valid_bands[
                    : min(len(valid_bands), num_averaging_bands)
                ]
                if len(selected_bands) > 1:
                    geo_corrected_peaks.loc[idx] = np.median(
                        multi_band_peaks.loc[selected_bands].values
                    )
                else:
                    geo_corrected_peaks.loc[idx] = multi_band_peaks.loc[
                        selected_bands
                    ].values[0]
                if np.isnan(geo_corrected_peaks.loc[idx]):
                    breakpoint()
                _peak_snr[j] = snr_threshold
            else:
                # peak amplitude is taken from the highest snr frequency
                # band (implies error on magnitude estimation)
                high_freq = multi_band_snr.index > low_snr_freq_min_hz
                freq_idx = multi_band_snr[high_freq].index[
                    multi_band_snr[high_freq].argmax()
                ]

                w_ = multi_band_snr.loc[high_freq]
                p_ = multi_band_peaks.loc[high_freq]
                sum_ = w_.sum()
                sum_ = 1.0 if sum_ == 0.0 else sum_
                geo_corrected_peaks.loc[idx] = (w_ * p_).sum() / sum_
                _peak_snr[j] = (w_ * multi_band_snr.loc[high_freq]).sum() / sum_

        _peak_snr[geo_corrected_peaks == 0.0] = 0.0
        weights = snr_based_weights(_peak_snr, snr_threshold)

        weighted_log_peaks = np.zeros(len(weights), dtype=np.float64)
        weighted_log_peaks[weights > 0.0] = (
            np.log10(geo_corrected_peaks[weights > 0.0]) * weights[weights > 0.0]
        )
        estimated_log10_M0 = weighted_log_peaks.sum() / weights.sum()
        Mw_approx = magnitude_log_moment_scaling * (estimated_log10_M0 - 9.1)
        print(
            f"{ph.upper()}-wave: Approx. Mw: {Mw_approx:.2f} (approx. log10 M0: {estimated_log10_M0:.2f})"
        )
        approx_moment[ph] = Mw_approx

    return approx_moment


# workflow function
def extract_windows(
    event,
    duration_sec,
    offset_ot_sec_noise,
    data_folder,
    attach_response=True,
    time_shifted=False,
    phase_on_comp_p={"N": "P", "1": "P", "E": "P", "2": "P", "Z": "P"},
    phase_on_comp_s={"N": "S", "1": "S", "E": "S", "2": "S", "Z": "S"},
    offset_phase={"P": 0.5, "S": 0.5},
    cleanup_stream=None,
):
    #                 extract waveforms
    # first, read short extract before signal as an estimate of noise
    event.read_waveforms(
        duration_sec,
        time_shifted=False,
        data_folder=data_folder,
        offset_ot=offset_ot_sec_noise,
        attach_response=attach_response,
    )
    if cleanup_stream is not None:
        cleanup_stream(event.traces)
    noise = event.traces.copy()

    # then, read signal
    event.read_waveforms(
        duration_sec,
        phase_on_comp=phase_on_comp_p,
        offset_phase=offset_phase,
        time_shifted=time_shifted,
        data_folder=data_folder,
        attach_response=attach_response,
    )
    if cleanup_stream is not None:
        cleanup_stream(event.traces)
    p_wave = event.traces.copy()

    event.read_waveforms(
        duration_sec,
        phase_on_comp=phase_on_comp_s,
        offset_phase=offset_phase,
        time_shifted=time_shifted,
        data_folder=data_folder,
        attach_response=attach_response,
    )
    if cleanup_stream is not None:
        cleanup_stream(event.traces)
    s_wave = event.traces.copy()

    # correct for instrument response and integrate to get displacement seismograms
    for st in [noise, p_wave, s_wave]:
        for tr in st:
            fnyq = tr.stats.sampling_rate / 2.0
            pre_filt = [
                1.0 / duration_sec,
                1.05 / duration_sec,
                0.95 * fnyq,
                0.98 * fnyq,
            ]
            tr.detrend("constant")
            tr.detrend("linear")
            tr.taper(0.25, type="cosine")
            tr.remove_response(
                pre_filt=pre_filt,
                zero_mean=False,
                taper=False,
                # taper_fraction=0.25,
                output="DISP",
                plot=False,
            )

    windows = {"noise": noise, "p": p_wave, "s": s_wave}
    return windows


def compute_moment_magnitude(
    event,
    windows,
    method="regular",
    phases=None,
    freq_min_hz=None,
    freq_max_hz=None,
    num_freqs=None,
    frequency_bands=None,
    window_buffer_sec=None,
    snr_threshold=10.0,
    min_num_valid_channels_per_freq_bin=3,
    max_relative_distance_err_pct=33.0,
    medium_properties={
        "Q_1Hz": None,
        "attenuation_n": None,
        "rho_source_kgm3": None,
        "vp_source_ms": None,
        "vs_source_ms": None,
        "rho_receiver_kgm3": None,
        "vp_receiver_ms": None,
        "vs_receiver_ms": None,
    },
    approximate_moment_magnitude_args={
        "num_averaging_bands": 3,
        "low_snr_freq_min_hz": 2.0,
        "magnitude_log_moment_scaling": 2.0 / 3.0,
    },
    qc=True,
    full_output=False,
    spectral_model="brune",
    min_fraction_valid_points_below_fc=0.20,
    num_channel_weighted_fit=True,
    max_rel_m0_err_pct=33.0,
    max_rel_fc_err_pct=33.0,
    stress_drop_mpa_min=1.0e-3,
    stress_drop_mpa_max=1.0e4,
    plot_above_mw=100.0,
    plot_above_random=1.0,
    plot_spectrum=False,
    figsize=(8, 8),
):
    """ """
    # initialize spectrum instance
    spectrum = Spectrum(event=event)
    if phases is None:
        phases = list(windows.keys())

    # -----------------------------------------------------------
    #           Compute displacement spectra
    # -----------------------------------------------------------
    if method == "regular":
        for ph in phases:
            spectrum.compute_spectrum(windows[ph], ph, alpha=0.15)
        spectrum.set_target_frequencies(freq_min_hz, freq_max_hz, num_freqs)
        spectrum.resample(spectrum.frequencies, spectrum.phases)
    elif method == "multiband":
        spectrum.set_frequency_bands(frequency_bands)
        for ph in phases:
            spectrum.compute_multi_band_spectrum(windows[ph], ph, window_buffer_sec)
        # does the following resampling even make sense?
        spectrum.set_target_frequencies(
                spectrum.frequencies.min(),
                spectrum.frequencies.max(),
                num_freqs
                )
        spectrum.resample(spectrum.frequencies, spectrum.phases)

    for ph in phases:
        if ph == "noise":
            continue
        spectrum.compute_signal_to_noise_ratio(ph)

    # -----------------------------------------------------------
    #        compute propagation effects and corrections
    # -----------------------------------------------------------
    Q = medium_properties["Q_1HZ"] * np.power(
        spectrum.frequencies, medium_properties["attenuation_n"]
    )
    spectrum.set_Q_model(Q, spectrum.frequencies)
    spectrum.compute_correction_factor(
        medium_properties["rho_source_kgm3"],
        medium_properties["rho_receiver_kgm3"],
        medium_properties["vp_source_ms"],
        medium_properties["vp_receiver_ms"],
        medium_properties["vs_source_ms"],
        medium_properties["vs_receiver_ms"],
    )

    # -----------------------------------------------------------
    #              Correct for propagation effects
    # -----------------------------------------------------------
    spectrum.correct_geometrical_spreading()
    spectrum.correct_attenuation()

    # -----------------------------------------------------------
    #                  Initialize output
    # -----------------------------------------------------------
    phases = list(phases)
    phases.remove("noise")
    success = False
    source_parameters = {}
    if plot_spectrum:
        figs = []
    for ph in phases:
        source_parameters[ph] = {}
        if len(getattr(spectrum, f"{ph}_spectrum")) == 0:
            print(f"Could not compute a single {ph}-wave spectrum!")
            source_parameters[ph]["Mw*"] = np.nan
            source_parameters[ph]["Mw"] = np.nan
            source_parameters[ph]["Mw_err"] = np.nan
        else:
            success = True
    if not success:
        # failed for all requested phases
        output = (event, spectrum, source_parameters)
        if full_output:
            output = output + (
                pd.DataFrame(),
                pd.DataFrame(),
            )
        if plot_spectrum:
            output = output + (figs,)
        return output

    # -----------------------------------------------------------
    #       Compute network-averaged displacement spectrum
    # -----------------------------------------------------------
    for ph in phases:
        spectrum.compute_network_average_spectrum(
            ph,
            snr_threshold,
            min_num_valid_channels_per_freq_bin=min_num_valid_channels_per_freq_bin,
            max_relative_distance_err_pct=max_relative_distance_err_pct,
        )

    # -----------------------------------------------------------
    #         Compute approximate moment magnitude
    # -----------------------------------------------------------
    approx_moment = approximate_moment_magnitude(
        spectrum,
        phases=phases,
        snr_threshold=snr_threshold,
        num_averaging_bands=approximate_moment_magnitude_args["num_averaging_bands"],
        low_snr_freq_min_hz=approximate_moment_magnitude_args["low_snr_freq_min_hz"],
        magnitude_log_moment_scaling=approximate_moment_magnitude_args[
            "magnitude_log_moment_scaling"
        ],
    )
    # fetch approximate moment magnitude
    if ph in approx_moment:
        source_parameters[ph]["Mw*"] = approx_moment[ph]

    if full_output:
        corr_disp_spectra = {}
        snr_spectra = {}
        for ph in phases:
            # 1) store propagation-corrected displacement
            # spectra in a pandas.DataFrame
            corr_disp_spectra[ph] = spectrum._spectra_pd(ph)
            # 2) store snr in a pandas.DataFrame
            snr_spectra[ph] = spectrum._snr_spectra_pd(ph)

    # -----------------------------------------------------
    #          Compute moment magnitude
    # -----------------------------------------------------
    if qc:
        # event passed location quality criterion
        for ph in spectrum.average_spectra:
            spectrum.fit_average_spectrum(
                ph,
                model=spectral_model,
                min_fraction_valid_points_below_fc=min_fraction_valid_points_below_fc,
                weighted=num_channel_weighted_fit,
            )
            if spectrum.inversion_success:
                print(f"-------------- {ph.upper()} -------------")
                rel_M0_err = 100.0 * spectrum.M0_err / spectrum.M0
                rel_fc_err = 100.0 * spectrum.fc_err / spectrum.fc
                print(f"Relative error on M0: {rel_M0_err:.2f}%")
                print(f"Relative error on fc: {rel_fc_err:.2f}%")
                if (
                    rel_M0_err > max_rel_m0_err_pct
                    or spectrum.fc < 0.0
                    or rel_fc_err > max_rel_fc_err_pct
                ):
                    print("Relative error is too large")
                else:
                    # calculate stress drop
                    stress_drop_MPa = (
                        stress_drop_circular_crack(spectrum.Mw, spectrum.fc, phase=ph)
                        / 1.0e6
                    )
                    figtitle = (
                        f"{event.origin_time.strftime('%Y-%m-%dT%H:%M:%S')}: "
                        f"{event.latitude:.3f}"
                        "\u00b0"
                        f"N, {event.longitude:.3f}"
                        "\u00b0"
                        f"E, {event.depth:.1f}km,\n"
                        r"$\Delta M_0 / M_0=$"
                        f"{rel_M0_err:.1f}%, "
                        r"$\Delta f_c / f_c=$"
                        f"{rel_fc_err:.1f}%, "
                        f"$\Delta \sigma=$"
                        f"{stress_drop_MPa:.2f}"
                    )
                    reasonable_stress_drop = (stress_drop_MPa > stress_drop_mpa_min) & (
                        stress_drop_MPa < stress_drop_mpa_max
                    )
                    reasonable_stress_drop = True
                    if reasonable_stress_drop:
                        source_parameters[ph]["M0"] = spectrum.M0
                        source_parameters[ph]["Mw"] = spectrum.Mw
                        source_parameters[ph]["fc"] = spectrum.fc
                        source_parameters[ph]["M0_err"] = spectrum.M0_err
                        source_parameters[ph]["fc_err"] = spectrum.fc_err
                        if (
                            plot_spectrum
                            and (spectrum.Mw > plot_above_mw)
                            or np.random.random() > plot_above_random
                        ):
                            fig = spectrum.plot_average_spectrum(
                                ph,
                                plot_fit=True,
                                figname=f"{ph}_spectrum_{event.id}",
                                figsize=figsize,
                                figtitle=figtitle,
                                plot_std=True,
                                plot_num_valid_channels=True,
                            )
                            figs.append(fig)
                    else:
                        print(f"Anomalous stress drop! {stress_drop_MPa:.2f}MPa")

    Mw_exists = False
    norm = 0.0
    Mw = 0.0
    Mw_err = 0.0
    Mw_app = 0.0
    for ph in phases:
        if "Mw" in source_parameters[ph]:
            Mw += source_parameters[ph]["Mw"]
            Mw_err += (
                2.0
                / 3.0
                * source_parameters[ph]["M0_err"]
                / source_parameters[ph]["M0"]
            )
            norm += 1
            Mw_exists = True
    if Mw_exists:
        Mw /= norm
        Mw_err /= norm
        print(f"The P-S averaged moment magnitude is {Mw:.2f} +/- {Mw_err:.2f}")
    else:
        Mw = np.nan
        Mw_err = np.nan
        fig = None
    source_parameters["Mw"] = Mw
    source_parameters["Mw_err"] = Mw_err

    Mw_app = 0.0
    norm = 0.0
    for ph in phases:
        if "Mw*" in source_parameters[ph]:
            Mw_app += source_parameters[ph]["Mw*"]
            norm += 1.0
    Mw_app /= norm
    source_parameters["Mw*"] = Mw_app

    output = (event, spectrum, source_parameters)
    if full_output:
        output = output + (
            corr_disp_spectra,
            snr_spectra,
        )
    if plot_spectrum:
        output = output + (figs,)
    return output
