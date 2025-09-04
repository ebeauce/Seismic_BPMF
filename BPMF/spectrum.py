import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scisig
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
            object's attributes `correction_factor` and `attenuation_factor`.

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
        correction_factor = pd.DataFrame(index=stations)
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
            correction_factor.loc[sta, f"correction_S"] = corr_s
            if hasattr(self, "Q"):
                attenuation_factor.loc[sta, f"attenuation_S"] = np.exp(
                    np.pi * tt_s * np.asarray(self.frequencies) / self.Q
                )
            else:
                attenuation_factor.loc[sta, f"attenuation_S"] = None

            tt_p = self.event.arrival_times.loc[sta, "P_tt_sec"]
            corr_p = (
                   4.0 * np.pi
                   *
                   np.sqrt(rho_receiver) * np.sqrt(rho_source)
                   *
                   np.sqrt(vp_receiver) * vp_source**(5./2.)
                   *
                   r_m / radiation_P
                   )
            correction_factor.loc[sta, f"correction_P"] = corr_p
            if hasattr(self, "Q"):
                attenuation_factor.loc[sta, f"attenuation_P"] = np.exp(
                    np.pi * tt_p * np.asarray(self.frequencies) / self.Q
                )
            else:
                attenuation_factor.loc[sta, f"attenuation_P"] = None
        self.correction_factor = correction_factor
        self.attenuation_factor = attenuation_factor

    def compute_network_average_spectrum(
        self,
        phase,
        snr_threshold,
        correct_propagation=True,
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
        correct_propagation : bool, optional
            Flag indicating whether to correct for propagation effects. Default is True.
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
        if correct_propagation and not hasattr(self, "correction_factor"):
            print("You requested correcting for propagation effects. ")
            print("You need to use compute_correction_factor first.")
            return
        average_spectrum = np.ma.zeros(len(self.frequencies), dtype=np.float64)
        masked_spectra = []
        signal_spectrum = getattr(self, f"{phase}_spectrum")
        snr_spectrum = getattr(self, f"snr_{phase}_spectrum")
        if hasattr(self, "Q"):
            self.update_Q_model()
            self.update_attenuation_factor()
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
            if correct_propagation:
                sta = trid.split(".")[1]
                amplitude_spectrum *= self.correction_factor.loc[
                    sta, f"correction_{phase.upper()}"
                ]
                if (
                    self.attenuation_factor.loc[sta, f"attenuation_{phase.upper()}"]
                    is not None
                ):
                    # att = self.attenuation_factor.loc[
                    #    sta, f"attenuation_{phase.upper()}"
                    # ](signal_spectrum[trid]["freq"])
                    amplitude_spectrum *= self.attenuation_factor.loc[
                        sta, f"attenuation_{phase.upper()}"
                    ]
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
            nyq = tr.stats.sampling_rate / 2.
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
                            endtime=tr_band.stats.endtime - buffer_seconds
                            )
                    spectrum[tr.id]["filtered_traces"][str(band)] = tr_band
                if len(trimmed_tr) == 0:
                    # gap in data?
                    continue
                max_amp = np.max(np.abs(trimmed_tr)) / bandwidth
                spectrum[tr.id]["spectrum"][i] = max_amp
            #print(spectrum[tr.id]["spectrum"])
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
            snr_ = np.zeros(
                len(signal_spectrum[trid]["spectrum"]), dtype=np.float64
            )
            if trid in noise_spectrum:
                signal = np.abs(signal_spectrum[trid]["spectrum"])
                noise = np.abs(noise_spectrum[trid]["spectrum"])
                zero_by_zero = (signal == 0.) & (noise == 0.)
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
        bounds = (np.array([0.0, 0.0]), np.array([np.inf, 1.e3 * fc_first_guess]))
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
                outside_bandwidth = (
                        new_frequencies >= 0.99 * np.max(spectrum[trid]["freq"])
                        )
                spectrum[trid]["spectrum"][outside_bandwidth] = 0.
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
                    amplitude_spec *= self.correction_factor.loc[
                        sta, f"correction_{ph.upper()}"
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


# workflow function
def compute_moment_magnitude(
    event,
    mag_params,
    plot_above_Mw=100.0,
    plot_above_random=1.0,
    path_figures="",
    phases=["p", "s"],
    plot=False,
    figsize=(8, 8),
):
    """ """
    event.set_moveouts_to_theoretical_times()
    event.set_moveouts_to_empirical_times()
    spectrum = Spectrum(event=event)

    # ---------------------------------------------------------------
    #                 extract waveforms
    # first, read short extract before signal as an estimate of noise
    event.read_waveforms(
        mag_params["DURATION_SEC"],
        time_shifted=False,
        data_folder=mag_params["DATA_FOLDER"],
        offset_ot=mag_params["OFFSET_OT_SEC_NOISE"],
        attach_response=mag_params["ATTACH_RESPONSE"],
    )
    noise = event.traces.copy()
    noise.remove_sensitivity()
    spectrum.compute_spectrum(noise, "noise")
    # then, read signal
    if "p" in phases:
        event.read_waveforms(
            mag_params["DURATION_SEC"],
            phase_on_comp=mag_params["PHASE_ON_COMP_P"],
            offset_phase=mag_params["OFFSET_PHASE"],
            time_shifted=mag_params["TIME_SHIFTED"],
            data_folder=mag_params["DATA_FOLDER"],
            attach_response=mag_params["ATTACH_RESPONSE"],
        )
        event.traces.remove_sensitivity()
        event.zero_out_clipped_waveforms(kurtosis_threshold=-1)
        p_wave = event.traces.copy()
        spectrum.compute_spectrum(p_wave, "p")
    if "s" in phases:
        event.read_waveforms(
            mag_params["DURATION_SEC"],
            phase_on_comp=mag_params["PHASE_ON_COMP_S"],
            offset_phase=mag_params["OFFSET_PHASE"],
            time_shifted=mag_params["TIME_SHIFTED"],
            data_folder=mag_params["DATA_FOLDER"],
            attach_response=mag_params["ATTACH_RESPONSE"],
        )
        event.traces.remove_sensitivity()
        event.zero_out_clipped_waveforms(kurtosis_threshold=-1)
        s_wave = event.traces.copy()
        spectrum.compute_spectrum(s_wave, "s")

    # -----------------------------------------------------------
    spectrum.set_target_frequencies(
        mag_params["FREQ_MIN_HZ"], mag_params["FREQ_MAX_HZ"], mag_params["NUM_FREQS"]
    )
    spectrum.resample(spectrum.frequencies, spectrum.phases)
    for ph in phases:
        spectrum.compute_signal_to_noise_ratio(ph)

    # from Ford et al 2008, BSSA
    Q = mag_params["Q_1HZ"] * np.power(
        spectrum.frequencies, mag_params["ATTENUATION_N"]
    )
    spectrum.attenuation_Q_model(Q, spectrum.frequencies)
    spectrum.compute_correction_factor(
        mag_params["RHO_SOURCE_KGM3"],
        mag_params["RHO_RECEIVER_KGM3"],
        mag_params["VP_SOURCE_MS"],
        mag_params["VP_RECEIVER_MS"],
        mag_params["VS_SOURCE_MS"],
        mag_params["VS_RECEIVER_MS"],
    )

    source_parameters = {}
    for phase_for_mag in phases:
        spectrum.compute_network_average_spectrum(
            phase_for_mag,
            mag_params["SNR_THRESHOLD"],
            min_num_valid_channels_per_freq_bin=mag_params[
                "MIN_NUM_VALID_CHANNELS_PER_FREQ_BIN"
            ],
            max_relative_distance_err_pct=mag_params["MAX_RELATIVE_DISTANCE_ERR_PCT"],
        )
        if not phase_for_mag in spectrum.average_spectra:
            continue
        spectrum.integrate(phase_for_mag, average=True)
        spectrum.fit_average_spectrum(
            phase_for_mag,
            model=mag_params["SPECTRAL_MODEL"],
            min_fraction_valid_points_below_fc=mag_params[
                "MIN_FRACTION_VALID_POINTS_BELOW_FC"
            ],
            weighted=mag_params["NUM_CHANNEL_WEIGHTED_FIT"],
        )
        if spectrum.inversion_success:
            rel_M0_err = 100.0 * spectrum.M0_err / spectrum.M0
            rel_fc_err = 100.0 * spectrum.fc_err / spectrum.fc
            if (
                rel_M0_err > mag_params["MAX_REL_M0_ERR_PCT"]
                or spectrum.fc < 0.0
                or spectrum.fc > mag_params["MAX_REL_FC_ERR_PCT"]
            ):
                continue
            print(f"Relative error on M0: {rel_M0_err:.2f}%")
            print(f"Relative error on fc: {rel_fc_err:.2f}%")
            # event.set_aux_data({f"Mw_{phase_for_mag}": spectrum.Mw})
            figtitle = (
                f"{event.origin_time.strftime('%Y-%m-%dT%H:%M:%S')}: "
                f"{event.latitude:.3f}"
                "\u00b0"
                f"N, {event.longitude:.3f}"
                "\u00b0"
                f"E, {event.depth:.1f}km, "
                r"$\Delta M_0 / M_0=$"
                f"{rel_M0_err:.1f}%, "
                r"$\Delta f_c / f_c=$"
                f"{rel_fc_err:.1f}%"
            )
            source_parameters[f"M0_{phase_for_mag}"] = spectrum.M0
            source_parameters[f"Mw_{phase_for_mag}"] = spectrum.Mw
            source_parameters[f"fc_{phase_for_mag}"] = spectrum.fc
            source_parameters[f"M0_err_{phase_for_mag}"] = spectrum.M0_err
            source_parameters[f"fc_err_{phase_for_mag}"] = spectrum.fc_err
            if (
                plot
                and (spectrum.Mw > plot_above_Mw)
                or np.random.random() > plot_above_random
            ):
                fig = spectrum.plot_average_spectrum(
                    phase_for_mag,
                    plot_fit=True,
                    figname=f"{phase_for_mag}_spectrum_{event.id}",
                    figsize=figsize,
                    figtitle=figtitle,
                    plot_std=True,
                    plot_num_valid_channels=True,
                )
                fig.savefig(
                    os.path.join(path_figures, fig._label + ".png"),
                    format="png",
                    bbox_inches="tight",
                )
                plt.close(fig)

    Mw_exists = False
    norm = 0.0
    Mw = 0.0
    Mw_err = 0.0
    for ph in ["p", "s"]:
        if f"Mw_{ph}" in source_parameters:
            Mw += source_parameters[f"Mw_{ph}"]
            Mw_err += (
                2.0
                / 3.0
                * source_parameters[f"M0_err_{ph}"]
                / source_parameters[f"M0_{ph}"]
            )
            norm += 1
            Mw_exists = True
    if Mw_exists:
        Mw /= norm
        Mw_err /= norm
        source_parameters["Mw"] = Mw
        source_parameters["Mw_err"] = Mw_err
    else:
        Mw = np.nan
        Mw_err = np.nan

    if Mw_exists:
        print(f"The P-S averaged moment magnitude is {Mw:.2f} +/- {Mw_err:.2f}")
        source_parameters["Mw"] = Mw
        source_parameters["Mw_err"] = Mw_err

    event.set_aux_data(source_parameters)

    return event, spectrum
