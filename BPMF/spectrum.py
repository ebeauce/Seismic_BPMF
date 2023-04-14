import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scisig


class Spectrum:
    def __init__(self, event=None):
        self.event = event

    def attenuation_Q_model(self, Q, frequencies):
        from scipy.interpolate import interp1d

        interpolator = interp1d(
            frequencies, Q, kind="linear", fill_value=(Q[0], Q[-1]), bounds_error=False
        )
        self.attenuation_model = {}
        self.attenuation_model["p"] = np.vectorize(interpolator)
        self.attenuation_model["s"] = np.vectorize(interpolator)

    def compute_correction_factor(
        self,
        rho,
        vp,
        vs,
        radiation_S=np.sqrt(2.0 / 5.0),
        radiation_P=np.sqrt(4.0 / 15.0),
    ):
        """
        Compute the correction factor and attenuation factor for a seismic event.

        Parameters
        ----------
        rho : float
            Density of the medium, in kg/m3.
        vp : float
            P-wave velocity of the medium, in m/s.
        vs : float
            S-wave velocity of the medium, in m/s.
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
        if not hasattr(self, "event"):
            print("Attach the BPMF.dataset.Event instance first.")
            return
        if not hasattr(self.event, "_source_receiver_dist"):
            print("Call event.set_source_receiver_dist(network) first.")
            return
        correction_factor = pd.DataFrame()
        attenuation_factor = pd.DataFrame()
        for sta in self.event.source_receiver_dist.index:
            r_m = 1000.0 * self.event.source_receiver_dist.loc[sta]
            tt_s = self.event.arrival_times.loc[sta, "S_tt_sec"]
            corr_s = 2.0 * rho * vs**3 * r_m / radiation_S
            correction_factor.loc[sta, f"correction_S"] = corr_s
            if hasattr(self, "attenuation_model"):
                attenuation_factor.loc[sta, f"attenuation_S"] = lambda freq: np.exp(
                    np.pi * tt_s * freq / self.attenuation_model["s"](freq)
                )
            else:
                attenuation_factor.loc[sta, f"attenuation_S"] = None
            tt_p = self.event.arrival_times.loc[sta, "P_tt_sec"]
            corr_p = 2.0 * rho * vp**3 * r_m / radiation_P
            correction_factor.loc[sta, f"correction_P"] = corr_p
            if hasattr(self, "attenuation_model"):
                attenuation_factor.loc[sta, f"attenuation_P"] = lambda freq: np.exp(
                    np.pi * tt_p * freq / self.attenuation_model["p"](freq)
                )
            else:
                print("No attenuation model found!")
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
    ):
        phase = phase.lower()
        assert phase in ["p", "s"], "phase should be 'p' or 's'"
        assert phase in self.phases, f"You need to compute the {phase} spectrum first"
        assert hasattr(
            self, "frequencies"
        ), "You need to use set_target_frequencies first"
        if correct_propagation and not hasattr(self, "correction_factor"):
            print("You requested correcting for propagation effects. ")
            print("You need to use compute_correction_factor first.")
            return
        average_spectrum = np.ma.zeros(len(self.frequencies), dtype=np.float32)
        masked_spectra = []
        signal_spectrum = getattr(self, f"{phase}_spectrum")
        snr_spectrum = getattr(self, f"snr_{phase}_spectrum")
        for trid in signal_spectrum:
            mask = snr_spectrum[trid]["snr"] < snr_threshold
            amplitude_spectrum = signal_spectrum[trid]["fft"].copy()
            if correct_propagation:
                sta = trid.split(".")[1]
                amplitude_spectrum *= self.correction_factor.loc[
                    sta, f"correction_{phase.upper()}"
                ]
                if not pd.isnull(
                    self.attenuation_factor.loc[sta, f"attenuation_{phase.upper()}"]
                ):
                    att = self.attenuation_factor.loc[
                        sta, f"attenuation_{phase.upper()}"
                    ]
                    amplitude_spectrum *= self.attenuation_factor.loc[
                        sta, f"attenuation_{phase.upper()}"
                    ](signal_spectrum[trid]["freq"])
            masked_spectra.append(
                np.ma.masked_array(data=amplitude_spectrum, mask=mask)
            )
        if len(masked_spectra) == 0:
            # there seems to be cases when to spectra were in signal_spectrum??
            print(f"No spectra found in {phase}_spectrum")
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
        if average_log:
            log10_masked_spectra = np.ma.log10(masked_spectra)
            average_spectrum = np.ma.power(
                10.0, np.ma.mean(log10_masked_spectra, axis=0)
            )
            std_spectrum = np.ma.std(log10_masked_spectra, axis=0)
        else:
            average_spectrum = np.ma.mean(masked_spectra, axis=0)
            std_spectrum = np.ma.std(masked_spectra, axis=0)

        setattr(
            self,
            f"average_{phase}_spectrum",
            {
                "fft": average_spectrum,
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

    def compute_spectrum(self, traces, phase, taper=None, **taper_kwargs):
        """ """
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
            spectrum[tr.id]["fft"] = (
                np.fft.rfft(tr.data * taper(tr.stats.npts, **taper_kwargs))
                * tr.stats.delta
            )
            spectrum[tr.id]["freq"] = np.fft.rfftfreq(tr.stats.npts, d=tr.stats.delta)
        setattr(self, f"{phase.lower()}_spectrum", spectrum)
        if hasattr(self, "phases"):
            self.phases.append(phase)
        else:
            self.phases = [phase]

    def compute_signal_to_noise_ratio(self, phase):
        phase = phase.lower()
        assert phase in ["p", "s"], "phase should be 'p' or 's'"
        assert phase in self.phases, f"You need to compute the {phase} spectrum first"
        assert "noise" in self.phases, f"You need to compute the noise spectrum first"
        signal_spectrum = getattr(self, f"{phase}_spectrum")
        noise_spectrum = getattr(self, f"noise_spectrum")
        snr = {}
        for trid in signal_spectrum:
            snr[trid] = {}
            if trid in noise_spectrum:
                snr_ = np.abs(signal_spectrum[trid]["fft"]) / np.abs(
                    noise_spectrum[trid]["fft"]
                )
            else:
                # no noise spectrum, probably because of gap
                snr_ = np.zeros(len(signal_spectrum[trid]["fft"]), dtype=np.float32)
            snr[trid]["snr"] = snr_
            snr[trid]["freq"] = signal_spectrum[trid]["freq"]
        setattr(self, f"snr_{phase}_spectrum", snr)

    def integrate(self, phase, average=True):
        phase = phase.lower()
        if average:
            assert (
                phase in self.average_spectra
            ), f"You need to compute the average {phase} spectrum first."
            getattr(self, f"average_{phase}_spectrum")["fft"] = (
                getattr(self, f"average_{phase}_spectrum")["fft"]
                / getattr(self, f"average_{phase}_spectrum")["freq"]
            )
        else:
            assert (
                phase in self.phases
            ), f"You need to compute the {phase} spectrum first."
            spectrum = getattr(self, f"{phase}_spectrum")
            for trid in spectrum:
                spectrum[trid]["fft"] /= spectrum[trid]["freq"]

    def differentiate(self, phase, average=True):
        phase = phase.lower()
        if average:
            assert (
                phase in self.average_spectra
            ), f"You need to compute the average {phase} spectrum first."
            getattr(self, f"average_{phase}_spectrum")["fft"] = (
                getattr(self, f"average_{phase}_spectrum")["fft"]
                * getattr(self, f"average_{phase}_spectrum")["freq"]
            )
        else:
            assert (
                phase in self.phases
            ), f"You need to compute the {phase} spectrum first."
            spectrum = getattr(self, f"{phase}_spectrum")
            for trid in spectrum:
                spectrum[trid]["fft"] *= spectrum[trid]["freq"]

    def fit_average_spectrum(
        self,
        phase,
        model="brune",
        log=True,
        min_fraction_valid_points_below_fc=0.50,
        weighted=False,
    ):
        """Fit average displacement spectrum with model."""
        from scipy.optimize import curve_fit
        from functools import partial

        phase = phase.lower()
        assert (
            phase in self.average_spectra
        ), f"You need to compute the average {phase} spectrum first."
        spectrum = getattr(self, f"average_{phase}_spectrum")
        if np.sum(~spectrum["fft"].mask) == 0:
            print("Spectrum is below SNR threshold everywhere, cannot fit it.")
            self.inversion_success = False
            return
        omega0_first_guess = spectrum["fft"].data[~spectrum["fft"].mask][0]
        fc_first_guess = fc_circular_crack(moment_to_magnitude(omega0_first_guess))
        standardized_num_valid_channels = (
            spectrum["num_valid_channels"] - spectrum["num_valid_channels"].mean()
        ) / np.std(spectrum["num_valid_channels"])
        sigmoid_weights = 1.0 / (1.0 + np.exp(-standardized_num_valid_channels))
        if model == "brune":
            mod = brune
        elif model == "boatwright":
            mod = boatwright
        if log:
            obs = np.log10(spectrum["fft"].data)
        else:
            obs = spectrum["fft"].data
        # misfit = (
        #        lambda f, omega0, fc:
        #        weights * (obs - mod(f, omega0, fc, log=log))
        #        )
        mod = partial(mod, log=log)
        y = obs[~spectrum["fft"].mask]
        x = spectrum["freq"][~spectrum["fft"].mask]
        if weighted:
            inverse_weights = 1.0 / sigmoid_weights[~spectrum["fft"].mask]
        else:
            inverse_weights = None
        p0 = np.array([omega0_first_guess, fc_first_guess])
        bounds = (np.array([0.0, 0.0]), np.array([np.inf, np.inf]))
        try:
            popt, pcov = curve_fit(
                mod, x, y, p0=p0, bounds=bounds, sigma=inverse_weights
            )
            self.inversion_success = True
        except (RuntimeError, ValueError):
            self.inversion_success = False
            return
        # check whether the low-frequency plateau was well constrained
        npts_below_fc = np.sum(spectrum["freq"] < popt[1])
        npts_valid_below_fc = np.sum(x < popt[1])
        if npts_below_fc <= int(0.05 * len(spectrum["freq"])):
            self.inversion_success = False
            return
        if (
            float(npts_valid_below_fc) / float(npts_below_fc)
        ) < min_fraction_valid_points_below_fc:
            self.inversion_success = False
            return
        perr = np.sqrt(np.diag(pcov))
        self.M0 = popt[0]
        self.fc = popt[1]
        self.Mw = moment_to_magnitude(self.M0)
        self.M0_err = perr[0]
        self.fc_err = perr[1]
        self.model = model

    def resample(self, new_frequencies, phase):
        if isinstance(phase, str):
            phase = [phase]
        for ph in phase:
            ph = ph.lower()
            if not hasattr(self, f"{ph}_spectrum"):
                print(f"Attribute {ph}_spectrum does not exist.")
                continue
            spectrum = getattr(self, f"{ph}_spectrum")
            for trid in spectrum:
                spectrum[trid]["fft"] = np.interp(
                    new_frequencies,
                    spectrum[trid]["freq"],
                    np.abs(spectrum[trid]["fft"]),
                )
                spectrum[trid]["freq"] = new_frequencies

    def set_target_frequencies(self, freq_min, freq_max, num_points):
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
        import fnmatch
        from matplotlib.ticker import MaxNLocator

        if isinstance(phase, str):
            phase = [phase]
        fig, ax = plt.subplots(num=figname, figsize=figsize)
        ax.set_title(figtitle)
        for ph in phase:
            ph = ph.lower()
            if not hasattr(self, f"average_{ph}_spectrum"):
                print(f"Attribute {ph}_spectrum does not exist.")
                continue
            spectrum = getattr(self, f"average_{ph}_spectrum")
            fft = spectrum["fft"]
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
                fft = spectrum[trid]["fft"]
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
    """Compute corner frequency assuming circular crack model (Eshelby)."""
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


# workflow function
def compute_moment_magnitude(
    event,
    mag_params,
    plot_above_Mw=100.0,
    plot_above_random=1.0,
    path_figures="",
    plot=False,
    figsize=(8, 8),
):
    """ """
    event.set_moveouts_to_theoretical_times()
    event.set_moveouts_to_empirical_times()

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
    # then, read signal
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

    # -----------------------------------------------------------
    spectrum = Spectrum(event=event)
    spectrum.compute_spectrum(noise, "noise")
    spectrum.compute_spectrum(p_wave, "p")
    spectrum.compute_spectrum(s_wave, "s")

    spectrum.set_target_frequencies(
        mag_params["FREQ_MIN_HZ"], mag_params["FREQ_MAX_HZ"], mag_params["NUM_FREQS"]
    )
    spectrum.resample(spectrum.frequencies, spectrum.phases)
    spectrum.compute_signal_to_noise_ratio("p")
    spectrum.compute_signal_to_noise_ratio("s")

    # from Ford et al 2008, BSSA
    Q = mag_params["Q_1HZ"] * np.power(
        spectrum.frequencies, mag_params["ATTENUATION_N"]
    )
    spectrum.attenuation_Q_model(Q, spectrum.frequencies)
    spectrum.compute_correction_factor(
        mag_params["RHO_KGM3"],
        mag_params["VP_MS"],
        mag_params["VS_MS"],
    )

    source_parameters = {}
    for phase_for_mag in ["p", "s"]:
        spectrum.compute_network_average_spectrum(
            phase_for_mag,
            mag_params["SNR_THRESHOLD"],
            min_num_valid_channels_per_freq_bin=mag_params[
                "MIN_NUM_VALID_CHANNELS_PER_FREQ_BIN"
            ],
        )
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
