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
        """ """
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
                print(attenuation_factor.loc[sta, f"attenuation_P"])
            else:
                print("No attenuation model found!")
                attenuation_factor.loc[sta, f"attenuation_P"] = None
        self.correction_factor = correction_factor
        self.attenuation_factor = attenuation_factor

    def compute_network_average_spectrum(
        self, phase, snr_threshold, correct_propagation=True, average_log=True
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
                    print(att)
                    print(pd.isnull(att), ~pd.isnull(att))
                    amplitude_spectrum *= self.attenuation_factor.loc[
                        sta, f"attenuation_{phase.upper()}"
                    ](signal_spectrum[trid]["freq"])
            masked_spectra.append(
                np.ma.masked_array(data=amplitude_spectrum, mask=mask)
            )
        masked_spectra = np.ma.asarray(masked_spectra)
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
                snr_ = np.zeros(
                        len(signal_spectrum[trid]["fft"]),
                        dtype=np.float32
                        )
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
        self, phase, model="brune", log=True, min_fraction_valid_points_below_fc=0.50
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
        weights = np.ones(len(spectrum["fft"]), dtype=np.float32)
        weights[spectrum["fft"].mask] = 0.0
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
        p0 = np.array([omega0_first_guess, fc_first_guess])
        bounds = (np.array([0.0, 0.0]), np.array([np.inf, np.inf]))
        try:
            popt, pcov = curve_fit(mod, x, y, p0=p0, bounds=bounds)
            self.inversion_success = True
        except RuntimeError:
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
    ):
        import fnmatch

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
