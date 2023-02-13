#!/bin/env python3

# JLS 20230205

import numpy as np
from numpy.random import randn
from scipy import signal
from plotter import PlotType, Plotter


def tone(freq : float) -> np.ndarray:
    Ts = 1/Fs # time per sample

    # t is an evenly distributed range from 0 to Ts*(samples - 1)
    t = Ts * np.arange(num_samples)

    # e^(j omega t) = cos ωt + j sin ωt
    # ω is 2π * freq
    return np.exp(1j * (2 * np.pi * freq) * t)


def noise(n_pwr : int, num_samples : int) -> np.ndarray:
    noise = (randn(num_samples) + randn(num_samples)*1j) / np.sqrt(2)
    return noise * np.sqrt(n_pwr)


def bandpass_via_lowpass(samples, band, Fs) -> np.ndarray:
    '''
    performs a bandpass filter by combining tuning with a lowpass filter
    '''

    center_hz = (band[0] + band[1]) / 2
    bandwidth_hz = band[1] - band[0]
    cutoff = bandwidth_hz * 0.1

    output = samples * tone(-center_hz)
    lowpass = signal.firwin(numtaps=91, cutoff=cutoff, fs=Fs)
    output = np.convolve(output, lowpass, mode='same')
    output *= tone(center_hz)

    return output


def bandstop_via_highpass(samples, band, Fs) -> np.ndarray:
    '''
    performs a bandstop filter by combining tuning with a highpass filter
    '''

    center_hz = (band[0] + band[1]) / 2
    bandwidth_hz = band[1] - band[0]
    cutoff = bandwidth_hz * 0.1

    output = samples * tone(-center_hz)
    # a large number of taps is needed to maintain the fidelity of the large
    # bandwidth being retained through the filter
    lowstop = signal.firwin(numtaps=1191, cutoff=cutoff, fs=Fs, pass_zero=False)
    output = np.convolve(output, lowstop, mode='same')
    output *= tone(center_hz)

    return output


def bandpass_via_filter_shift(samples, band, Fs) -> np.ndarray:
    '''
    Creates a filter, rotates the filter by the center freq of the
    bandpass, and then convolves the samples with this complex filter.
    (After shifting the filter, it is a complex filter. Convolution
    will be more processor intensive than the low-pass filter approach.)
    '''

    center_hz = (band[0] + band[1]) / 2
    bandwidth_hz = band[1] - band[0]
    cutoff = (bandwidth_hz/2) * 1.1

    taps = signal.firwin(numtaps=39, cutoff=cutoff, fs=Fs)

    # technique adapted from
    # https://github.com/gnuradio/gnuradio/blob/main/gr-filter/python/filter/freq_xlating_fft_filter.py
    phase_inc = (2.0 * np.pi * center_hz) / Fs
    rtaps = [x * np.exp(i * phase_inc * 1j) for i, x in enumerate(taps)]

    output = np.convolve(samples, rtaps, mode='same')

    return output


if __name__ == '__main__':

    # choose a power of 2 for the FFT to perform efficiently/correctly
    num_samples = 2**13

    Fs = 3000 # frequencies covered per sample, aka sample rate

    # initialize sample storage
    samples = np.zeros(num_samples, dtype=complex)
    
    p = Plotter(cols=4)

    # make some noise 🎉
    samples += noise(n_pwr=2, num_samples=num_samples)
    p.add_plot(PlotType.PSD, samples, Fs, "noise")

    # make waves 🏄‍♂️
    for freq in [ 900, -1250, 150 ]:
        samples += tone(freq)
    p.add_plot(PlotType.PSD, samples, Fs, "+ tones")

    # tune out the nonsense and really, truly focus on one tone -- achieve inner ☮
    #band = [-1290, -1210]
    band = [860, 940]

    bandpass_via_lp = bandpass_via_lowpass(samples, band, Fs)
    p.add_plot(PlotType.PSD, bandpass_via_lp, Fs, "bandpass via tuning/lowpass")

    bandstop_via_hp = bandstop_via_highpass(samples, band, Fs)
    p.add_plot(PlotType.PSD, bandstop_via_hp, Fs, "bandstop via tuning/highpass")

    bp_via_filter_shift = bandpass_via_filter_shift(samples, band, Fs)
    p.add_plot(PlotType.PSD, bp_via_filter_shift, Fs, "bandpass via complex filter convolution")

    p.add_plot(PlotType.SPECTROGRAM, bp_via_filter_shift, Fs, "oooh pretty")

    # SHOW your work 📈
    p.show()
    