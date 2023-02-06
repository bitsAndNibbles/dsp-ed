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


def bandpass_via_bandpass(samples, band, Fs) -> np.ndarray:
    '''
    performs a bandpass filter using a range of cutoff freqs
    '''

    # scipy doesn't permit filters to be created with negative bandpasses.
    # so if band contains negative values, then we must shift the frequencies
    # such that the filtered area is positive and then shift back after
    # convolution.

    shift = 0
    output = samples

    if band[0] < 0:
        shift = Fs/2

    if shift: output = samples * tone(-shift)

    bandpass_filter = signal.firwin(
        numtaps=191,
        cutoff=[band[0]+shift, band[1]+shift],
        fs=Fs,
        pass_zero=False)
    
    output = np.convolve(output, bandpass_filter, mode='same')

    if shift: output *= tone(shift)

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

    bandpass_direct = bandpass_via_bandpass(samples, band, Fs)
    p.add_plot(PlotType.PSD, bandpass_direct, Fs, "bandpass w/ conditional tuning")

    p.add_plot(PlotType.SPECTROGRAM, bandpass_via_lp, Fs, "oooh pretty")

    # SHOW your work 📈
    p.show()
    