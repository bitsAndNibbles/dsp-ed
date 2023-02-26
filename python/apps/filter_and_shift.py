#!/bin/env python3

# JLS 20230205

if __name__ == '__main__':
    from sys import path as sys_path
    from os.path import dirname as path_dirname, join as path_join

    sys_path.append(path_join(path_dirname(__file__), ".."))


import numpy as np

from dsp_toolbox import siggen, filter
from dsp_toolbox.plotter import PlotType, Plotter


if __name__ == '__main__':

    # choose a power of 2 for the FFT to perform efficiently/correctly
    num_samples = 2**13

    Fs = 3000 # frequencies covered per sample, aka sample rate

    # initialize sample storage
    samples = np.zeros(num_samples, dtype=complex)
    
    p = Plotter(cols=4)

    # make some noise 🎉
    samples += siggen.noise(n_pwr=2, num_samples=num_samples)
    p.add_plot(PlotType.PSD, samples, Fs, "noise")

    # make waves 🏄‍♂️
    for freq in [ 900, -1250, 150 ]:
        samples += siggen.tone(freq, Fs, num_samples)
    p.add_plot(PlotType.PSD, samples, Fs, "+ tones")

    # tune out the nonsense and really, truly focus on one tone -- achieve inner ☮
    #band = [-1290, -1210]
    band = [860, 940]

    bandpass_via_lp = filter.bandpass_via_lowpass(samples, band, Fs)
    p.add_plot(PlotType.PSD, bandpass_via_lp, Fs, "bandpass via tuning/lowpass")

    bandstop_via_hp = filter.bandstop_via_highpass(samples, band, Fs)
    p.add_plot(PlotType.PSD, bandstop_via_hp, Fs, "bandstop via tuning/highpass")

    bp_via_filter_shift = filter.bandpass_via_filter_shift(samples, band, Fs)
    p.add_plot(PlotType.PSD, bp_via_filter_shift, Fs, "bandpass via complex filter convolution")

    p.add_plot(PlotType.SPECTROGRAM, bp_via_filter_shift, Fs, "oooh pretty")

    # SHOW your work 📈
    p.show()
    