#!/bin/env python3

# JLS 20230226

if __name__ == '__main__':
    from sys import path as sys_path
    from os.path import dirname as path_dirname, join as path_join

    sys_path.append(path_join(path_dirname(__file__), ".."))


import math
import numpy as np
from scipy import signal

from dsp_toolbox import siggen, filter
from dsp_toolbox.plotter import PlotType, Plotter


p = Plotter(cols = 2)

# read in file
from os.path import dirname, join
raw_in_file_path = \
    join(dirname(__file__), "../../data/FM_CF98.45M_SR1.2M_i8.dat")
dat = np.fromfile(raw_in_file_path, dtype="int8")

# convert the interleaved I and Q samples into complex values
# the syntax "dat[0::2]" means "every 2nd value in the array starting from the
# 0th until the end"
dat = dat[0::2] + (1j * dat[1::2])

file_Fs = int(1.2e6) # frequencies covered per sample, aka sample rate
num_samples = len(dat)
p.add_plot(PlotType.SPECTROGRAM, dat, file_Fs, "input file")
p.add_plot(PlotType.PSD, dat, file_Fs, "input file")

# tune to interesting channel that I observe by sight on the above plot, which
# is 250 kHz to the right of the original data file's center freq
tune_freq = 250e3
tuned_dat = y = dat * siggen.tone(-tune_freq, file_Fs, num_samples)
# note: our signal is now complex-float since siggen.tone returns a
# floating point array
p.add_plot(PlotType.SPECTROGRAM, tuned_dat, file_Fs, title="tuned")
p.add_plot(PlotType.PSD, tuned_dat, file_Fs, title="tuned")

p.show()

# filter out unwanted signals
channel_bw = 100e3  # as observed in plot
filtered_dat = filter.lowpass_via_remez(tuned_dat, channel_bw, file_Fs, 49)
p.add_plot(PlotType.SPECTROGRAM, filtered_dat, file_Fs, title="filtered")
p.add_plot(PlotType.PSD, filtered_dat, file_Fs, title="filtered")

# downsample
decimation = int(math.floor(file_Fs / channel_bw))
channel_Fs = file_Fs / decimation
decimated_dat = signal.decimate(filtered_dat, decimation)
p.add_plot(PlotType.SPECTROGRAM, decimated_dat, channel_Fs, title="decimated")
p.add_plot(PlotType.PSD, decimated_dat, channel_Fs, title="decimated")

# write narrowband to file
channel_iq_file_path = \
    join(dirname(__file__), "../../data/FM_CF98.7M_SR100K_cplx_f64.dat")
np.ndarray.tofile(decimated_dat, channel_iq_file_path)

# show plots we've recently added
p.show()
