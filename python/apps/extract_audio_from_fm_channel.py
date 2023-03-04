#!/bin/env python3

# JLS 20230303

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

# read narrowband IQ from file

from os.path import dirname, join
channel_iq_file_path = \
    join(dirname(__file__), "../../data/FM_CF98.7M_SR200K_cplx_f64.dat")
dat = np.fromfile(channel_iq_file_path, dtype=np.complex128)

file_Fs = int(200e3) # frequencies covered per sample, aka sample rate
num_channel_samples = len(dat)

p.add_plot(PlotType.SPECTROGRAM, dat, file_Fs, "input file")
p.add_plot(PlotType.PSD, dat, file_Fs, "input file")

# run frequency discriminator to get FM (demod)
# with help from http://witestlab.poly.edu/~ffund/el9043/labs/lab1.html
y = dat[1:] * np.conj(dat[:-1])
demod = np.angle(y)
p.add_plot(PlotType.PSD, demod, file_Fs, "FM demod")

# run FM deemphasis at 75 us
# Given a signal 'x5' (in a numpy array) with sampling rate Fs_y
d = file_Fs * 75e-6   # Calculate the # of samples to hit the -3dB point
x = np.exp(-1/d)      # Calculate the decay between each sample
b = [1-x]             # Create the filter coefficients
a = [1,-x]
audio_oversamp = signal.lfilter(b, a, demod)
#p.add_plot(PlotType.TIME, audio_oversamp, file_Fs, "FM deemphasis")

# downsample to 48 kHz for audio hardware
audio_samp_rate = 48e3
num_audio_samples = num_channel_samples * audio_samp_rate / file_Fs
audio_samp = signal.resample(x=audio_oversamp,
                             num=math.ceil(num_audio_samples))

# scale so it's audible
audio_samp *= 10000 / np.max(np.abs(audio_samp))

#p.add_plot(PlotType.TIME, audio_samp, audio_samp_rate, title="audio (time)")

# write to file (and play in Audacity or similar)
raw_audio_file_path = \
    join(dirname(__file__), "../../data/audio_from_CF98.7M_SR48K-mono.raw")
audio_samp.astype("int16").tofile(raw_audio_file_path)

# show plots we've recently added
p.show()
