#!/bin/env python3

# JLS 20230126

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from numpy.random import randn
from scipy import signal

# choose a power of 2 for the FFT to perform efficiently/correctly
num_samples = 2**13

Fs = 3000 # frequencies covered per sample, aka sample rate
Ts = 1/Fs # time per sample

# t is an evenly distributed range from 0 to Ts*(samples - 1)
t = Ts * np.arange(num_samples)

# initialize sample storage
samples = np.zeros(num_samples, dtype=complex)

def tone(freq):
  # e^(j omega t) = cos ωt + j sin ωt
  # ω is 2π * freq
  return np.exp(1j * (2 * np.pi * freq) * t)

def noise(n_pwr):
  noise = (randn(num_samples) + randn(num_samples)*1j) / np.sqrt(2)
  return noise * np.sqrt(n_pwr)

def band_pass(samples, min_hz, max_hz):
  '''
  performs a band pass filter by combining tuning with a low pass filter
  '''

  center_hz = (max_hz + min_hz) / 2
  bandwidth_hz = max_hz - min_hz
  cutoff = bandwidth_hz * 0.1

  samples *= tone(-center_hz)
  lowpass = signal.firwin(numtaps=91, cutoff=cutoff, fs=Fs)
  samples = np.convolve(samples, lowpass, mode='same')
  samples *= tone(center_hz)

  return samples

def plot_psd():
  psd = np.abs(fft.fft(samples))**2/(num_samples*Fs)
  psd_log = 10.0 * np.log10(psd)
  psd_shifted = fft.fftshift(psd_log)

  f = np.arange(Fs/-2.0, Fs/2.0, Fs/num_samples)

  plt.plot(f, psd_shifted)
  plt.grid(True)
  plt.show()


if __name__ == '__main__':

  # make some noise 🎉
  samples += noise(2)

  # make waves 🏄‍♂️
  for freq in [ 900, -1250, 150 ]:
    samples += tone(freq)

  # tune out the nonsense and really, truly focus on one tone -- achieve inner ☮
  samples = band_pass(samples, 880, 920)

  # SHOW your work 📈
  plot_psd()
