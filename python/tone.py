#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

Fs = 3000
Ts = 1/Fs
total_samples = 2048
t = Ts * np.arange(total_samples)
# e^(j omega t) = cos wt + j sin wt
x = np.exp(1j * 2 * np.pi * 750 * t)
noise = (np.random.randn(total_samples) + 1j * np.random.randn(total_samples)) / np.sqrt(2)
n_pwr = 3
r = x + noise * np.sqrt(n_pwr)

psd = np.abs(np.fft.fft(r))**2/(total_samples*Fs)

psd_log = 10.0 * np.log10(psd)

psd_shifted = np.fft.fftshift(psd_log)

f = np.arange(Fs/-2.0, Fs/2.0, Fs/total_samples)

plt.plot(f, psd_shifted)
plt.grid(True)
plt.show()
