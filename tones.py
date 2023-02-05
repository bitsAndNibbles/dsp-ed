# JLS 20230108

import numpy as np
import matplotlib.pyplot as plt

# Plot showing tone at 900, tone at -1250, tone at 150, and background noise.

total_samples = 2048

Fs = 3000
Ts = 1/Fs
t = Ts * np.arange(total_samples)
# e^(j omega t) = cos wt + j sin wt

# initialize sample storage
samples = np.full(total_samples, 0 + 0j)

# generate & accumulate tones

def add_tone(samples, freq):
  tone = np.exp(1j * 2 * np.pi * freq * t)
  return samples + tone

for freq in [ 900, -1250, 150 ]:
  samples = add_tone(samples, freq)

# generate & accumulate noise

noise = (np.random.randn(total_samples) + 1j * np.random.randn(total_samples)) / np.sqrt(2)
n_pwr = 4
samples = samples + noise * np.sqrt(n_pwr)

# plot results

psd = np.abs(np.fft.fft(samples))**2/(total_samples*Fs)
psd_log = 10.0 * np.log10(psd)
psd_shifted = np.fft.fftshift(psd_log)

f = np.arange(Fs/-2.0, Fs/2.0, Fs/total_samples)

plt.plot(f, psd_shifted)
plt.grid(True)
plt.show()
