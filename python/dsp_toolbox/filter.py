import numpy as np
from scipy import signal

from dsp_toolbox import siggen


def bandpass_via_lowpass(samples, band, Fs) -> np.ndarray:
    '''
    performs a bandpass filter by combining tuning with a lowpass filter
    '''

    center_hz = (band[0] + band[1]) / 2
    bandwidth_hz = band[1] - band[0]
    cutoff = bandwidth_hz * 0.1

    output = samples * siggen.tone(-center_hz, Fs, len(samples))
    lowpass = signal.firwin(numtaps=91, cutoff=cutoff, fs=Fs)
    output = np.convolve(output, lowpass, mode='same')
    output *= siggen.tone(center_hz, Fs, len(samples))

    return output


def bandstop_via_highpass(samples, band, Fs) -> np.ndarray:
    '''
    performs a bandstop filter by combining tuning with a highpass filter
    '''

    center_hz = (band[0] + band[1]) / 2
    bandwidth_hz = band[1] - band[0]
    cutoff = bandwidth_hz * 0.1

    output = samples * siggen.tone(-center_hz, Fs, len(samples))
    # a large number of taps is needed to maintain the fidelity of the large
    # bandwidth being retained through the filter
    lowstop = signal.firwin(numtaps=1191, cutoff=cutoff, fs=Fs, pass_zero=False)
    output = np.convolve(output, lowstop, mode='same')
    output *= siggen.tone(center_hz, Fs, len(samples))

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


