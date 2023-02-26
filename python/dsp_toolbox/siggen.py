import numpy as np
from numpy.random import randn

def tone(freq : float, Fs : float, num_samples : int) -> np.ndarray:
    '''
    freq: frequency of tone in Hz
    Fs: samples per second to produce
    num_samples: total number of samples to output

    output: an array of IQ samples containing a clean tone at freq
    '''

    Ts = 1/Fs # time per sample

    # t is an evenly distributed range from 0 to Ts*(samples - 1)
    t = Ts * np.arange(num_samples)

    # e^(j omega t) = cos ωt + j sin ωt
    # ω is 2π * freq
    return np.exp(1j * (2 * np.pi * freq) * t)


def noise(n_pwr : int, num_samples : int) -> np.ndarray:
    '''
    n_pwr: noise power level
    num_samples: total number of samples to output

    output: an array of IQ samples containing random noise
    '''
    
    noise = (randn(num_samples) + randn(num_samples)*1j) / np.sqrt(2)
    return noise * np.sqrt(n_pwr)

