import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from math import ceil


class _Plot(object):
    
    _next = 0

    def __init__(self, x, y, title=None, plot_num=None):
        self.x = x
        self.y = y
        self.title = title

        if plot_num==None:
            self.plot_num = _Plot._next
            _Plot._next += 1
        else:
            self.plot_num = plot_num
            _Plot._next = max(_Plot._next + 1, plot_num+1)


class Plotter(object):
    '''
    Wraps matplotlib to make it a bit easier to get our stuff plotted
    '''

    def __init__(self):
        self._pending_plots = []

        plt.rcParams['font.family'] = [
            'Segoe UI',
            'Lucida Sans',
            'Lucida Sans Unicode',
            'sans-serif']
    
    def plot_psd(self, samples, samp_rate, title=None, plot_num=None):
        num_samples = len(samples)

        # power spectral data -- convert samples from time to freq domain
        psd = np.abs(fft.fft(samples))**2/(num_samples*samp_rate)
        psd_log = 10.0 * np.log10(psd)
        psd_shifted = fft.fftshift(psd_log)
        # psd_shifted will be our y-axis

        # frequency is our x-axis
        f = np.arange(
            samp_rate/-2.0,
            samp_rate/2.0,
            samp_rate/num_samples)

        self._pending_plots.append(
            _Plot(f, psd_shifted, title, plot_num))

    def show(self):
        '''
        Display all pending plots in a grid and clear the pending plots list.
        '''

        fig = plt.figure("DSP Ed", tight_layout=True)

        # create a 2-column gridspec that will layout our multiple plots
        num_plots = max([pp.plot_num for pp in self._pending_plots]) + 1
        if num_plots > 1:
            gs = fig.add_gridspec(int(ceil(num_plots/2)), 2)
        else:
            gs = fig.add_gridspec(1, 1)

        while self._pending_plots:
            plot_data = self._pending_plots.pop(0)

            row = int(plot_data.plot_num/2)
            col = plot_data.plot_num % 2

            # let the last plot take the remainder of its row
            if plot_data.plot_num == num_plots-1:
                subplot_spec = gs[row, col:]
            else:
                subplot_spec = gs[row, col]

            ax = fig.add_subplot(subplot_spec)
            ax.set_title(plot_data.title)
            ax.grid()
            ax.plot(plot_data.x, plot_data.y)

        plt.show()
