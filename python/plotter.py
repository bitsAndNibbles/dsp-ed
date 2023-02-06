import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from enum import Enum, auto
from math import ceil

class PlotType(Enum):
    PSD = auto(),
    SPECTROGRAM = auto()

class PlotRequest(object):
    
    _next = 0

    def __init__(self,
        plot_type: PlotType,
        samples,
        samp_rate,
        title=None,
        plot_num=None):

        self.plot_type = plot_type
        self.samples = samples
        self.samp_rate = samp_rate
        self.title = title

        if plot_num==None:
            self.plot_num = PlotRequest._next
            PlotRequest._next += 1
        else:
            self.plot_num = plot_num
            PlotRequest._next = max(PlotRequest._next + 1, plot_num+1)


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
    
    def add_plot(self,
        type : PlotType,
        samples,
        samp_rate,
        title=None,
        plot_num=None):

        self._pending_plots.append(PlotRequest(
            type,
            samples,
            samp_rate,
            title=title,
            plot_num=plot_num))

    def _plot_psd_render(self, ax : plt.Axes, req : PlotRequest):
        simplified = False

        if simplified:
            ax.psd(req.samples, Fs=req.samp_rate)
        else:
            num_samples = len(req.samples)

            # power spectral data -- convert samples from time to freq domain
            psd = np.abs(fft.fft(req.samples))**2/(num_samples*req.samp_rate)
            psd_log = 10.0 * np.log10(psd)
            psd_shifted = fft.fftshift(psd_log)
            # psd_shifted will be our y-axis

            # frequency is our x-axis
            f = np.arange(
                req.samp_rate/-2.0,
                req.samp_rate/2.0,
                req.samp_rate/num_samples)
            
            ax.plot(f, psd_shifted)

    def _plot_spectrogram_render(self, ax : plt.Axes, req : PlotRequest):
        ax.specgram(req.samples, Fs=req.samp_rate)

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
            req = self._pending_plots.pop(0)

            row = int(req.plot_num/2)
            col = req.plot_num % 2

            # let the last plot take the remainder of its row
            if req.plot_num == num_plots-1:
                subplot_spec = gs[row, col:]
            else:
                subplot_spec = gs[row, col]

            ax = fig.add_subplot(subplot_spec)
            ax.set_title(req.title)
            ax.grid()

            if req.plot_type is PlotType.PSD:
                self._plot_psd_render(ax, req)
            elif req.plot_type is PlotType.SPECTROGRAM:
                self._plot_spectrogram_render(ax, req)
            else:
                raise Exception(f"Unsupported plot type {req.plot_type}")

        plt.show()
