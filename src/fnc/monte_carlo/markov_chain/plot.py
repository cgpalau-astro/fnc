"""Diagnosis plots for mcmc."""

import numpy as _np
import fnc.plot as _plot

from fnc.utils import lazy as _lazy
from . import _metropolis_hastings as _mh
from . import diagnosis as _dg

_plt = _lazy.Import("matplotlib.pyplot")

__all__ = ["chains", "norm_autocorr", "corner"]

#-----------------------------------------------------------------------------

def _plot_horizontal_line(ax, x, y, color):
    ax.plot([x[0], x[-1]], [y, y], alpha=1.0, linestyle="--", color=color)

#-----------------------------------------------------------------------------

def chains(dim, walkers, bounds=None, step_limit=-1):
    """Plot walkers of a given dimension."""
    #-------------------------------------------------------------
    _fig, ax = _plot.figure(fc=(2.0, 1.4))
    #Mean acceptance all walkers
    mean_acc = _dg._mean_walkers(_dg.acceptance, walkers, dim)
    _plt.title(f"Mean acceptance = {mean_acc:0.3f}", fontsize=10)
    #-------------------------------------------------------------
    steps = _np.arange(0, len(walkers[0][0]), 1)
    step_limit = steps[step_limit]
    steps = steps[0:step_limit + 1]

    #-------------------------------------------------------------
    for i in range(len(walkers)):
        ax.plot(steps, walkers[i][dim][0:step_limit + 1], alpha=1.0)
    #-------------------------------------------------------------
    if bounds is not None:
        _plot_horizontal_line(ax, steps, bounds[dim][0], "r")
        _plot_horizontal_line(ax, steps, bounds[dim][1], "r")
    _plot_horizontal_line(ax, steps, 1.0, "k")
    #-------------------------------------------------------------
    ax.set_xlim(0.0, step_limit)
    ax.set_xlabel("steps")
    ax.set_ylabel(f"dim : {dim}")
    ax.grid("on")
    #-------------------------------------------------------------

def norm_autocorr(dim, walkers):
    """Plot normalized autocorrelation coefficient of a given dimension."""
    #--------------------------------------------------------
    _fig, ax = _plot.figure(fc=(2.0, 1.4))
    #Mean autocorrelation coefficient for all walkers
    mcc = _dg._mean_walkers(_dg.autocorr_coeff, walkers, dim)
    title = f"[dim : {dim}] Mean autocorr_coeff = {mcc:0.3f}"
    _plt.title(title, fontsize=10)
    #--------------------------------------------------------
    n = len(walkers[0][0])
    steps = _np.linspace(0.0, 1.0, n)

    for i in range(len(walkers)):
        ac = _dg.norm_autocorr(walkers[i][dim])
        ax.plot(steps, ac, alpha=1.0)
    #--------------------------------------------------------
    _plot_horizontal_line(ax, steps, 0.0, "k")
    _plot_horizontal_line(ax, steps, 0.05, "g")
    _plot_horizontal_line(ax, steps, -0.05, "g")
    _plot_horizontal_line(ax, steps, 0.15, "y")
    _plot_horizontal_line(ax, steps, -0.15, "y")
    #--------------------------------------------------------
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Normalized steps")
    ax.set_ylabel("Normalized autocorrelation")
    ax.grid("on")
    #--------------------------------------------------------

#-----------------------------------------------------------------------------

def _set_limits(ax, bounds, aspect):
    dim = _np.shape(ax)[0]

    if aspect == 1:
        limits = _np.asarray(bounds).T
        lim_inf = _np.min(limits[0])
        lim_sup = _np.max(limits[1])
        limits = ((lim_inf, lim_sup), ) * dim
    else:
        limits = bounds

    #Set x limits
    for i in range(dim):
        for j in range(dim):
            ax[j, i].set_xlim(limits[i])

    #Set y limits
    for i in range(1, dim):
        for j in range(0, i):
            ax[i, j].set_ylim(limits[i])

def _set_axis(ax, labels):
    dim = _np.shape(ax)[0]

    #Integers as default labels
    if labels is None:
        labels = [str(n) for n in range(0, dim)]

    #Set labels
    for i in range(1, dim):
        ax[i, 0].set_ylabel(labels[i])
    for i in range(dim):
        ax[dim - 1, i].set_xlabel(labels[i])

    #Remove superior triangle
    for i in range(dim):
        for j in range(i + 1, dim):
            ax[i, j].axis('off')

    #Remove ticks labels x axis
    for i in range(dim - 1):
        for j in range(0, dim):
            ax[i, j].set_xticklabels([])

    #Remove labels y axis. Set ticks to zero size to keep grid lines
    ax[0, 0].set_yticklabels([])
    ax[0, 0].tick_params(axis='y', which='both', length=0, width=0)
    for i in range(dim):
        for j in range(1, dim):
            ax[i, j].set_yticklabels([])
            ax[i, j].tick_params(axis='y', which='both', length=0, width=0)

def _plot_hist1d(ax, bins, sample, bounds, normalize_axis):
    """Plot 1d histograms."""
    dim = _np.shape(ax)[0]
    max_hist = [[]] * dim
    for i in range(dim):
        h = ax[i, i].hist(sample[i],
                          density=True,
                          histtype="step",
                          bins=bins,
                          range=bounds[i],
                          color="k")
        max_hist[i] = _np.max(h[0])

    if normalize_axis:
        max_hist1d = _np.max(max_hist)
        for i in range(dim):
            y_lim = _np.array([0.0, max_hist1d]) * 1.2
            ax[i, i].set_ylim(y_lim)
            ax[i, i].plot([1.0, 1.0], y_lim, color="r", linestyle="--")
    else:
        for i in range(dim):
            y_lim = _np.array(ax[i, i].get_ylim()) * 1.2
            ax[i, i].set_ylim(y_lim)
            ax[i, i].plot([1.0, 1.0], y_lim, color="r", linestyle="--")

def _hist2d(bins, sample, bounds):
    """Compute 2d histograms and return max value for normalization."""
    dim = len(bounds)
    h = [[0] * dim for i in range(dim)]
    for i in range(1, dim):
        for j in range(0, i):
            limits = (bounds[j], bounds[i])
            hist = _np.histogram2d(sample[j],
                                   sample[i],
                                   bins=bins,
                                   range=limits,
                                   density=True)
            h[i][j] = hist[0].T

    max_hist = _np.zeros((dim, dim))
    for i in range(1, dim):
        for j in range(0, i):
            max_hist[i, j] = _np.max(h[i][j])
    return h, _np.max(max_hist)

def _plot_hist2d(ax, bins, sample, bounds, normalize_axis, cmap):
    """Plot 2d histograms."""
    h, max_hist2d = _hist2d(bins, sample, bounds)

    if not normalize_axis:
        max_hist2d = None

    dim = _np.shape(ax)[0]
    for i in range(1, dim):
        for j in range(0, i):
            limits = (bounds[j], bounds[i])
            ax[i, j].imshow(
                h[i][j],
                origin='lower',
                extent=_np.asarray(limits).flatten(),
                cmap=cmap,
                vmin=0.0,
                vmax=max_hist2d,
                aspect='auto',
                zorder=0,
            )
            ax[i, j].scatter(1.0, 1.0, s=3.0, color='r')

def corner(walkers,
           bounds,
           bins=101,
           normalize_axis=True,
           aspect=None,
           labels=None,
           fc=1.0,
           cmap='rainbow'):
    """Corner plot sample.

    Note
    ----
    1) Useful cmap: rainbow, cubehelix_r, gnuplot2_r, gist_heat_r"""
    #-----------------------------------------------------------
    #Compute sample from walkers
    sample = _mh.get_sample(walkers)
    #-----------------------------------------------------------
    #Figure definition
    dim = _np.shape(sample)[0]
    fc = (2.0 * fc, 1.5 * _np.sqrt(2.0) * fc)
    _fig, ax = _plot.figure(dim, dim, fc=fc)
    _set_axis(ax, labels)
    _set_limits(ax, bounds, aspect)
    #-----------------------------------------------------------
    #1-D histograms
    _plot_hist1d(ax, bins, sample, bounds, normalize_axis)
    #-----------------------------------------------------------
    #2-D histograms
    _plot_hist2d(ax, bins, sample, bounds, normalize_axis, cmap)
    #-----------------------------------------------------------

#-----------------------------------------------------------------------------
