"""Diagnosis tools for mcmc."""

import numpy as _np
import scipy as _scipy
import termcolor as _tc
from . import _metropolis_hastings as _mh

__all__ = ["acceptance", "norm_autocorr", "autocorr_coeff", "print_stats"]

#-----------------------------------------------------------------------------

def acceptance(x):
    """Fraction of non-repeated elements in a chain.

    Note
    ----
    1) Larger is better.
    2) Smaller is the step_size -> Larger the acceptance."""
    return len(set(x)) / len(x)

def norm_autocorr(x):
    """Normalized autocorrelation.

    Note
    ----
    1) Defined from 1 to 0.
    2) Faster it decays and closer to zero -> Less autocorrelated."""
    x = (x - _np.mean(x)) / _np.std(x)
    corr = _scipy.signal.correlate(x, x, mode='full')
    n = len(x)
    return corr[n - 1:] / n

def autocorr_coeff(x):
    """Normalized autocorrelation coefficient.

    Note
    ----
    1) [0, 1] = [Weak, Strong] autocorrelation.
    2) Smaller is better.
    3) Longer is the chain -> Smaller autocorrelation coefficient.
    4) Larger is the step_size -> Smaller autocorrelation coefficient."""
    return _np.mean(_np.abs(norm_autocorr(x)))

#-----------------------------------------------------------------------------

def _mean_walkers(statistic, walkers, dim):
    """Mean statistic for all walkers of a given dimension."""
    n = len(walkers)
    stat = _np.zeros(n)
    for i in range(n):
        stat[i] = statistic(walkers[i][dim])
    return _np.mean(stat)

#-----------------------------------------------------------------------------

def print_stats(walkers):
    """Print diagnosis statistics of the mcmc."""

    #--------------------------------------------------
    def colour_scale_acceptance(x):
        term = f"{x: 0.2f}"
        if 0.6 <= x:
            return _tc.colored(term, "green")
        if 0.4 <= x < 0.6:
            return _tc.colored(term, "yellow")
        return _tc.colored(term, "red")

    def colour_scale_autocorr(x):
        term = f"{x: 0.2f}"
        if x <= 0.1:
            return _tc.colored(term, "green")
        if 0.1 < x <= 0.2:
            return _tc.colored(term, "yellow")
        return _tc.colored(term, "red")

    #--------------------------------------------------
    dim = len(walkers[0])
    n_walkers = len(walkers)
    n_steps = len(walkers[0][0])
    size_sam = len(_mh.get_sample(walkers)[0])

    line_0, line_1, line_2 = "", "", ""
    for dim in range(dim):
        acc = _mean_walkers(acceptance, walkers, dim)
        ac = _mean_walkers(autocorr_coeff, walkers, dim)
        line_0 += f"{dim}" + " " * 4
        line_1 += colour_scale_acceptance(acc)
        line_2 += colour_scale_autocorr(ac)
    #--------------------------------------------------
    print(_tc.colored("Number dimensions : ", attrs=["bold"]) + f"{dim:_}")
    print(
        _tc.colored("Number walkers    : ", attrs=["bold"]) + f"{n_walkers:_}")
    print(_tc.colored("Number steps      : ", attrs=["bold"]) + f"{n_steps:_}")
    print(
        _tc.colored("Sample size       : ", attrs=["bold"]) + f"{size_sam:_}")
    print()
    print(_tc.colored("Dimensions          " + line_0, attrs=["bold"]))
    print(_tc.colored("Acceptance     :", attrs=["bold"]) + line_1)
    print(_tc.colored("Autocorr_coeff :", attrs=["bold"]) + line_2)

#-----------------------------------------------------------------------------
