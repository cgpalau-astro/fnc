"""Miscellany statistical functions."""

import numpy as _np
import scipy as _scipy
import fnc as _fnc


__all__ = [
    "mad",
    "std_mad",
    "cv",
    "cv_mad",
    "weighted_mean",
    "weighted_std",
    "rms",
    "correlation",
    #-------------------
    "best_seed_uniform",
    "best_seed_norm",
    #-------------------
    "cdf_quad",
    "sigma_error",
]

#-----------------------------------------------------------------------------

def mad(x):
    """Compute the Median Absolute Deviation (MAD).
    (https://en.wikipedia.org/wiki/Median_absolute_deviation)

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    float"""
    return _np.median(abs(x - _np.median(x)))


def std_mad(x):
    """Compute an estimate of the Standard Deviation using the Median Absolute
    Deviation (MAD) assuming x ~ Gaussian distribution. This estimate es more
    robust than the numpy.std()

    #from scipy.special import erfinv

    #Factor for a Gaussian:
    #k = 1.0/(np.sqrt(2.0)*erfinv(1.0/2.0))

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    float"""
    k = 1.4826022185056018
    return mad(x) * k

#-----------------------------------------------------------------------------

def cv(x):
    """Compute the Coefficient of Variation or Relative Standard Deviation.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    float"""
    return abs(_np.std(x) / _np.mean(x))


def cv_mad(x):
    """Estimate the Coefficient of Variation or Relative Standard Deviation
    with the median and MAD.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    float"""
    return abs(std_mad(x) / _np.median(x))

#-----------------------------------------------------------------------------

def weighted_mean(x, weights):
    """Weighted mean.

    Parameters
    ----------
    x : numpy.ndarray
    weights : numpy.ndarray

    Returns
    -------
    float"""
    return _np.sum(x * weights) / _np.sum(weights)


def weighted_std(x, weights):
    """Weighted standard deviation.

    Parameters
    ----------
    x : numpy.ndarray
    weights : numpy.ndarray

    Returns
    -------
    float"""
    w_mean = weighted_mean(x, weights)
    w_variance = weighted_mean((x - w_mean)**2.0, weights)
    return _np.sqrt(w_variance)

#-----------------------------------------------------------------------------

def rms(x):
    """Root mean square.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    float"""
    return _np.sqrt(_np.sum(x**2.0) / len(x))

#-----------------------------------------------------------------------------

def correlation(x, method='pearson'):
    """Compute the correlation matrix using Pearson and Spearman methods.

    Note
    ----
    0) Correlation: https://en.wikipedia.org/wiki/Correlation
    1) Pearson: It measures linear correlation between two sets of data.
                https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
                https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    2) Spearman: It assesses how well the relationship between two variable
                 can be described using a monotonic function.
                 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
                 https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

    Parameters
    ----------
    x : array_like
        Random samples.
    method : {'pearson', 'spearman'}
        Method, by default 'pearson'.

    Returns
    -------
    numpy.ndarray"""

    #----------------------------------------------------------------
    def spearman(x):
        n = len(x)
        m = _np.ones((n, n))
        for i in range(n):
            for j in range(i, n):
                m[i, j] = _scipy.stats.spearmanr(x[i], x[j]).statistic
                m[j, i] = m[i, j]
        return m

    #----------------------------------------------------------------
    if method == 'pearson':
        return _np.corrcoef(x)
    if method == 'spearman':
        return spearman(x)
    raise ValueError("method : {'pearson', 'spearman'}.")

"""
Biweight midcorrelation: https://en.wikipedia.org/wiki/Biweight_midcorrelation

import astropy.stats as _astropy

def bicorr(x):
    n = len(x)
    corr = _np.ones((n,n))
    for i in range(n):
        for j in range(i, n):
            corr[i,j] = _astropy.biweight_midcorrelation(x[i], x[j])
            corr[j,i] = corr[i,j]
    return corr
"""
#-----------------------------------------------------------------------------

def best_seed_uniform(interval, n_sample):
    """Determines the seed of a uniform distribution that generates the sample
    'n_sample' elements with the closest mean and std to the real ones.

    Note
    ----
    The best seed is independent of the loc and scale of the uniform
    distribution. It might be different due to numerical error.

    Parameters
    ----------
    interval: int
        Seeds tested in the interval [0, interval-1]
    n_sample: int
        Number elements sample

    Returns
    -------
    seed: int"""

    lk = _np.zeros(interval)

    for i in range(interval):
        x = _scipy.stats.uniform.rvs(size=n_sample,
                                     loc=0.0,
                                     scale=1.0,
                                     random_state=i)
        lk_m = _np.abs(_np.mean(x) - 0.5)
        lk_s = _np.abs(_np.std(x) - 1.0 / _np.sqrt(12.0))
        lk[i] = lk_m + lk_s

    seed = _np.argmin(lk)
    return seed


def best_seed_norm(interval, n_sample):
    """Determines the seed of a norm (Normal or Gaussian) distribution that
    generates the sample 'n_sample' elements with the closest mean and std to
    the real ones.

    Note
    ----
    The best seed is independent of the loc and scale of the norm
    distribution. It might be different due to numerical error.

    Parameters
    ----------
    interval: int
        Seeds tested in the interval [0, interval-1]
    n_sample: int
        Number elements sample

    Returns
    -------
    seed: int"""

    lk = _np.zeros(interval)

    for i in range(interval):
        x = _scipy.stats.norm.rvs(size=n_sample,
                                  loc=1.0,
                                  scale=1.0,
                                  random_state=i)
        lk_m = _np.abs(_np.mean(x) - 1.0)
        lk_s = _np.abs(_np.std(x) - 1.0)
        lk[i] = lk_m + lk_s

    seed = _np.argmin(lk)
    return seed

#-----------------------------------------------------------------------------

@_fnc.utils.decorators.vectorize
def cdf_quad(x, limit_inf, pdf, args=None, kwargs=None):
    """Cumulative distribution function computed integrating numerically."""
    if args is None:
        args = ()
    res = _scipy.integrate.quad(func=pdf,
                                a=limit_inf,
                                b=x,
                                args=args,
                                limit=1_500)
    return res[0]

#-----------------------------------------------------------------------------

def sigma_error(x, y, mu, x_inf, x_sup, x0_inf=None, x0_sup=None, limit=1_000, verbose=True):
    """Determine the errors of a PDF."""

    #Definition PDF using a spline
    pdf_spl = _scipy.interpolate.make_interp_spline(x, y, k=3)

    #1-Sigma level
    sigma = _fnc.stats.norm.sigma_level(1)

    #Superior area
    area_sup = _scipy.integrate.quad(pdf_spl, mu, x_sup, limit=limit)

    def f_min_sup(h):
        return _np.square( _scipy.integrate.quad(pdf_spl, mu, h, limit=limit)[0] - sigma*area_sup[0] )

    res_sup = _scipy.optimize.minimize(f_min_sup, x0=x0_sup, method='Nelder-Mead', bounds=((mu, x_sup),))

    h_sup = res_sup.x[0]

    #Inferior area
    area_inf = _scipy.integrate.quad(pdf_spl, x_inf, mu, limit=limit)

    def f_min_inf(h):
        return _np.square( _scipy.integrate.quad(pdf_spl, h, mu, limit=limit)[0] - sigma*area_inf[0] )

    res_inf = _scipy.optimize.minimize(f_min_inf, x0=x0_inf, method='Nelder-Mead', bounds=((x_inf, mu),))

    h_inf = res_inf.x[0]

    #Print results
    if verbose:
        print(res_sup)
        print()
        print(res_inf)
        print()
        print(f"Total area = {area_sup[0] + area_inf[0]}\n")

    return _np.array([mu-h_inf, h_sup-mu])

#-----------------------------------------------------------------------------
