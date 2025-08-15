"""Kullback–Leibler divergence."""

import numpy as _np
import scipy as _scipy

__all__ = ["quad", "rvs", "norm_1d", "norm", "rayleigh"]

#-----------------------------------------------------------------------------

def quad(p, q, limit_inf, limit_sup):
    """Kullback–Leibler divergence computed integrating numerically."""

    def f(x, p, q):
        return p(x) * _np.log(p(x) / q(x))

    return _scipy.integrate.quad(f, limit_inf, limit_sup, args=(p, q))

def rvs(p, q, size, random_state=None):
    """Kullback–Leibler divergence estimated with a random sample.

    Note
    ----
    1)  p and q are scipy.stats distributions."""
    sample = p.rvs(size=size, random_state=random_state)
    p_rvs = p.logpdf(sample)
    q_rvs = q.logpdf(sample)
    return _np.mean(p_rvs - q_rvs)

#-----------------------------------------------------------------------------

def norm_1d(loc_p, scale_p, loc_q, scale_q):
    """Kullback–Leibler divergence between two uni-dimensional Gaussian
    distributions."""
    t0 = ((loc_q - loc_p) / scale_q)**2.0
    t1 = (scale_p / scale_q)**2.0
    t2 = _np.log(t1)
    return 0.5 * (t0 + t1 - t2 - 1.0)

def norm(mean_p, cov_p, mean_q, cov_q):
    """Kullback–Leibler divergence between two n-dimensional Gaussian
    distributions."""
    mean_p = _np.asarray(mean_p)
    mean_q = _np.asarray(mean_q)

    dim = _np.size(mean_p)

    det_p = _np.linalg.det(cov_p)
    det_q = _np.linalg.det(cov_q)

    inv_q = _np.linalg.inv(cov_q)

    m = mean_q - mean_p

    t0 = _np.dot(m.T, _np.dot(inv_q, m))
    t1 = _np.trace(_np.dot(inv_q, cov_p))
    t2 = _np.log(det_q / det_p)
    return 0.5 * (t0 + t1 + t2 - dim)

#-----------------------------------------------------------------------------

def rayleigh(scale_p, scale_q):
    """Kullback–Leibler divergence between two Rayleigh distributions.

    Note
    ----
    loc_p = loc_q = 0"""
    t0 = (scale_p / scale_q)**2.0
    return t0 - _np.log(t0) - 1.0

#-----------------------------------------------------------------------------
