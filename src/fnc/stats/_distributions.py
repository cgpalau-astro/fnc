"""Probability density functions."""

import numpy as _np
import scipy as _scipy

__all__ = ["norm", "hoyt", "rayleigh", "powerlaw", "DistSpline"]

#-----------------------------------------------------------------------------

class norm:
    """Gaussian distribution.

    Note
    ----
    1)  This implementation is faster than scipy.stats.norm."""

    @staticmethod
    def pdf(x, loc=0.0, scale=1.0):
        """Probability density function"""
        t0 = -0.5 * ((x - loc) / scale)**2.0
        t1 = _np.sqrt(2.0 * _np.pi) * scale
        return _np.exp(t0) / t1

    @classmethod
    def logpdf(cls, *args, **kwargs):
        """Log-probability density function."""
        pdf = cls.pdf(*args, **kwargs)
        return _np.log(pdf)

    @staticmethod
    def entropy(loc=0.0, scale=1.0):
        """Differential entropy."""
        t0 = 2.0 * _np.pi * _np.e
        t1 = scale * _np.sqrt(t0)
        return _np.log(t1)

    @staticmethod
    def mean(loc=0.0, scale=1.0):
        """Mean."""
        return loc

    @staticmethod
    def std(loc=0.0, scale=1.0):
        """Standard deviation."""
        return scale

    @staticmethod
    def sigma_level(x, loc=0.0, scale=1.0):
        """Area within x*scale or x-sigma level.

        Note
        ----
        1)  seaborn.kdeplot levels = [1 - fnc.stats.norm.sigma_level(3),
                                      1 - fnc.stats.norm.sigma_level(2),
                                      1 - fnc.stats.norm.sigma_level(1),
                                      1]"""
        norm = _scipy.stats.norm(loc=loc, scale=1.0)
        return 1.0 - 2.0*norm.cdf(-(x - loc))

#-----------------------------------------------------------------------------

class hoyt:
    """Hoyt distribution defined as the distribution of distances from the
    origin of coordinates to the points of a bivariate Gaussian distribution
    centred at the origin.

    Note
    ----
    1)  Given a Gaussian distribution with covariance matrix cov:
        scale1 = np.sqrt(cov[0][0])
        scale2 = np.sqrt(cov[1][1])
        corrcoef = cov[0][1]/np.sqrt(cov[0][0]*cov[1][1])

    2)  Modified Bessel function of the first kind of real order:
        scipy.special.iv(0, z)
        Mathematica.BesselI[0, z]

    3)  https://math.stackexchange.com/questions/369634/what-is-the-distribution-of-sqrtx2y2-when-x-and-y-are-gaussian-but-c/4741433#4741433"""

    @staticmethod
    def pdf(x, scale1=1.0, scale2=1.0, corrcoef=0.0):
        """Probability density function"""
        omega = _np.sqrt(1.0 - corrcoef**2.0) * scale1 * scale2

        t0 = (x / 2.0 / omega)**2.0
        t1 = scale1**2.0 + scale2**2.0
        t2 = _np.exp(-t0 * t1)

        v = 0
        t3 = (2.0 * corrcoef * scale1 * scale2)**2.0
        t4 = (scale1**2.0 - scale2**2.0)**2.0
        z = t0 * _np.sqrt(t3 + t4)
        t5 = _scipy.special.iv(v, z)

        return x / omega * t2 * t5

    @classmethod
    def logpdf(cls, *args, **kwargs):
        """Log-probability density function."""
        pdf = cls.pdf(*args, **kwargs)
        return _np.log(pdf)

    @staticmethod
    def rvs(scale1=1.0, scale2=1.0, corrcoef=0.0, size=1, random_state=None):
        """Random variable sample."""
        mean = (0.0, 0.0)
        c = corrcoef * scale1 * scale2
        cov = ((scale1**2.0, c), (c, scale2**2.0))
        x = _scipy.stats.multivariate_normal.rvs(mean=mean,
                                                 cov=cov,
                                                 size=size,
                                                 random_state=random_state)
        if size == 1:
            return _np.sqrt(_np.sum(x**2.0))
        return _np.sqrt(_np.sum(x**2.0, 1))

#-----------------------------------------------------------------------------

class rayleigh:
    """Rayleigh distribution.

    Note
    ----
    1)  This implementation is faster than scipy.stats.rayleigh."""

    @staticmethod
    def pdf(x, loc=0.0, scale=1.0):
        """Probability density function"""
        t0 = (x - loc) / scale
        t1 = t0 / scale
        return t1 * _np.exp(-t0**2.0 / 2.0)

    @classmethod
    def logpdf(cls, *args, **kwargs):
        """Log-probability density function."""
        pdf = cls.pdf(*args, **kwargs)
        return _np.log(pdf)

    @staticmethod
    def cdf(x, loc=0.0, scale=1.0):
        """Cumulative distribution function"""
        t0 = (x - loc) / scale
        return 1.0 - _np.exp(-t0**2.0 / 2.0)

    @classmethod
    def logcdf(cls, *args, **kwargs):
        """Log-cumulative distribution function."""
        cdf = cls.cdf(*args, **kwargs)
        return _np.log(cdf)

    @staticmethod
    def entropy(loc=0.0, scale=1.0):
        """Differential entropy."""
        t0 = _np.euler_gamma - _np.log(2.0)
        return 1.0 + _np.log(scale) + t0 / 2.0

    @staticmethod
    def mean(loc=0.0, scale=1.0):
        """Mean."""
        return loc + _np.sqrt(_np.pi / 2.0) * scale

    @staticmethod
    def std(loc=0.0, scale=1.0):
        """Standard deviation."""
        t0 = 4.0 - _np.pi
        return _np.sqrt(t0 / 2.0) * scale

    @staticmethod
    def fit(data, floc=None):
        """Fit loc and scale given a random sample.

        Note
        ----
        1)  Fixing loc reduces the variance of the estimated scale."""
        if floc is None:
            scale = _np.std(data) * _np.sqrt(2 / (4.0 - _np.pi))
            mu = _np.sqrt(_np.pi / 2.0) * scale
            loc = _np.mean(data) - mu
        else:
            loc = floc
            #The mean has smaller variance than the std to estimate the scale
            mu = _np.mean(data) - loc
            scale = _np.sqrt(2.0 / _np.pi) * mu
        return loc, scale

#-----------------------------------------------------------------------------

class powerlaw:
    """Power-law distribution.

    Note
    ----
    1)  pdf(x) = C * x**alpha
    2)  loc <= a < b
    3)  a <= x <= b
    4)  alpha != -1"""

    @staticmethod
    def _value_error(x, loc, alpha, a, b):
        """Parameters within limits."""
        if not loc <= a < b:
            raise ValueError("loc <= a < b")
        if not _np.asarray( ((a <= x) & (x <= b)) ).all():
            raise ValueError("a <= x <= b")
        if alpha == -1.0:
            raise ValueError("alpha != -1")

    @classmethod
    def pdf(cls, x, loc=0.0, alpha=-1.5, a=1.0, b=2.0):
        """Probability density function"""
        cls._value_error(x, loc, alpha, a, b)
        t0 = 1.0 + alpha
        t1 = (b - loc)**t0 - (a - loc)**t0
        return t0 / t1 * (x - loc)**alpha

    @classmethod
    def logpdf(cls, *args, **kwargs):
        """Log-probability density function."""
        pdf = cls.pdf(*args, **kwargs)
        return _np.log(pdf)

    @classmethod
    def cdf(cls, x, loc=0.0, alpha=-1.5, a=1.0, b=2.0):
        """Cumulative distribution function"""
        cls._value_error(x, loc, alpha, a, b)
        t0 = 1.0 + alpha
        t1 = (a - loc)**t0 - (x - loc)**t0
        t2 = (a - loc)**t0 - (b - loc)**t0
        return t1 / t2

    @classmethod
    def logcdf(cls, *args, **kwargs):
        """Log-cumulative distribution function."""
        cdf = cls.cdf(*args, **kwargs)
        return _np.log(cdf)

    @classmethod
    def mean(cls, loc=0.0, alpha=-1.5, a=1.0, b=2.0):
        """Mean."""
        cls._value_error(a, loc, alpha, a, b)
        t0 = 1.0 + alpha
        t1 = t0 + 1.0
        t2 = (a - loc)**t1 - (b - loc)**t1
        t3 = (a - loc)**t0 - (b - loc)**t0
        return t0 * t2 / t1 / t3 + loc

    @classmethod
    def std(cls, loc=0.0, alpha=-1.5, a=1.0, b=2.0):
        """Standard deviation."""
        cls._value_error(a, loc, alpha, a, b)
        t0 = 1.0 + alpha
        t1 = 2.0 + alpha
        t2 = 3.0 + alpha
        t3 = t0 / ((a - loc)**t0 - (b - loc)**t0)**2.0 / t1**2.0 / t2
        t4 = 2.0*t1
        t5 = (a - loc)*(b - loc)
        t6 = ((a - loc)*t1)**2.0 + ((b - loc)*t1)**2.0 - 2.0*t5*t0*t2
        variance = t3 * ((a - loc)**t4 + (b - loc)**t4 - t5**t0 * t6)
        return _np.sqrt(variance)

    @classmethod
    def rvs(cls, loc=0.0, alpha=-1.5, a=1.0, b=2.0, size=1, random_state=None):
        """Random variable sample."""
        cls._value_error(a, loc, alpha, a, b)

        rng = _np.random.default_rng(random_state)
        u = rng.random(size=size)

        t0 = 1.0 + alpha
        t1 = (b - loc)**t0 - (a - loc)**t0
        return (t1*u + (a - loc)**t0)**(1.0/t0) + loc

#-----------------------------------------------------------------------------

class DistSpline():
    """Evaluate the CDF and generate random samples following a PDF using
    cubic splines."""

    def __init__(self, pdf, x_inf, x_sup, n_points=1E6):
        self.pdf = pdf
        self.x_inf = x_inf
        self.x_sup = x_sup
        self.n_points = n_points


    def cdf(self, x, *args, **kwargs):

        x_intv = _np.linspace(self.x_inf, self.x_sup, _np.int64(self.n_points))

        #Cummulative distribution function (CDF)
        cdf = _np.cumsum(self.pdf(x_intv, *args, **kwargs))
        cdf -= cdf.min()
        cdf /= cdf.max()

        #Definition CDF spline
        spl = _scipy.interpolate.make_interp_spline(x_intv, cdf, k=3)

        return spl(x)


    def rvs(self, size, random_state, *args, **kwargs):

        x_intv = _np.linspace(self.x_inf, self.x_sup, _np.int64(self.n_points))

        #Cummulative distribution function (CDF)
        cdf = _np.cumsum(self.pdf(x_intv, *args, **kwargs))
        cdf -= cdf.min()
        cdf /= cdf.max()

        #Definition inverse CDF using a spline
        spl = _scipy.interpolate.make_interp_spline(cdf, x_intv, k=3)

        #Uniform random sample
        rng = _np.random.default_rng(random_state)
        u = rng.random(size)

        return spl(u)

#-----------------------------------------------------------------------------
