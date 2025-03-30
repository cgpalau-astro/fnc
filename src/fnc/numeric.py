"""Numpy extension."""

import numpy as _np
import fnc.utils.decorators as _decorators

__all__ = ["is_even", "is_odd",
           "is_not_nan", "within", "within_equal",
           "simspace", "arctan2",
           "print_array"]

#-----------------------------------------------------------------------------

@_decorators.vectorize
def is_even(x):
    """Determines whether x is even.

    Parameters
    ----------
    x: int

    Returns
    -------
    bool:
        True: even, False: odd"""
    #----------------------------
    if not isinstance(x, int):
        raise TypeError("x: int")
    #----------------------------
    if x % 2:
        return False
    return True


@_decorators.vectorize
def is_odd(x):
    """Determines whether x is odd.

    Parameters
    ----------
    x: int

    Returns
    -------
    bool:
        True: odd, False: even"""
    #----------------------------
    if not isinstance(x, int):
        raise TypeError("x: int")
    #----------------------------
    return not is_even(x)

#-----------------------------------------------------------------------------

def is_not_nan(x):
    return _np.logical_not(_np.isnan(x))


def within(x, x_inf, x_sup):
    """Return the truth value of (x_inf < x < x_sup) element-wise."""
    return _np.greater(x, x_inf) & _np.less(x, x_sup)


def within_equal(x, x_inf, x_sup):
    """Return the truth value of (x_inf <= x <= x_sup) element-wise."""
    return _np.greater_equal(x, x_inf) & _np.less_equal(x, x_sup)

#-----------------------------------------------------------------------------

def simspace(x0, interval, n):
    """Return n equally spaced numbers within the given interval with respect
    to the central point x0.

    Parameters
    ----------
    x0: float
        Central point.
    interval: float
        Interval [x0-interval, x0+interval]
    n: int
        Number of points.

    Returns
    -------
    numpy.ndarray[float]"""

    if is_odd(n):
        x = _np.linspace(x0 - interval, x0, n // 2 + 1)
        step = x[1] - x[0]
        y = _np.linspace(x0 + step, x0 + interval, n // 2)
        return _np.append(x, y)

    raise ValueError("'n' must be odd.")

#-----------------------------------------------------------------------------

def arctan2(y, x):
    """Numpy implementation: phi in [-pi, pi]
    This implementation:  phi in [0.0, 2.0*pi]

    Parameters
    ----------
    y: array_like
    x: array_like

    Returns
    -------
    numpy.ndarray[float]"""
    #Numpy implementation
    phi = _np.asarray(_np.arctan2(y, x))

    #This implementation
    neg = phi < 0.0
    phi[neg] = phi[neg] + 2.0*_np.pi

    return phi

#-----------------------------------------------------------------------------

def print_array(x, precision=4, include_np=False, verbose=True):

    str_array = _np.array2string(x,
                                 separator=', ',
                                 formatter={'float_kind':lambda x: f"%.{precision}f" % x},
                                 precision=precision,
                                 floatmode='fixed').replace('\n', '')

    if include_np:
        str_array = 'np.array(' + str_array + ')'

    if verbose:
        print(str_array)

    return str_array

#-----------------------------------------------------------------------------
