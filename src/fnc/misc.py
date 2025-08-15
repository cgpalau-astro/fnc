import numpy as _np

__all__ = ["func_lots_minima", "func_interp"]

#-----------------------------------------------------------------------------

def func_lots_minima(x, n):
    """Function with a lot of local minima.

    Note
    ----
    1) Periodic with period [0,1].
    2) Fractal with a local minima at each rational p/q with q<sqrt(n).

    Reference
    ---------
    1)  Stefan Steinerberger. An Amusing Sequence of Functions.
        Mathematics Magazine, Vol. 91, No. 4 (October 2018), pp. 262–266"""
    k = _np.arange(1, n + 1)
    t0 = _np.abs(_np.sin(k*_np.pi*x))
    return _np.sum(t0/k)

#-----------------------------------------------------------------------------

def func_interp(x):
    return _np.sin(x[0]) + 2.0*_np.cos(x[1]) + _np.sin(5.0*x[0]*x[1])

#-----------------------------------------------------------------------------
