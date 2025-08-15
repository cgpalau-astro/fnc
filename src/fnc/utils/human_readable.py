"""Functions that return data in human readable format."""

import humanize as _humanize
import termcolor as _tc
import numpy as _np

__all__ = ["time", "memory", "matrix", "corner", "dictionary"]

#-----------------------------------------------------------------------------

def time(seconds):
    """Human readable time.

    Parameters
    ----------
    seconds : float_like
        Units: s

    Returns
    -------
    str"""
    #----------------------------------------------------
    if seconds < 0:
        raise ValueError("seconds : float >= 0")
    #----------------------------------------------------
    d = int(seconds // 3600 // 24)
    h = int(seconds // 3600 % 24)
    m = int(seconds % 3600 // 60)
    #----------------------------------------------------
    if 10.0 < seconds:
        s = int(round(seconds % 3600 % 60, 0))
        time_hr = ""
        units = ["d", "h", "m", "s"]
        put_rest = False
        for i, item in enumerate([d, h, m, s]):
            if (item == 0) & (put_rest is False) & (i < 3):
                pass
            else:
                time_hr += f"{item}{units[i]} "
                put_rest = True
        return time_hr[0:-1]

    if 1.0 <= seconds <= 10.0:
        s = round(seconds, 1)
        time_hr = f"{s} s"
        return time_hr

    if 1E-3 <= seconds < 1.0E0:
        s = int(round(seconds * 1E3, 0))
        time_hr = f"{s} ms"
        return time_hr

    if 1E-6 <= seconds < 1.0E-3:
        s = int(round(seconds * 1E6, 0))
        time_hr = f"{s} μs"
        return time_hr

    s = int(round(seconds * 1E9, 0))
    time_hr = f"{s} ns"
    return time_hr
    #----------------------------------------------------

#-----------------------------------------------------------------------------

def memory(x):
    """Human readable memory.

    Parameters
    ----------
    x : int
        Units : bytes

    Returns
    -------
    str"""
    #---------------------------------------------
    if not isinstance(x, int | _np.int64):
        raise TypeError("x : int >= 0")
    #---------------------------------------------
    mem_hr = _humanize.naturalsize(x, binary=True)
    return mem_hr

#-----------------------------------------------------------------------------

def _y_index(i, n_digits):
    term = f"{str(i+1):>{n_digits}} "
    return _tc.colored(term, attrs=["bold"])


def _x_index(n, n_digits):
    line = " " * (3 + n_digits)
    for i in range(n):
        n_digits_i = len(str(i))
        line += str(i) + " " * (6 - n_digits_i)
    return _tc.colored(line, attrs=["bold"])


def matrix(x):
    """Human readable matrix.

    Parameters
    ----------
    x : array_like"""
    #---------------------------------
    x = _np.asarray(x)
    n = _np.shape(x)[0]
    n_digits = len(str(n))
    line = _y_index(0, n_digits)
    for i in range(0, n):
        for j in range(0, n):
            line += f"{x[i,j]: 0.2f} "
        print(line)
        line = _y_index(i + 1, n_digits)
    print(_x_index(n, n_digits))


def corner(x):
    """Human readable corner correlation matrix.

    Parameters
    ----------
    x : array_like"""

    #----------------------------------------
    def coloured_term(x):
        term = f"{x: 0.2f} "
        if 0.90 <= abs(x):
            return _tc.colored(term, 'red')
        if 0.50 <= abs(x) < 0.90:
            return _tc.colored(term, 'blue')
        if 0.25 <= abs(x) < 0.50:
            return _tc.colored(term, 'green')
        return term
    #----------------------------------------
    x = _np.asarray(x)
    n = _np.shape(x)[0]
    n_digits = len(str(n))
    line = _y_index(0, n_digits)
    for i in range(1, n):
        for j in range(0, i):
            line += coloured_term(x[i, j])
        print(line)
        line = _y_index(i, n_digits)
    print(_x_index(n - 1, n_digits))

#-----------------------------------------------------------------------------

def dictionary(x):
    """Human readable dictionary.

    Note
    ----
    1)  Nested dictionaries are printed as python prints them."""

    if not isinstance(x, dict):
        raise ValueError(f"{x} is not a dict.")

    length = []
    for key in x.keys():
        length += [len(key)]
    space = max(length)

    for key in x.keys():
        print(f"{key:>{space}}: {x[key]}")

 #-----------------------------------------------------------------------------
