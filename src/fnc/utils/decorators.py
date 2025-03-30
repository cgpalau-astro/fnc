"""Definition of decorators.

Example
-------
def decorator_with_args(*dec_args, **dec_kwargs):
    "Decorator with arguments template."
    def decorator(function):
        @_functools.wraps(function)
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            print(dec_args)
            return output
        return wrapper
    return decorator"""

import time as _time
import functools as _functools
import termcolor as _tc
import numpy as _np

from fnc import plot as _plot
from . import human_readable as _hr

__all__ = ["time", "benchmark", "vectorize"]

#-----------------------------------------------------------------------------

def _ns_to_s(x):
    return x/1E9

def _s_to_ns(x):
    return x*1E9

def _eval_time(function, *args, **kwargs):
    start_time = _time.perf_counter_ns()
    _output = function(*args, **kwargs)
    end_time = _time.perf_counter_ns()
    time = end_time - start_time
    return time

def time(function):
    """Decorator to print the execution time of a function.

    Example
    -------
    import fnc

    @fnc.utils.decorators.time
    def function():
        return ...

    function()"""

    @_functools.wraps(function)
    def wrapper(*args, **kwargs):
        time = _eval_time(function, *args, **kwargs)
        print(f"Wall time: {_hr.time(_ns_to_s(time))}")

    return wrapper

#-----------------------------------------------------------------------------

def _data_time(time):
    #-------------------------------
    def factor_units(seconds):
        if 1E0 <= seconds:
            return 1E0, "s"
        if 1E-3 <= seconds < 1.0E0:
            return 1E-3, "ms"
        if 1E-6 <= seconds < 1.0E-3:
            return 1E-6, "μs"
        return 1E-9, "ns"

    def std_mad(x):
        mad = _np.median(abs(x - _np.median(x)))
        k = 1.4826022185056018
        return mad * k

    #-------------------------------
    time = _ns_to_s(_np.array(time))
    median = _np.median(time)
    factor, units = factor_units(median)
    return {
        'factor': factor,
        'units': units,
        'time': time / factor,
        'median': median / factor,
        'std_mad': std_mad(time) / factor,
        'minimum': _np.min(time) / factor,
        'total_time': _np.sum(time),
        'n': _np.size(time)
    }

def _print_results(data):
    #-------------------
    def decimals(x):
        d = len(str(int(x)))
        if d <= 2:
            return 0.1
        return 0.0

    #-------------------
    color = 'blue'
    time_bench = _tc.colored("    Time benchmark:", color)
    num_eval = _tc.colored("Number evaluations:", color)
    minimum = _tc.colored("           Minimum:", color)
    m_std = _tc.colored("  Median ± Std MAD:", color)

    d = decimals(data['minimum'])
    un = data['units']

    print(f"{time_bench} {_hr.time(data['total_time'])}")
    print(f"{num_eval} {data['n']:_}")
    print(f"{minimum} {data['minimum']:{d}f} {un}")
    print(f"{m_std} {data['median']:{d}f} ± {data['std_mad']:{d}f} {un}")

def _plot_hist(data):
    time = data['time']
    median = data['median']
    std_mad = data['std_mad']
    #--------------------------------
    _fig, ax = _plot.figure(fc=(0.85, 0.5))
    h = ax.hist(time,
                bins=100,
                range=[median - std_mad * 3, median + std_mad * 6],
                density=True,
                histtype="step",
                color=_plot.clr["b"])

    y_lim = max(h[0]) * 1.25
    x = [median, median]
    y = [0, y_lim]
    ax.plot(x, y, linestyle="--", c="r", label="median")
    x = [median - std_mad, median + std_mad]
    ax.fill_between(x, 0, y_lim, color='red', alpha=0.1)

    exp = int(_np.log10(data['factor']))
    ax.set_xlabel(f"time      [{data['units']}] = [·$10^{{{exp}}}$ s]")

    ax.set_ylim([0, y_lim])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0, width=0)
    #ax.legend()

def benchmark(hist=True, max_time=30.0):
    """Decorator to benchmark the execution time of a function.

    Example
    -------
    from fnc.utils import decorators

    @decorators.benchmark(hist=True, max_time=10.0)
    def benchmark(x):
        return f(x)

    benchmark(x)"""
    max_time = _s_to_ns(max_time)

    def decorator(function):

        @_functools.wraps(function)
        def wrapper(*args, **kwargs):

            time = []
            elapsed_time = 0.0
            while elapsed_time <= max_time:
                t = _eval_time(function, *args, **kwargs)
                time.append(t)
                elapsed_time += t

            data = _data_time(time)

            _print_results(data)

            if hist:
                _plot_hist(data)

        return wrapper

    return decorator

#-----------------------------------------------------------------------------

def _first_input(*args, **kwargs):
    if args:
        return args[0]
    return kwargs[list(kwargs)[0]]

def _lt_vectorize(function):
    """Vectorize lists and tuples."""

    def wrapper(*args, **kwargs):
        #--------------------------------------------
        if args:
            x = args[0]
            output = [[]] * len(x)
            for i, item in enumerate(x):
                args = (item, ) + args[1:]
                output[i] = function(*args, **kwargs)
        #--------------------------------------------
        else:
            x = kwargs[list(kwargs)[0]]
            output = [[]] * len(x)
            for i, item in enumerate(x):
                kwargs[list(kwargs)[0]] = item
                output[i] = function(*args, **kwargs)
        #--------------------------------------------
        return output

    return wrapper

def vectorize(function):
    """Decorator that vectorizes the first input of a function.

    Returns
    -------
    Same type then the input"""
    #-------------------------------------------------------------
    #Add Vectorized entry in the documentation of the function.
    function.__doc__ += "\n\n    Vectorized\n    ----------\n\
    input/output: array_like"

    @_functools.wraps(function)
    def wrapper(*args, **kwargs):
        #---------------------------------------------------------
        #Obtain first input parameter
        x = _first_input(*args, **kwargs)
        #---------------------------------------------------------
        #Return the same type than the input
        if isinstance(x, list):
            return _lt_vectorize(function)(*args, **kwargs)
        if isinstance(x, tuple):
            return tuple(_lt_vectorize(function)(*args, **kwargs))
        if isinstance(x, _np.ndarray):
            return _np.vectorize(function)(*args, **kwargs)
        return function(*args, **kwargs)
        #---------------------------------------------------------

    return wrapper

#-----------------------------------------------------------------------------
