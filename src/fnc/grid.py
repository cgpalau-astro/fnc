"""Evaluate a function in a multidimensional volume defined by linearly spaced
points. Plot the two-dimensional case.

Note
----
1) The test functions perform the same calculation for 2 and 4 dimensions.

Example 1
---------
import os
import scipy
from fnc import grid

#Definition multivariate Gaussian distribution
mean = [1.0, 1.0, 1.0, 1.0]
cov = [[ 0.05, 0.00, -0.03, 0.00],
       [ 0.00, 0.05,  0.00, 0.00],
       [-0.03, 0.00,  0.05, 0.00],
       [ 0.00, 0.00,  0.00, 0.04]]
norm = scipy.stats.multivariate_normal(mean=mean, cov=cov)

n_points = (15, 11, 13, 21)
bounds = ((0.5, 1.5), (0.25, 1.75), (0.75, 1.25), (0.5, 1.5))
#n_cpu = os.cpu_count()
points, values = grid.eval(n_points, bounds, norm.pdf, n_cpu=1)

xi = [0.53, 1.45, 0.89, 0.88]
fx_intp = scipy.interpolate.interpn(points, values, xi, method='pchip')[0]
fx = norm.pdf(xi)
fx_intp/fx
#>>>0.953561551958445

Example 2
---------
from fnc import grid
from fnc import stats

#Evaluation Gaussian distribution
points, values = grid.eval(n_points=150,
                           bounds=(0.5, 1.5),
                           f=stats.norm.pdf,
                           kwargs={'loc':1.0, 'scale':0.12},
                           progress=True)

Example 3
---------
import os
import scipy
import numpy as np
from fnc import grid

#Definition bi-variate Gaussian distribution
mean = [1.0, 1.0]
cov = [[ 0.05, 0.005],
       [ 0.005, 0.05]]
norm = scipy.stats.multivariate_normal(mean=mean, cov=cov)

n_points = (15, 15)
bounds = ((0.25, 1.75), (0.25, 1.75))
points, values = grid.eval(n_points, bounds, norm.pdf, n_cpu=1)

#2D plot
ax = grid.plot_2d(points, values, number_colors=1_000, cmap="rainbow")
[X, Y] = np.meshgrid(points[0], points[1])
ax.contour(X, Y, values.T, levels=[0.5], origin="lower", colors="k", zorder=1)

#Interpolation
def interp(x, points, values):
    return scipy.interpolate.interpn(points, values, x, method='pchip')[0]

n_points = (100, 100)
points_interp, values_interp = grid.eval(n_points, bounds, interp, args=(points, values), n_cpu=1)

#2D interpolated plot
ax = grid.plot_2d(points_interp, values_interp, number_colors=1_000, cmap="rainbow")
[X, Y] = np.meshgrid(points_interp[0], points_interp[1])
ax.contour(X, Y, values_interp.T, levels=[0.5], origin="lower", colors="k", zorder=1)"""

import numpy as _np
import tqdm as _tqdm
import fnc as _fnc

_plt = _fnc.utils.lazy.Import("matplotlib.pyplot")

__all__ = ["eval", "number_evals", "plot_2d", "nan_to_new_value"]

#-----------------------------------------------------------------------------

def _f_multicore(init):
    return init['f'](init['x'], *init['args'], **init['kwargs'])


def _shape(points):
    """Return n_points from points."""
    dim = len(points)
    n_points = _np.zeros(dim, dtype='int')
    for i, item in enumerate(points):
        n_points[i] = _np.size(item)
    return n_points


def _points_grid(n_points, bounds):
    """The points defining the regular grid. The points are linearly spaced
    within boundaries.

    Note
    ----
    shape(points) = (dim, n_points)
    """
    #Number dimensions
    dim = len(n_points)
    #Boundaries of the volume
    b_min, b_max = _np.asarray(bounds).T
    #Linearly spaced points
    points = [[]] * dim
    for i in range(dim):
        points[i] = _np.linspace(b_min[i], b_max[i], n_points[i])
    return points


def _eval_points(points):
    """The points where the function is evaluated.

    Note
    ----
    np.shape(x) = (prod(n_points), dim)
    """
    #Number of dimensions and total number of points
    dim = len(points)
    tot_points = _np.prod(_shape(points))
    #Define Grid
    grid = _np.asarray(_np.meshgrid(*points, indexing='ij'))
    #Points where the function will be evaluated
    x = _np.reshape(grid, (dim, tot_points), order='C').T
    return x


def _values_grid(fx, n_points):
    """Put the evaluations of the function as a regular grid.

    Note
    ----
    np.shape(values) = n_points
    """
    values = _np.reshape(fx, n_points, order='C')
    return values


def _evaluate_function(points, f, args, kwargs, n_cpu, progress):
    """Evaluate function in a regular spaced grid of points covering a
    multidimensional volume."""
    #--------------------------------------------------------------------
    #n_points and total number of points
    n_points = _shape(points)
    tot_points = _np.prod(n_points)

    #Points where the function will be evaluated
    x = _eval_points(points)
    #--------------------------------------------------------------------
    #Evaluation function single cpu
    if n_cpu == 1:
        fx = [[]] * tot_points
        for i, item in enumerate(_tqdm.tqdm(x, disable=not progress,
                                            ncols=78)):
            fx[i] = f(item, *args, **kwargs)
    #--------------------------------------------------------------------
    #Evaluation function n-cpus
    else:
        init = [[]] * tot_points
        for i in range(tot_points):
            init[i] = {'f': f, 'x': x[i], 'args': args, 'kwargs': kwargs}
        fx = _fnc.utils.pool.run(_f_multicore, init, n_cpu, progress)
    #--------------------------------------------------------------------
    #Evaluations as a regular grid
    values = _values_grid(fx, n_points)
    return values

#-----------------------------------------------------------------------------

def eval(n_points, bounds, f, args=None, kwargs=None, n_cpu=1, progress=True):
    """Evaluate a function in a multidimensional volume defined by linearly
    spaced points.

    Parameters
    ----------
    n_points : dim
               (dim_0, dim_1, ...)
        Number of points of each dimension.
    bounds : (dim_inf, dim_sup)
             ((dim_0_inf, dim_0_sup), (dim_1_inf, dim_1_sup), ...)
        Boundaries defining the volume.
    f : function
        Function to evaluate.
    args : list
        List of arguments for f.
    kwargs :
        Dictionary with arguments for f.
    n_cpu : int
        Number of cores.
    progress : bool

    Returns
    -------
    points : numpy.ndarray
    values : numpy.ndarray"""
    #--------------------------------------------------------------------
    #Initialization
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    #Accept int for 1-dimensional case
    if isinstance(n_points, int):
        n_points, bounds = (n_points, ), (bounds, )
    #--------------------------------------------------------------------
    points = _points_grid(n_points, bounds)
    values = _evaluate_function(points, f, args, kwargs, n_cpu, progress)
    #--------------------------------------------------------------------
    #1-dimensional case
    if len(points) == 1:
        return points[0], values
    return points, values

#-----------------------------------------------------------------------------

def number_evals(n_points, bounds, norm_f, args=None, kwargs=None):
    """Determine number of evals.

    Note
    ----
    The norm_f function returns 1 if f is evaluated and 0 if it is not
    evaluated."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    _points, values = eval(n_points,
                           bounds,
                           norm_f,
                           args=args,
                           kwargs=kwargs,
                           n_cpu=1,
                           progress=False)
    n_eval = _np.sum(values)
    return n_eval

#-----------------------------------------------------------------------------

def _test_function(x, a, b=3):
    """2-dimensional function for testing purpose."""
    return x[0] * x[1] * a + x[1] - x[0]**b


def _test_2d(n_points, bounds, f, args=None, kwargs=None, progress=True):
    """Code equivalent to eval in 2 dimensions for testing
    purpose."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    x0 = _np.linspace(bounds[0][0], bounds[0][1], n_points[0])
    x1 = _np.linspace(bounds[1][0], bounds[1][1], n_points[1])
    values = _np.zeros(n_points)

    for i in _tqdm.tqdm(range(n_points[0]), disable=not progress, ncols=78):
        for j in range(n_points[1]):
            x = [x0[i], x1[j]]
            values[i, j] = f(x, *args, **kwargs)

    points = [x0, x1]
    return points, values


def _test_4d(n_points, bounds, f, args=None, kwargs=None, progress=True):
    """Code equivalent to eval in 4 dimensions for testing
    purpose."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    x0 = _np.linspace(bounds[0][0], bounds[0][1], n_points[0])
    x1 = _np.linspace(bounds[1][0], bounds[1][1], n_points[1])
    x2 = _np.linspace(bounds[2][0], bounds[2][1], n_points[2])
    x3 = _np.linspace(bounds[3][0], bounds[3][1], n_points[3])
    values = _np.zeros(n_points)

    for i in _tqdm.tqdm(range(n_points[0]), disable=not progress, ncols=78):
        for j in range(n_points[1]):
            for k in range(n_points[2]):
                for l in range(n_points[3]):
                    x = [x0[i], x1[j], x2[k], x3[l]]
                    values[i, j, k, l] = f(x, *args, **kwargs)

    points = [x0, x1, x2, x3]
    return points, values

#-----------------------------------------------------------------------------

def plot_2d(points,
            values,
            show_points=False,
            number_colors=11,
            exclude_value=None,
            grid=True,
            cmap='gnuplot2',
            fig_size=1,
            vmin=None,
            vmax=None,
            **kwargs):
    """Plot points and values of a two-dimensional grid.

    Note
    ----
    1)  The excluded value is shown in grey."""

    #--------------------------------------------------------
    def limit(x):
        dx = (x[1] - x[0])*0.5
        limits = [x[0] - dx, x[-1] + dx]
        return limits

    def exclude(values, excluded_value):
        """Change excluded_value for np.nan in the values grid."""
        excluded = _np.copy(values, order='C')

        n_points = _np.shape(excluded)
        tot_points = _np.prod(n_points)

        excluded = _np.reshape(excluded, (1, tot_points), order='C')
        excluded[0][excluded[0] == excluded_value] = _np.nan
        excluded = _np.reshape(excluded, n_points, order='C')
        return excluded

    def axis_colorbar(values, number_colors):
        #Set maximum 15 ticks in colorbar
        n = min(number_colors, 15)
        ticks = _np.linspace(_np.min(values), _np.max(values), n)
        vmin, vmax = limit(ticks)
        return vmin, vmax, ticks

    def axis_colorbar(vmin, vmax, values, number_colors):
        #Set maximum 15 ticks in colorbar
        n = min(number_colors, 15)

        if (vmin is None) & (vmax is not None):
            vmin = _np.min(values)
            ticks = _np.linspace(vmin, vmax, n+1)

        elif (vmin is not None) & (vmax is None):
            vmax = _np.max(values)
            ticks = _np.linspace(vmin, vmax, n+1)

        elif (vmin is None) & (vmax is None):
            ticks = _np.linspace(_np.min(values), _np.max(values), n)
            vmin, vmax = limit(ticks)
        else:
            ticks = _np.linspace(vmin, vmax, n+1)

        return vmin, vmax, ticks

    def colormap(cmap, number_colors):
        cmap = _plt.get_cmap(cmap, number_colors)
        cmap.set_bad(color='grey')
        cmap.set_under('grey')
        return cmap

    def set_colorbar(cbar, vmax, ticks):
        #Exponent in base 10
        exp = _np.floor(_np.log10(_np.abs(vmax)))

        #Set label
        if exp == 0.0:
            label = "values"
        else:
            exp_str = str(int(exp))
            label = f"values [$x10^{{{exp_str}}}$]"

        #Set colorbar
        cb = _plt.colorbar(cbar,
                           label=label,
                           ticks=ticks,
                           extend='min',
                           pad=0.02)

        #Set ticks label colorbar
        ticks_labels = [[]] * len(ticks)
        for i, item in enumerate(ticks):
            ticks_labels[i] = f"{item/10**exp:0.1f}"
        cb.ax.set_yticklabels(ticks_labels)

    def plot_image(ax, limit_x, limit_y, values, number_colors, exclude_value,
                   cmap, vmin, vmax, **kwargs):

        vmin, vmax, ticks = axis_colorbar(vmin, vmax, values, number_colors)

        excluded_values = exclude(values, excluded_value=exclude_value)

        cbar = ax.imshow(excluded_values.T,
                         origin='lower',
                         extent=_np.append(limit_x, limit_y),
                         cmap=colormap(cmap, number_colors),
                         zorder=0,
                         vmin=vmin,
                         vmax=vmax,
                         **kwargs)

        set_colorbar(cbar, vmax, ticks)

    def plot_points(ax, points):
        x, y = _eval_points(points).T
        ax.scatter(x, y, c='r', s=10.0)

    def set_axis(ax, limit_x, limit_y):
        ax.set_xlim(limit_x)
        ax.set_ylim(limit_y)
        ax.set_xlabel("points[0]")
        ax.set_ylabel("points[1]")

    #--------------------------------------------------------
    x, y = points
    limit_x = limit(x)
    limit_y = limit(y)
    #--------------------------------------------------------
    #Size box
    if isinstance(fig_size, int | float):
        fig_size = (1.92*fig_size, 1.68*fig_size)
    #--------------------------------------------------------
    _fig, ax = _fnc.plot.figure(fc=fig_size)
    plot_image(ax, limit_x, limit_y, values, number_colors, exclude_value,
               cmap, vmin, vmax, **kwargs)
    if show_points:
        plot_points(ax, points)
        #Limits defined by a rectangle centered at the points
        set_axis(ax, limit_x, limit_y)
    else:
        #Limits defined by points
        set_axis(ax, [x[0], x[-1]], [y[0], y[-1]])
    ax.grid(grid)
    #--------------------------------------------------------
    return ax

#-----------------------------------------------------------------------------

def nan_to_new_value(values, new_value):
    """Change np.nan values to 'new_value'."""
    n_points = _np.shape(values)
    tot_points = _np.prod(n_points)

    values_new = _np.copy(values)

    values_new = _np.reshape(values_new, (1, tot_points), order='C')
    values_new[0][_np.isnan(values_new[0])] = new_value
    values_new = _np.reshape(values_new, n_points, order='C')
    return values_new

#-----------------------------------------------------------------------------
