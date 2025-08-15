"""Metropolis Hastings algorithm and tools."""

import os as _os
import numpy as _np
import tqdm as _tqdm
import fnc.grid as _grid
import fnc.utils.pool as _pool

__all__ = ["within_bounds", "cut_init_steps", "get_sample", "get_last_position", "nan_to_zero",
           "run"]

#-----------------------------------------------------------------------------

def within_bounds(x, bounds):
    """Check whether x is within bounds.
    Note
    ----
    bounds = ((x0_inf, x0_sup), (x1_inf, x1_sup), ...)

    Parameters
    ----------
    x : array_like
    bounds : array_like

    Returns
    -------
    bool
        True if x is within bounds."""
    b_min, b_max = _np.asarray(bounds).T
    within = (x >= b_min) & (x <= b_max)
    return _np.all(within, axis=0)

#-----------------------------------------------------------------------------

def cut_init_steps(walkers, init_steps):
    """Cut off the first init_steps of each walker."""
    n_steps = _np.size(walkers[0][0])
    return walkers[:].T[init_steps:n_steps].T


def get_sample(walkers):
    """Get random sample from the walkers."""
    return _np.hstack(walkers)


def get_last_position(walkers):
    """Get last position of the walkers. It can be used as initial condition
    x0."""
    return walkers[:][:].T[-1].T


def nan_to_zero(values):
    """Change np.nan values to 0.0."""
    n_points = _np.shape(values)
    tot_points = _np.prod(n_points)

    values = _np.reshape(values, (1, tot_points), order='C')
    values[0][_np.isnan(values[0])] = 0.0
    values = _np.reshape(values, n_points, order='C')
    return values

#-----------------------------------------------------------------------------

def _metropolis_hastings(x0, n_steps, step_size, seed, progress, f, args, kwargs):
    """Generate a random sample following a given probability density
    function using the Metropolis Hastings algorithm.

    Parameters
    ----------
    x0 : array_like
        Initial condition.
    n_steps : int
        Number steps of the chain
    step_size : array_like
        Scale (std) of the Gaussian kernel.
    seed : int
    progress : bool
        Print tqdm progress bar.
    f : function
        Probability density function.
    args : array_like
    kwargs : dict

    Returns
    -------
    walker : numpy.ndarray"""
    #-------------------------------------------------------------------
    rng = _np.random.default_rng(seed)
    #-------------------------------------------------------------------
    dim = len(x0)
    walker = _np.zeros((n_steps, dim))
    #-------------------------------------------------------------------
    #For small divisors, numerical error can be minimized as follows:
    #f0 = _np.log(f(x0, *args, **kwargs))
    #...
    #fp = _np.log(f(xp, *args, **kwargs))
    #...
    #if _np.log(rng.random()) <= fp - f0:
    #...
    #-------------------------------------------------------------------
    f0 = f(x0, *args, **kwargs)

    for i in _tqdm.tqdm(range(n_steps), disable=not progress, ncols=78):

        xp = rng.normal(loc=x0, scale=step_size, size=dim)
        fp = f(xp, *args, **kwargs)

        if rng.random() <= fp / f0:
            x0, f0 = xp, fp

        walker[i] = x0
    #-------------------------------------------------------------------
    return walker.T

#-----------------------------------------------------------------------------

def _sampler(init):
    """Initialization sampling algorithm."""
    return _metropolis_hastings(
        x0=init['x0'],
        n_steps=init['number_steps'],
        step_size=init['step_size'],
        seed=init['seed'],
        progress=False,
        f=init['pdf'],
        args=init['args'],
        kwargs=init['kwargs'],
    )

#-----------------------------------------------------------------------------

def _norm_x0(options, n_walkers):
    """Generate initial conditions following a Gaussian distribution and check
    whether they are within bounds if defined."""

    #Initialize parameters
    loc = options['x0']
    scale = options['std_x0']
    dim = len(loc)
    seed = options['seed']
    bounds = options['bounds']

    #Normal random sample
    rng = _np.random.default_rng(seed)
    x0 = rng.normal(loc=loc, scale=scale, size=(n_walkers, dim))

    #No bounds defined
    if bounds is None:
        return x0

    #Check whether x0 is within bounds.
    within = within_bounds(x0, bounds)
    if _np.all(within, axis=0):
        return x0

    out = _np.logical_not(within)
    dim_out = tuple(_np.arange(0, dim, 1)[out])
    raise ValueError(f"Dimensions {dim_out} out of bounds.")

#-----------------------------------------------------------------------------

def run(options, pdf, args=None, kwargs=None, n_walkers=1, n_cpus=None, progress=False):
    """Initialize and run Monte Carlo Markov Chain Metropolis Hastings.

    Note
    ----
    options = {'seed': int,
               'number_steps': int,
               'init_steps': int,
               'x0': array_like,
               'std_x0': array_like,
               'step_size': array_like,
               'bounds': ((dim_0_inf, dim_0_sup),
                          (dim_1_inf, dim_1_sup), ...) | None}"""
    #-------------------------------------------------------
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    #-------------------------------------------------------
    #Add init_steps
    options['number_steps'] += options['init_steps']
    #Different seed for each walker
    init_seed = options['seed']
    seeds = range(init_seed, init_seed + n_walkers, 1)
    #Initial condition
    x0 = _norm_x0(options, n_walkers)
    #-------------------------------------------------------
    #Input parameters for each walker
    init = [[]] * n_walkers
    for i in range(n_walkers):
        init[i] = options.copy()
        init[i]['x0'] = x0[i]
        init[i]['seed'] = seeds[i]
        init[i]['pdf'] = pdf
        init[i]['args'] = args
        init[i]['kwargs'] = kwargs
    #-------------------------------------------------------
    #Number cpus
    if n_cpus is None:
        n_cpus = _os.cpu_count()
    #-------------------------------------------------------
    #Run pool
    walkers = _pool.run(_sampler, init, n_cpus, progress)
    walkers = _np.asarray(walkers)
    #-------------------------------------------------------
    #Cut walkers if init_steps > 0
    walkers = cut_init_steps(walkers, options['init_steps'])
    return walkers

#-----------------------------------------------------------------------------
