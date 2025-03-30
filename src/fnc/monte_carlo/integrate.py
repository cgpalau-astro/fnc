"""Integrate functions using the Monte Carlo technique.

Example
-------
import scipy
import numpy as np
import tqdm
from fnc.monte_carlo import integrate
from fnc import plot
from fnc import grid

def f(x):
    c = 1.0/5.326011645264725
    return c*np.exp(np.sin(x[0]*x[1]))

#-----------------------------------------------------------------------------

seed = 12
n_sample = 20
bounds = ((0.0,2.0), (0.0,1.5))

#Plot function
points, values = grid.eval((201, 201), bounds, f)
grid.plot_2d(points, values, number_colors=101)

#-----------------------------------------------------------------------------

#Numerical integration
def f_num_int(x, y):
    return f([x, y])
int_f, int_f_err = scipy.integrate.nquad(f_num_int, ranges=bounds,
                                         opts={'epsabs':1.0E-12})
print(f"nQuad integral = {int_f:0.4f} ± {int_f_err:0.4f}")

#Uniform sample
sample, pdf_sample = integrate.uniform(bounds, n_sample, seed)
integral, std_integral = integrate.run(sample, pdf_sample, f)
print(f"U-MC integral  = {integral:0.4f} ± {std_integral:0.4f}")

#Truncated Gaussian sample
loc = (1.35, 1.35)
scale = (0.95, 0.95)
sample, pdf_sample = integrate.truncnorm(bounds, loc, scale, n_sample, seed)
integral, std_integral = integrate.run(sample, pdf_sample, f)
print(f"tG-MC integral = {integral:0.4f} ± {std_integral:0.4f}")

#-----------------------------------------------------------------------------
#Std integration

N = 1_000

int_U, std_int_U = np.zeros(N), np.zeros(N)
int_tG, std_int_tG = np.zeros(N), np.zeros(N)

for i in tqdm.tqdm(range(N)):
    sample, pdf_sample = integrate.uniform(bounds, n_sample, i)
    int_U[i], std_int_U[i] = integrate.run(sample, pdf_sample, f)

    sample, pdf_sample = integrate.truncnorm(bounds, loc, scale, n_sample, i)
    int_tG[i], std_int_tG[i] = integrate.run(sample, pdf_sample, f)

fig, ax = plot.figure()
h = ax.hist(int_U, bins=100, range=[0.75, 1.25], density=True,
            histtype="step", label="Uniform")
h = ax.hist(int_tG, bins=100, range=[0.75, 1.25], density=True,
            histtype="step", label="Trunc. Gauss.")
print(f"U-MC std integral  = {np.std(int_U):0.4f}")
print(f"tG-MC std integral = {np.std(int_tG):0.4f}")
ax.legend()
t = ax.set_xlabel("Integral")"""

import numpy as _np
import scipy as _scipy
import tqdm as _tqdm
import fnc.utils.pool as _pool

__all__ = ["run", "uniform", "truncnorm", "optimize_truncnorm"]

#-----------------------------------------------------------------------------

def _f_multicore(init):
    return init['f'](init['x'], *init['args'], **init['kwargs'])

def run(sample,
        pdf_sample,
        f,
        args=None,
        kwargs=None,
        n_cpu=1,
        progress=False):
    """Monte Carlo integration.

    Parameters
    ----------
    sample : array_like
        Random sample following a pdf truncated within defined boundaries.
    pdf_sample : array_like
        Evaluation of the sample on the pdf.
    f : function
        Function f(x, args, kwargs) to integrate.
    n_cpu : int
        Number of cores.
    progress : bool
        Display tqdm progress bar.

    Returns
    -------
    integral : float
        Value of the integral within defined boundaries for the given sample.
    std_integral : float
        Estimated std of the distribution of integrals obtained for many
        samples."""

    #-----------------------------------------------------------------------
    def f_multicore(init):
        return init['f'](init['x'], *init['args'], **init['kwargs'])

    #-----------------------------------------------------------------------
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    n = _np.shape(sample)[1]
    #-----------------------------------------------------------------------
    #Evaluation function single cpu
    if n_cpu == 1:
        fnc = _np.zeros(n)
        for i in _tqdm.tqdm(range(n), disable=not progress, ncols=78):
            fnc[i] = f(sample.T[i], *args, **kwargs)
    #Evaluation function n-cpus
    else:
        init = [[]] * n
        for i in range(n):
            init[i] = {
                'f': f,
                'x': sample.T[i],
                'args': args,
                'kwargs': kwargs
            }
        fnc = _pool.run(f_multicore, init, n_cpu, progress)
    #-----------------------------------------------------------------------
    qn = fnc / pdf_sample

    integral = _np.mean(qn)
    std_integral = _np.std(qn) / _np.sqrt(n)

    return integral, std_integral

#-----------------------------------------------------------------------------

def uniform(bounds, n_sample, seed):
    """Generate random sample following uniform distributions for each
    dimension and evaluate pdf(sample)."""
    #---------------------------------------------------------------------
    #Accept (a,b) for 1-dimensional case
    if _np.shape(_np.asarray(bounds)) == (2, ):
        bounds = (bounds, )
    #---------------------------------------------------------------------
    rng = _np.random.default_rng(seed)
    dim = len(bounds)
    sample = [[]] * dim

    for i in range(dim):
        sample[i] = rng.uniform(bounds[i][0], bounds[i][1], size=n_sample)
    sample = _np.array(sample)

    #Inverse of the n-dimensional volume
    pdf_sample = 1.0 / _np.prod(_np.diff(bounds, 1).T[0])

    return sample, pdf_sample

def truncnorm(bounds, loc, scale, n_sample, seed):
    """Generate random sample following truncated Gaussian distributions for
    each dimension and evaluate pdf(sample)."""

    #------------------------------------------------------------------
    def trunc(bound, loc, scale):
        return (bound[0] - loc) / scale, (bound[1] - loc) / scale

    #------------------------------------------------------------------
    #Accept int or float for 1-dimensional case
    if isinstance(loc, int | float):
        bounds, loc, scale = (bounds, ), (loc, ), (scale, )
    #------------------------------------------------------------------
    dim = len(bounds)
    sample = [[]] * dim
    pdf_sample = [[]] * dim

    for i in range(dim):
        a, b = trunc(bounds[i], loc[i], scale[i])
        t_norm = _scipy.stats.truncnorm(a, b, loc=loc[i], scale=scale[i])
        sample[i] = t_norm.rvs(size=n_sample, random_state=i + seed)
        pdf_sample[i] = t_norm.pdf(sample[i])

    sample = _np.array(sample)
    pdf_sample = _np.array(pdf_sample).T
    pdf_sample = _np.prod(pdf_sample, 1)

    return sample, pdf_sample

#-----------------------------------------------------------------------------

def optimize_truncnorm(bounds, sample, x0):
    """Optimize loc and scale of truncated Gaussian distributions within
    bounds."""

    #----------------------------------------------------------------------
    def trunc(bound, loc, scale):
        return (bound[0] - loc) / scale, (bound[1] - loc) / scale

    def min_loglk(x, i, bounds, sample):
        """Negative Log-Likelihood."""
        loc, scale = x[0], x[1]
        a, b = trunc(bounds[i], loc, scale)
        tnorm = _scipy.stats.truncnorm(a, b, loc=loc, scale=scale)
        return -_np.sum(tnorm.logpdf(sample[i]))

    #----------------------------------------------------------------------
    dim = len(bounds)
    res = [[]] * dim
    for i in range(dim):
        res[i] = _scipy.optimize.minimize(min_loglk,
                                          x0=x0,
                                          bounds=(bounds[i], (None, None)),
                                          args=(i, bounds, sample),
                                          method="Nelder-Mead").x
    res = _np.array(res).T
    loc, scale = res[0], res[1]
    return loc, scale

#-----------------------------------------------------------------------------
