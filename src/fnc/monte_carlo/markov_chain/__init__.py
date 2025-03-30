"""Monte Carlo Markov Chain functions.

Example
-------
import scipy
from fnc.monte_carlo import markov_chain as mc

#Definition Gaussian distribution
mean = [1.0, 1.0, 1.0]
cov = [[0.05, 0.0, -0.03], [0.0, 0.05, 0.01], [-0.03, 0.01, 0.05]]
norm = scipy.stats.multivariate_normal

def pdf(x, mean, cov, bounds):
    if mc.within_bounds(x, bounds):
        return norm.pdf(x, mean=mean, cov=cov)
    return 0.0

options = {'seed': 1,
           'number_steps': 20_001,
           'init_steps': 200,
           'x0': (1.8, 1.8, 1.8),
           'std_x0': (0.1, 0.1, 0.1),
           'step_size': (0.05, 0.05, 0.05),
           'bounds': ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0))}

#Run mcmc
walkers = mc.run(options, pdf, args=(mean, cov, options['bounds']), kwargs={}, n_walkers=4, n_cpus=4, progress=True)

#Print diagnosis statistics
mc.diagnosis.print_stats(walkers)

#Diagnosis plots
dim = 0
bounds = options['bounds']

mc.plot.chains(dim, walkers, bounds)

mc.plot.norm_autocorr(dim, walkers)

mc.plot.corner(walkers, bounds, bins=101, aspect='auto', labels=None)"""

from ._metropolis_hastings import *

from . import diagnosis
from . import plot
