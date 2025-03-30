"""Definition of tests.

Note
----
1)  Run pytest from 'fnc' folder: 'python3.13 -m pytest -Wall -vs --durations=0 tests/tests.py'

    -Wall : show all warnings
    -x : stops after first failure
    -v : verbose output
    -s : prints output from print()
    --durations=0 : Print elapsed time for each test

    -k 'name_test_1 and name_test_2' : Run name_test_1 and name_test_2 only.
    -k 'not name_test_1 and not name_test_2' : Exclude name_test_1 and name_test_2.

2)  Name of the test functions has to start with 'test_'.

3)  Decorator to skip test: import pytest ; @pytest.mark.skip(reason="")"""

#import pytest

#-----------------------------------------------------------------------------

def test_farey():
    from fractions import Fraction
    from fnc.number_theory import farey

    order = 8
    assert farey.number_elements_sequence(order) == 23
    assert farey.sequence(order=order)[6] == Fraction(2, 7)
    assert farey.new_elements_sequence(order=order)[0] == Fraction(1, 8)

#-----------------------------------------------------------------------------

def test_irrationals_continued_fraction():
    import decimal as dec
    import fnc.number_theory.irrationals as irr
    import fnc.number_theory.continued_fraction as cf

    n = 4
    pi = irr.pi(21)
    a = cf.continued_fraction(pi, n=n)
    assert pi == dec.Decimal('3.14159265358979323846')
    assert a == [3, 7, 15, 1, 292]
    assert cf.convergents(a, n=n)[n] == (103993, 33102)

#-----------------------------------------------------------------------------

def test_primes():
    import fnc.number_theory.primes as p

    assert p.is_prime(p._large_primes[6])

    limit_number = 1_000
    se = p.sieve_of_eratosthenes(limit_number)
    assert p.is_prime(se[len(se) - 1])

    number = 60_060
    pf = p.prime_factorisation(number, se)
    assert p.inverse_prime_factorisation(pf) == number

#-----------------------------------------------------------------------------

def test_grid():
    import numpy as np
    import fnc.grid as g

    n_points = [30, 20]
    bounds = ((-2, 2), (0, 4))

    _points_test, values_test = g._test_2d(n_points,
                                           bounds,
                                           g._test_function,
                                           args=(10, ),
                                           kwargs={'b': 5},
                                           progress=False)

    _points, values = g.eval(n_points,
                             bounds,
                             g._test_function,
                             args=(10, ),
                             kwargs={'b': 5},
                             n_cpu=2,
                             progress=False)

    test_value = np.sum(values_test) / np.sum(values)
    assert np.isclose(test_value, 1.0)

#-----------------------------------------------------------------------------

def test_kullback_leibler():
    import numpy as np
    import scipy
    import fnc

    mean_p = [0.02, 0.01]
    cov_p = [[1.0, 0.0], [0.0, 1.0]]

    mean_q = [0.18, 0.71]
    cov_q = [[0.5, 0.23], [0.23, 0.89]]

    p = scipy.stats.multivariate_normal(mean=mean_p, cov=cov_p)
    q = scipy.stats.multivariate_normal(mean=mean_q, cov=cov_q)

    dkl_exact = fnc.stats.dkl.norm(mean_p, cov_p, mean_q, cov_q)
    dkl_estim = fnc.stats.dkl.rvs(p, q, size=400_000, random_state=123)
    assert np.isclose(dkl_estim, dkl_exact, atol=1.0E-3)

#-----------------------------------------------------------------------------

def test_monte_carlo_markov_chain():
    import numpy as np
    from fnc import stats
    from fnc.monte_carlo import markov_chain as mc

    options = {
        'seed': 1,
        'number_steps': 2_001,
        'init_steps': 200,
        'x0': (1.0, ),
        'std_x0': (0.1, ),
        'step_size': (0.03, ),
        'bounds': None
    }

    MEAN_MC = 1.0
    STD_MC = 0.123

    #Run mcmc Gaussian distribution
    walkers = mc.run(options,
                     stats.norm.pdf,
                     kwargs={'loc': MEAN_MC, 'scale': STD_MC},
                     n_walkers=4,
                     n_cpus=4,
                     progress=False)

    sample = mc.get_sample(walkers)

    frac_mean = np.mean(sample) / MEAN_MC
    frac_std = np.std(sample) / STD_MC

    assert np.isclose(frac_mean, 1.0, atol=1.0E-1)
    assert np.isclose(frac_std, 1.0, atol=1.0E-1)

#-----------------------------------------------------------------------------

def test_monte_carlo_integrate():
    import numpy as np
    from fnc.monte_carlo import integrate

    def f(x):
        C = 1.0/5.326011645264725
        return C * np.exp(np.sin(x[0]*x[1]))

    seed = 123
    n_sample = 10_000
    bounds = ((0.0, 2.0), (0.0, 1.5))

    sample, pdf_sample = integrate.uniform(bounds, n_sample, seed)
    integral, std_integral = integrate.run(sample, pdf_sample, f)

    assert np.isclose(integral, 1.0, atol=1E-3)

#-----------------------------------------------------------------------------

def test_pdf_spline():
    import numpy as np
    import scipy
    import fnc

    #Definition distribution
    dist = fnc.stats.DistSpline(fnc.stats.norm.pdf, x_inf=-30.0, x_sup=30.0, n_points=1E5)

    #Parameters normal distribution
    loc = 1.123
    scale = 4.567

    #Test CDF
    x = np.linspace(-25.0, 25.0, 100)
    cdf_ref = scipy.stats.norm.cdf(x, loc, scale)
    cdf_spl = dist.cdf(x, loc=loc, scale=scale)
    assert np.all(np.isclose(cdf_spl/cdf_ref, 1.0, rtol=9E-4))

    #Generation random sample
    size = 10_000
    random_state = 1234
    sample = dist.rvs(size, random_state, loc=loc, scale=scale)

    #Kolmogorov–Smirnov test
    ks = fnc.stats.kolmogorov_smirnov.test(sample, dist.cdf, args=(loc, scale))
    assert fnc.stats.kolmogorov_smirnov.result(sample, ks)

#-----------------------------------------------------------------------------
