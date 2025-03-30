"""Continued fraction functions."""

import decimal as _dec
from . import core as _core

#-----------------------------------------------------------------------------

@_core._set_context_precision
def pi(precision):
    r"""Compute Pi to the defined precision. The precesion includes the first
    digit. For example, precision=4 returns 3.142. The last decimal is
    rounded according to the rules specified for the general context.

    Method
    ------
    Newton's method: McLaurin serie of 6*arcsin(x) evaluated at x=1/2:

    \begin{equation}
    \pi = 3 + 3\sum_{n=0}^{\infty} \, \prod_{k=0}^{n} \,
    \frac{ (2k+1)^2 } { 16k^2+40k+24 }
    \end{equation}

    Parameters
    ----------
    precision : int

    Returns
    -------
    decimal.Decimal"""
    #----------------------------------------------
    #Initialisation numerator and denominator.
    num_step = -8
    num = 1

    den_step = -8
    den = 0

    #Term n=-1 .
    term = _dec.Decimal(3)

    pi_ant = _dec.Decimal(0)
    pi_approx = term
    #----------------------------------------------
    #Truncate the serie when the additional terms
    #are smaller than the requested precision.
    while pi_ant != pi_approx:
        num_step = num_step + 8
        num = num + num_step

        den_step = den_step + 32
        den = den + den_step

        term = (term * num) / den

        pi_ant = pi_approx
        pi_approx += term
    #----------------------------------------------
    return pi_approx

@_core._set_context_precision
def e(precision):
    r"""Compute e to the defined precision. The precision includes the first
    digit. For example, precision=4 returns 2.718. The last decimal is
    rounded according to the rules specified for the general context.

    Method
    ------
    Exponential implemented in the decimal package: decimal.Decimal.exp()

    Note
    ----
    E can be calculated by the Taylor series of exp(x) evaluated at x=1:
    \begin{equation}
    e = \sum_{n=0}^{\infty} \, \frac{1} {n!} \,.
    \end{equation}

    The following code is slower than the build-in function:

    >>> #----------------------------------------------
    >>> #Initialisation numerator and denominator.
    >>> k = 1
    >>> one = _dec.Decimal(1)
    >>> e_ant = _dec.Decimal(0)
    >>> e_approx = _dec.Decimal(1)
    >>> #----------------------------------------------
    >>> #Truncate the serie when the additional terms
    >>> #are smaller than the requested precision.
    >>> while e_ant != e_approx:
    >>>     e_ant = e_approx
    >>>     e_approx += one/_math.factorial(k)
    >>>     k += 1
    >>> #----------------------------------------------
    >>> return e_approx

    Parameters
    ----------
    precision : int

    Returns
    -------
    decimal.Decimal"""
    return _dec.Decimal(1).exp()

@_core._set_context_precision
def phi(precision):
    r"""Compute the golden ratio (phi) to the defined precision. The precision
    includes the first digit. For example, precision=4 returns 1.618. The last
    decimal is rounded according to the rules specified for the general
    context.

    \begin{equation}
    \phi = \frac{1+\sqrt{5}}{2}
    \end{equation}

    Method
    ------
    Square root implemented in the decimal package: decimal.Decimal.sqrt()

    Parameters
    ----------
    precision : int

    Returns
    -------
    decimal.Decimal"""
    return (1 + _dec.Decimal(5).sqrt()) / 2

@_core._set_context_precision
def sqrt(precision, x):
    r"""Compute the square root of the number x to the defined precision. The
    precision includes the first digit. For example, precision=4 returns
    1.414. The last decimal is rounded according to the rules specified for
    the general context.

    Method
    ------
    Square root implemented in the decimal package: decimal.Decimal.sqrt()

    Parameters
    ----------
    precision : int
    x : int, float, decimal.Decimal

    Returns
    -------
    decimal.Decimal"""
    return _dec.Decimal(x).sqrt()

#-----------------------------------------------------------------------------
