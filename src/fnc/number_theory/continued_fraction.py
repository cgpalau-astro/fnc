"""Continued fraction functions.

Example
-------
from fnc.number_theory import continued_fraction as cf
from fnc.number_theory import core
import decimal

x = decimal.Decimal('3.141')

a = cf.continued_fraction(x)
con = cf.convergents(a)
frac = core.int_to_Fraction(con)

#Convergents to decimal.Decimal:
core.Fraction_to_Decimal(15, frac)

#Convergents to float:
core.Fraction_to_float(frac)
"""

import fractions as _fractions
import math as _math
from . import core as _core

__all__ = ["continued_fraction", "convergents", "rational_magnitude"]

#-----------------------------------------------------------------------------

@_core._set_given_context_precision
def continued_fraction(x, n=5):
    r"""Compute the continued fraction expansion (a) of a rational number
    up to the n term:

    a = [a_0: a_1, a_2, ..., a_i, ..., a_n] = a_0 +         1
                                                    -----------------
                                                    a_1 +      1
                                                          -----------
                                                          a_2 +  ...
                                                                -----
                                                                 a_n.

    Example
    -------
    π ≈ [3; 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1] up to n = 14.

    Observations
    ------------
    1)  A fraction is defined with fractions.Fraction(numerator, denominator)
    2)  Due to precision limitations, an irrational number is represented with
        a finite number of decimals. Thus, the input is always a rational
        number, and its continued fraction expansion is finite.
    3)  When a_i = 0 then a_j = 0 for j > i.
    4)  This function does not return the zero terms.

    Parameters
    ----------
    x : int | float | decimal.Decimal | fractions.Fraction
    n : int

    Returns
    -------
    a : list[int]"""
    #---------------------------------------
    if not isinstance(n, int):
        raise TypeError("n: int >= 0")
    if n < 0:
        raise ValueError("n: int >= 0")
    #---------------------------------------
    if isinstance(x, _fractions.Fraction):
        num = x.numerator
        den = x.denominator
    else:
        num = x
        den = 1
    #---------------------------------------
    a = [[]] * (n + 1)
    i = 0
    while (den != 0) & (i <= n):
        q, r = divmod(num, den)
        a[i] = int(q)
        i += 1
        num = den
        den = r
    return a[0:i]

def convergents(a, n=-1):
    """Compute convergent rational numbers given the continued fraction up to
    an n <= len(a).

    Example
    -------
    π ≈ 3, 22/7, 333/106, 355/113, 103993/33102, 104348/33215
    (up to n = 5).

    Parameters
    ----------
    a : list[int]
    n : int
        Term n. If n=-1, it return all terms in a.

    Returns
    -------
    list[int, int]"""
    #-------------------------------------
    if not isinstance(n, int):
        raise TypeError("n: int >= 0 | -1")
    #-------------------------------------------------------------
    def convergents_serie(a, n, output):
        ant_ant, ant = {'num': (0, 1), 'den': (1, 0)}[output]
        serie = [[]] * n
        for i in range(n):
            serie[i] = a[i] * ant + ant_ant
            ant_ant = ant
            ant = serie[i]
        return serie

    #-------------------------------------------------------------
    if (n > len(a)) | (n == -1):
        n = len(a) - 1
    #-------------------------------------------------------------
    #Compute numerator and denominator of the convergents serie:
    num = convergents_serie(a, n + 1, 'num')
    den = convergents_serie(a, n + 1, 'den')
    #-------------------------------------------------------------
    return list(zip(num, den))

#-----------------------------------------------------------------------------

@_core._set_given_context_precision
def rational_magnitude(x, n=5):
    r"""Compute the rational magnitude of a rational number up to n term of
    the continued fraction.

    Note
    ----
    1) The order of the rational magnitude is defined for x between 0 and 1.
    2) The rational magnitude of a number is equal to its inverse.
        π ≈ [3; 7, 15, 1, 292]    up to order 4 -> rational_mag = 4.153 (n=5)
      1/π ≈ [0; 3, 7, 15, 1, 292] up to order 5 -> rational_mag = 4.153 (n=5)

    Example
    -------
    π ≈ [0]                              -> rational_mag = inf   (n=0)
    π ≈ [3]                up to order 0 -> rational_mag = 1.099 (n=1)
    π ≈ [3; 7]             up to order 1 -> rational_mag = 1.609 (n=2)
    π ≈ [3; 7, 15]         up to order 2 -> rational_mag = 2.120 (n=3)
    π ≈ [3; 7, 15, 1]      up to order 3 -> rational_mag = 1.872 (n=4)
    π ≈ [3; 7, 15, 1, 292] up to order 4 -> rational_mag = 4.153 (n=5)

    Parameters
    ----------
    x : int | float | decimal.Decimal | fractions.Fraction
    n : int
        Range: n >= 0

    Returns
    -------
    a : float"""
    #---------------------------------------
    if not isinstance(n, int):
        raise TypeError("n: int >= 0")
    if n < 0:
        raise ValueError("n: int >= 0")
    if n == 0:
        return _math.inf
    #---------------------------------------
    #Compute for x between 0 and 1
    if x > 1:
        x = 1 / x

    #Compute continued fraction
    a = continued_fraction(x, n)

    #It is rational up to the n term
    if len(a) <= n:
        return _math.inf

    #It is irrational up to the n term
    return _math.log(sum(a[1:]) / n)

#-----------------------------------------------------------------------------
