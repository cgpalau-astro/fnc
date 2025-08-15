"""Core functions for the number_theory package."""

import decimal as _dec
import fractions as _frac
import functools as _functools

__all__ = [
    "gcd", "lcm", "modular_multiplicative_inverse", "Fraction_to_int",
    "int_to_Fraction", "Fraction_to_float", "Fraction_to_Decimal"
]

#-----------------------------------------------------------------------------

def gcd(a, b):
    """Greatest Common Divisor of a and b.

    Observations
    ------------
    gcd(a, b) = gcd(|a|,|b|)

    Parameters
    ----------
    a: int
    b: int

    Returns
    -------
    int"""
    while b != 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Least Common Multiple of a and b.

    Parameters
    ----------
    a: int
    b: int

    Returns
    -------
    int"""
    return a * b // gcd(a, b)

def modular_multiplicative_inverse(a, m):
    """Solves the equation:
    x*a = 1 mod m -> x = a**-1 mod m = pow(a, -1, m)

    Parameters
    ----------
    a: int
    m: int

    Returns
    -------
    int"""
    x = pow(a, -1, m)
    return x

#-----------------------------------------------------------------------------

def _set_context_precision(function):
    """Decorator to set the context precision."""

    @_functools.wraps(function)
    def wrapper(*args, **kwargs):
        #--------------------------------------------
        #Obtain precision from the input parameter
        if args:
            precision = args[0]
        else:
            precision = kwargs[list(kwargs)[0]]
        #--------------------------------------------
        #Store general context.
        general_context = _dec.getcontext()

        #Define new context:
        #We compute 2 more decimals than the
        #requested precision to avoid rounding error
        #in the intermediate steps.
        _dec.setcontext(_dec.Context(prec=precision + 2))
        #----------------------------------------------
        output = function(*args, **kwargs)
        #----------------------------------------------
        #Set the requested precision (unary plus
        #applies new precision).
        _dec.setcontext(_dec.Context(prec=precision))
        if isinstance(output, list):
            for i, item in enumerate(output):
                output[i] = +item
        else:
            output = +output
        #----------------------------------------------
        #Re-set general context.
        _dec.setcontext(general_context)
        return output

    return wrapper

def _set_given_context_precision(function):
    """Decorator to set the context precision given by the input."""

    @_functools.wraps(function)
    def wrapper(*args, **kwargs):
        #------------------------------------------------
        #Obtain input number
        if args:
            x = args[0]
        else:
            x = kwargs[list(kwargs)[0]]
        #------------------------------------------------
        if isinstance(x, _dec.Decimal):
            #Store general context.
            general_context = _dec.getcontext()
            #Determines the precision of the input
            precision = len(str(x).replace('.', ''))
            #Define new context:
            _dec.setcontext(_dec.Context(prec=precision))
            #Run function
            output = function(*args, **kwargs)
            #Re-set general context.
            _dec.setcontext(general_context)
        else:
            output = function(*args, **kwargs)
        return output

    return wrapper

#-----------------------------------------------------------------------------

def Fraction_to_int(x):
    """Convert a list of Fractions to a list of tuples of int.

    Parameters
    ----------
    x : list[Fraction]

    Returns
    -------
    list[tuple[int]]"""

    n = len(x)
    num = [[]] * n
    den = [[]] * n
    for i, item in enumerate(x):
        num[i] = item.numerator
        den[i] = item.denominator

    return list(zip(num, den))

def int_to_Fraction(x):
    """Convert a list of tuples of int to a list of Fractions.

    Parameters
    ----------
    x : list[tuple[int]]

    Returns
    -------
    list[Fraction]"""

    n = len(x)
    frac = [[]] * n
    for i, item in enumerate(x):
        num = x[i][0]
        den = x[i][1]
        frac[i] = _frac.Fraction(num, den)
    return frac

def Fraction_to_float(x):
    """Convert a list of Fractions to a list of floats.

    Parameters
    ----------
    x : list[Fraction]

    Returns
    -------
    list[float]"""

    n = len(x)
    flt = [[]] * n
    for i, item in enumerate(x):
        flt[i] = item.numerator / item.denominator

    return flt

@_set_context_precision
def Fraction_to_Decimal(precision, x):
    """Convert a list of rationals to a list of Decimals.

    Parameters
    ----------
    precision : int
    x : list[Fraction]

    Returns
    -------
    list[Decimal]"""

    n = len(x)
    dec = [[]] * n
    for i, item in enumerate(x):
        dec[i] = _dec.Decimal(item.numerator) / _dec.Decimal(item.denominator)

    return dec

#-----------------------------------------------------------------------------
