"""Farey sequence."""

import fractions as _fractions

__all__ = ["number_elements_sequence", "sequence", "new_elements_sequence"]

#-----------------------------------------------------------------------------

def _euler_phi(n):
    """Eulers's totient or phi function.

    Note
    ----
    By definition, if n is a positive integer, then phi(n) is the number of
    integers k in the range 1 <= k <= n for which gcd(n, k) = 1.

    Parameters
    ----------
    n: int
        Range: n > 0

    Returns
    -------
    int"""
    #-------------------------------------------
    if not isinstance(n, int):
        raise TypeError("n: int > 0")
    if n <= 0:
        raise ValueError("n: int > 0")
    #-------------------------------------------
    phi = n
    for p in range(2, int(n**0.5) + 1):
        if not n % p:
            phi -= phi // p
            while not n % p:
                n //= p
    #n is prime if n > 1
    if n > 1:
        phi -= phi // n
    return phi

def number_elements_sequence(order):
    """Number of elements in a Farey sequence of given order.

    Parameters
    ----------
    order: int
        Range: order > 0

    Returns
    -------
    int"""
    #-------------------------------------
    if not isinstance(order, int):
        raise TypeError("order: int > 0")
    if order == 0:
        raise ValueError("order: int > 0")
    #-------------------------------------
    n = 1
    for k in range(1, order + 1):
        n += _euler_phi(k)
    return n

def sequence(order):
    """Compute Farey sequence of given order.

    Parameters
    ----------
    order: int
        Range: order > 0

    Returns
    -------
    list[Fraction]"""
    #----------------------------------------------
    if not isinstance(order, int):
        raise TypeError("order: int > 0")
    if order == 0:
        raise ValueError("order: int > 0")
    #----------------------------------------------
    #First element (0/1)
    sequence = [_fractions.Fraction(0, 1)]

    for den in range(1, order + 1):
        for num in range(1, den + 1):
            sequence.append(_fractions.Fraction(num, den))
    #----------------------------------------------
    #Define a set() to eliminate duplicates
    return sorted(set(sequence))

def new_elements_sequence(order):
    """Return elements of the Farey sequence of given order not present in the
    previous order sequences.

    Parameters
    ----------
    order: int
        Range: order > 0

    Returns
    -------
    list[Fraction]"""
    #--------------------------------------------
    if not isinstance(order, int):
        raise TypeError("order: int > 0")
    if order <= 0:
        raise ValueError("order: int > 0")
    if order == 1:
        return sequence(order)
    #--------------------------------------------
    sequence_0 = sequence(order - 1)
    sequence_1 = sequence(order)
    non_repeated = set(sequence_1) - set(sequence_0)
    #--------------------------------------------
    return sorted(non_repeated)

#-----------------------------------------------------------------------------
