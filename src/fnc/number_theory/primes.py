"""Prime number identification, generation, and factorisation.

Note
----
1)  The cryptography package 'rsa' implements more general methods to
    determine whether a number is prime and to generate prime numbers of a
    given length.

2)  The option @_numba.njit(cache=True) reduces the import time."""

import warnings as _warnings
import collections as _collections
import numba as _numba
from . import core as _core

__all__ = [
    "coprime", "is_prime", "sieve_of_eratosthenes", "prime_factorisation",
    "inverse_prime_factorisation", "factorisation_to_exponents"
]

#-----------------------------------------------------------------------------

_large_primes = [
    8_191, 131_071, 524_287, 6_700_417, 2_147_483_647, 999_999_000_001,
    67_280_421_310_721
]

_very_large_primes = [
    170_141_183_460_469_231_731_687_303_715_884_105_727,
    391581 * pow(2, 216193) - 1
]

#-----------------------------------------------------------------------------

def coprime(a, b):
    """Determines whether a and b are coprime, relatively prime or mutually
    prime.

    Parameters
    ----------
    a: int
    b: int

    Returns
    -------
    bool"""
    return _core.gcd(a, b) == 1

@_numba.njit(_numba.bool_(_numba.uint64), cache=True)
def is_prime(number):
    """Determines whether a positive integer is prime.

    Observations
    ------------
    1)  This function is implemented in Numba. It uses numba.uint64 types
        which are limited to 2**64-1 integers.

    Parameters
    ----------
    number: int
        Range: number > 0

    Returns
    -------
    bool"""
    #-------------------------------------------------
    #Numba converts input floats to integers and does
    #not accept negative numbers. It is only needed to
    #check for zero.
    if number == 0:
        raise ValueError("number: int > 0")
    #-------------------------------------------------
    #Two is prime
    if number == 2:
        return True

    #One and even numbers are not primes
    if (number == 1) | (number % 2 == 0):
        return False

    #Check odd numbers such that p <= sqrt(number) or
    #equivalently p**2 <= number.
    p = 3
    while p * p <= number:
        if number % p == 0:
            return False
        p += 2

    return True

#-----------------------------------------------------------------------------

@_numba.njit((_numba.uint64, ), cache=True)
def sieve_of_eratosthenes(n_limit):
    """Lists prime numbers up to a limit number.

    Algorithm
    ---------
    Sieve of Eratosthenes.

    Observations
    ------------
    1)  This algorithm is implemented in Numba. It uses numba.uint64 types
        which are limited to 2**64-1 integers.

    Parameters
    ----------
    n_limit: int
        Range: n_limit >= 0

    Returns
    -------
    list[int]"""
    #----------------------------------------------------------
    if n_limit < 0:
        raise ValueError("n_limit: int > 0")
    #----------------------------------------------------------
    #Assume all numbers within the range are primes
    primes_bool = [True] * (n_limit + 1)

    #Zero and one are not primes
    primes_bool[0] = False
    if n_limit > 0:
        primes_bool[1] = False

    p = 2
    while p * p <= n_limit:
        if primes_bool[p] is True:
            #All multiples of p are not primes
            for i in range(p * p, n_limit + 1, p):
                primes_bool[i] = False
        p += 1

    primes = []
    number = range(n_limit + 1)
    for item_num, item_primes_bool in zip(number, primes_bool):
        if item_primes_bool is True:
            primes.append(item_num)

    return primes

#-----------------------------------------------------------------------------

def prime_factorisation(number, prime_basis):
    """Determines the factors of a prime basis that decompose a positive
    integer number.

    Example
    -------
    number = 4840 = 2^3 * 5^1 * 7^0 * 11^2 -> factors = [2, 2, 2, 5, 11, 11]

    Parameters
    ----------
    number: int
    prime_basis: list[int]

    Returns
    -------
    list[int]"""
    #----------------------------------------------------------------
    if not isinstance(number, int):
        raise TypeError("number: int > 0")
    if number <= 0:
        raise ValueError("number: int > 0")
    #----------------------------------------------------------------
    n = len(prime_basis)
    factors = []
    i = 0
    while (number != 1) & (i < n):
        if number % prime_basis[i] == 0:
            factors.append(prime_basis[i])
            number = number // prime_basis[i]
        else:
            i += 1
    if i == n:
        _warnings.warn(
            "Factorisation failed. "
            "It is needed to extend the basis of primes "
            "to compute the factorisation.",
            stacklevel=3)
        return []
    return factors

def inverse_prime_factorisation(factors):
    """Convert the prime factorisation into its corresponding integer number.
    """

    def prod(x):
        result = 1
        for item in x:
            result = result * item
        return result

    return prod(factors)

def factorisation_to_exponents(factors):
    """Dictionary counting the number of different factors."""
    return dict(_collections.Counter(factors))

#-----------------------------------------------------------------------------
