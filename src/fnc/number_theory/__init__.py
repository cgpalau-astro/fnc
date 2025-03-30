"""Number theory functions."""

from fnc.utils import lazy as _lazy
from . import core
from . import continued_fraction
from . import farey
from . import irrationals

primes = _lazy.Import("fnc.number_theory.primes")
