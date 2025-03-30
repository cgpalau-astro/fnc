"""Import modules on first access (lazy import).

Note
----
The lazy import causes an overload every time a function is called from the
class Import. Therefore, lazy imports are not recommended when calling fast
functions many times.

For example, accessing np.sin() through 'np = lazy.Import("numpy")' is about
2 times slower than accessing through 'import numpy as np'.

Example
-------
from fnc.utils import lazy

#Set class np with module name 'numpy'.
np = lazy.Import('numpy')

#Numpy is imported and sin(1) is executed.
np.sin(1)

#Numpy is not imported again and sin(1) is executed.
np.sin(1)
"""

import importlib as _importlib

__all__ = ["Import"]

#-----------------------------------------------------------------------------

class Import():
    """Import module on first access.

    Note
    ----
    1)  Accessing functions through this class introduces a significant
        overload."""

    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def __getattr__(self, attr):
        """Import module on first access and return requested attribute."""
        try:
            return getattr(self._module, attr)

        except Exception as exc:
            if self._module is None:
                #Module is no loaded, then load it
                self._module = _importlib.import_module(self._module_name)
            else:
                #Module is loaded, but attribute is not present in the module
                raise exc

        #Return the attribute after loading the module
        return getattr(self._module, attr)

#-----------------------------------------------------------------------------
