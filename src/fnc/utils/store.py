"""Save and load python objects.

Note
----
1)  Python objects are stored with extension .opy. The extension .zip is added
    (.opy.zip) when they are compressed using gzip.
2)  If the python object is a class, it is necessary to define the class
    before loading it."""

import pickle as _pickle
import gzip as _gzip
import os as _os
from . import human_readable as _hr

__all__ = ["save", "load"]

#-----------------------------------------------------------------------------

def _is_extension_opy(name):
    return name.endswith('.opy')

def _is_extension_zip(name):
    return name.endswith('.zip')

def _is_extension_opy_zip(name):
    return name.endswith('.opy.zip')

def _print_size(name):
    #Get size stored object in bytes
    opy = _os.path.getsize(name)
    print(f"{name} : {_hr.memory(opy)}")

def _print_size_zip(x, name):
    #Estimate size object in bytes
    opy = len(_pickle.dumps(x))
    #Get size stored and compressed object in bytes
    opy_zip = _os.path.getsize(name)

    diff = opy_zip - opy
    sgn = ""
    if diff > 0:
        sgn = "+"
    print(f"{name} : {_hr.memory(opy_zip)} ({sgn}{_hr.memory(diff)})")

#-----------------------------------------------------------------------------

def save(name, x, gzip=False, verbose=True):
    """Save object (.opy) or compressed object (.opy.zip).

    Parameters
    ----------
    name : str
    x : object
    gzip : bool
        Enable Zip compression.
    verbose : bool"""
    #------------------------------------------------------------
    #Add .opy extension to name if it is not provided
    if not _is_extension_opy(name) | _is_extension_opy_zip(name):
        name += '.opy'
    #------------------------------------------------------------
    #Non-compressed object
    if gzip is False:
        with open(name, 'wb') as file:
            _pickle.dump(x, file)

        if verbose:
            _print_size(name)
    #------------------------------------------------------------
    #Compressed object in zip
    elif gzip is True:
        #Add .zip extension
        if not _is_extension_zip(name):
            name += '.zip'

        with _gzip.GzipFile(name, 'wb', compresslevel=9) as file:
            _pickle.dump(x, file)

        if verbose:
            _print_size_zip(x, name)
    #------------------------------------------------------------
    else:
        raise TypeError("gzip : bool")

def load(name):
    """Load object (.opy) and compressed object (.opy.zip)."""

    #Load .opy file
    if _is_extension_opy(name):
        with open(name, 'rb') as file:
            return _pickle.load(file)

    #Load .opy.zip file
    elif _is_extension_zip(name):
        with _gzip.GzipFile(name, 'rb') as file:
            return _pickle.load(file)

    else:
        raise NameError(
            "Python object requires an extension .opy or .zip to be loaded.")

#-----------------------------------------------------------------------------
