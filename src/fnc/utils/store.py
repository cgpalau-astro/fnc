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

def _is_extension_opy(file_name):
    return file_name.endswith('.opy')


def _is_extension_zip(file_name):
    return file_name.endswith('.zip')


def _is_extension_opy_zip(file_name):
    return file_name.endswith('.opy.zip')


def _print_size(file_name):
    #Get size stored object in bytes
    opy = _os.path.getsize(file_name)
    print(f"{file_name} : {_hr.memory(opy)}")


def _print_size_zip(x, file_name):
    #Estimate size object in bytes
    opy = len(_pickle.dumps(x))
    #Get size stored and compressed object in bytes
    opy_zip = _os.path.getsize(file_name)

    diff = opy_zip - opy
    sgn = ""
    if diff > 0:
        sgn = "+"
    print(f"{file_name} : {_hr.memory(opy_zip)} ({sgn}{_hr.memory(diff)})")

#-----------------------------------------------------------------------------

def save(file_name, x, gzip=False, verbose=True):
    """Save object (.opy) or compressed object (.opy.zip).

    Parameters
    ----------
    file_name : str
    x : object
    gzip : bool
        Enable Zip compression.
    verbose : bool"""
    #------------------------------------------------------------
    #Add .opy extension to file_name if it is not provided
    if not _is_extension_opy(file_name) | _is_extension_opy_zip(file_name):
        file_name += '.opy'
    #------------------------------------------------------------
    #Non-compressed object
    if gzip is False:
        with open(file_name, 'wb') as file:
            _pickle.dump(x, file)

        if verbose:
            _print_size(file_name)
    #------------------------------------------------------------
    #Compressed object in zip
    elif gzip is True:
        #Add .zip extension
        if not _is_extension_zip(file_name):
            file_name += '.zip'

        with _gzip.GzipFile(file_name, 'wb', compresslevel=9) as file:
            _pickle.dump(x, file)

        if verbose:
            _print_size_zip(x, file_name)
    #------------------------------------------------------------
    else:
        raise TypeError("gzip : bool")

def load(file_name):
    """Load object (.opy) and compressed object (.opy.zip)."""

    #Load .opy file
    if _is_extension_opy(file_name):
        with open(file_name, 'rb') as file:
            return _pickle.load(file)

    #Load .opy.zip file
    elif _is_extension_zip(file_name):
        with _gzip.GzipFile(file_name, 'rb') as file:
            return _pickle.load(file)

    else:
        raise NameError("Python object requires an extension .opy or .zip to be loaded.")

#-----------------------------------------------------------------------------
