"""Obtain type of objects, and input/output functions."""

import functools as _functools
import numpy as _np

__all__ = ["get"]

#-----------------------------------------------------------------------------

def get(x):
    """Obtain type and sub-types of an object."""

    #------------------------------------------------------------
    def remove_duplicates(x):
        #A dictionary cannot accept repeated keys like a set()
        #but it maintains the order of the elements. This code is
        #equivalent to:
        #def remove_duplicates(x):
        #   res = []
        #   for i in x:
        #       if i not in res:
        #           res.append(i)
        #   return res
        return list(dict.fromkeys(x))

    def get_subtype(x):
        t = [[]] * len(x)
        for i, item in enumerate(x):
            t[i] = get(item)
        return remove_duplicates(t)

    def get_subtype_dict(x):
        t = [[]] * len(x)
        for i, item in enumerate(x.items()):
            t[i] = get(item[1])
        return remove_duplicates(t)

    #------------------------------------------------------------
    if isinstance(x, list):
        return f"list{get_subtype(x)}".replace("\'", "")
    if isinstance(x, tuple):
        return f"tuple{get_subtype(x)}".replace("\'", "")
    if isinstance(x, set):
        return f"set{get_subtype(x)}".replace("\'", "")
    if isinstance(x, dict):
        return f"dict{get_subtype_dict(x)}".replace("\'", "")
    if isinstance(x, _np.ndarray):
        return f"numpy.ndarray{get_subtype(x)}".replace("\'", "")
    return f"{type(x).__name__}"

#-----------------------------------------------------------------------------

def _print_types(function):
    """Decorator that prints input and output types of a function."""

    #-----------------------------------------------------
    def define_pad(args, kwargs):
        length = [[]] * len(kwargs)
        for i, kwarg in enumerate(list(kwargs)):
            length[i] = len(str(kwarg))

        if length == []:
            pad = len(str(len(args))) + 1
        else:
            pad = max(length) + 1
        return pad

    #-----------------------------------------------------
    @_functools.wraps(function)
    def wrapper(*args, **kwargs):

        pad = define_pad(args, kwargs)

        for i, arg in enumerate(args):
            print(f"{i:<{pad}}: {get(arg)}")

        for i in range(len(kwargs)):
            entry = list(kwargs)[i]
            print(f"{entry:<{pad}}: {get(kwargs[entry])}")

        output = function(*args, **kwargs)
        print(f"{'>':<{pad}}: {get(output)}")

        return output
        #-------------------------------------------------

    return wrapper

#-----------------------------------------------------------------------------
