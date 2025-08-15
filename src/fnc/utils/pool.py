"""Pool execution."""

import multiprocessing as _multiprocessing
import tqdm.contrib.concurrent as _tqdm_concurrent
import tqdm as _tqdm

__all__ = ["run", "run_multi"]

#-----------------------------------------------------------------------------

def run(function, init, n_cpu, progress=True):
    """Run pool using concurrent.futures.ProcessPoolExecutor and display tqdm
    progress bar.

    Parameters
    ----------
    function : func
    init : dict, class
        Initialization parameters for the function.
    n_cpu : int
    progress : bool

    Note
    ----
    1)  https://tqdm.github.io/docs/contrib.concurrent/"""

    output = _tqdm_concurrent.process_map(function,
                                          init,
                                          max_workers=n_cpu,
                                          chunksize=1,
                                          tqdm_class=_tqdm.tqdm,
                                          disable=not progress,
                                          smoothing=0.0,
                                          ncols=78)
    return output

def run_multi(function, init, n_cpu):
    """Run pool using multiprocessing.Pool."""
    with _multiprocessing.Pool(n_cpu) as pool:
        output = pool.map(function, init)
    return output

#-----------------------------------------------------------------------------
