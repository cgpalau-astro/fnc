"""Print system information."""

import os as _os
import sys as _sys
import subprocess as _subprocess
import platform as _platform
import urllib.request as _url

import termcolor as _tc
import psutil as _psutil

import fnc.utils.human_readable as _hr

__all__ = ["system", "internet_connection"]

#-----------------------------------------------------------------------------

def system(color='light_blue', file=None, flush=True):
    """Print system information.

    Note
    ----
    1)  When 'color = None' no color is displayed."""

    def log_print(x):
        #'file' and 'flush' are global variables within this function.
        print(x, file=file, flush=flush)

    def colored(x, color):
        if color is None:
            return x
        return _tc.colored(x, color)

    #When output is redirected to a file no color is included
    if file is not None:
        color = None

    log_print("System Information:")
    log_print("-------------------")

    CPython = colored("CPython:", color)
    log_print(f"  {CPython} {_sys.version}")

    os_system = _platform.freedesktop_os_release()['PRETTY_NAME']
    System = colored("System:", color)
    log_print(f"   {System} {os_system}")

    op_system = _platform.system()
    release = _platform.release()
    machine = _platform.machine()
    Kernel = colored("Kernel:", color)
    log_print(f"   {Kernel} {op_system} {release}-{machine}")

    cpu_info = _subprocess.check_output("lscpu", shell=True).strip().decode()
    cpu_model = "Unknown"
    for line in cpu_info.split('\n'):
        if 'Model name' in line:
            cpu_model = line[36:]
            break

    CPU_model = colored("CPU model:", color)
    log_print(f"{CPU_model} {cpu_model[2:]}")
    CPU_count = colored("CPU count:", color)
    log_print(f"{CPU_count} {_os.cpu_count()}")

    memory = _psutil.virtual_memory()[0]
    Memory = colored("Memory:", color)
    log_print(f"   {Memory} {_hr.memory(memory)}")

#-----------------------------------------------------------------------------

def internet_connection(url='http://python.org/', timeout=5.0, verbose=False):
    """Check internet connection to the given 'url' during 'timeout' seconds."""
    try:
        _url.urlopen(url, timeout=timeout)
        return True
    except Exception as exc:
        if verbose:
            print(f"No internet connection to {url}: {exc}")
        return False

#-----------------------------------------------------------------------------
