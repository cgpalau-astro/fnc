"""Print system information."""

import os as _os
import sys as _sys
import subprocess as _subprocess
import platform as _platform

import termcolor as _tc
import psutil as _psutil

import fnc.utils.human_readable as _hr

__all__ = ["system"]

#-----------------------------------------------------------------------------

def system():
    print("System Information:")
    print("-------------------")

    CPython = _tc.colored("CPython:", 'light_blue')
    print(f"  {CPython} {_sys.version}")

    os_system = _platform.freedesktop_os_release()['PRETTY_NAME']
    System = _tc.colored("System:", 'light_blue')
    print(f"   {System} {os_system}")

    op_system = _platform.system()
    release = _platform.release()
    machine = _platform.machine()
    Kernel = _tc.colored("Kernel:", 'light_blue')
    print(f"   {Kernel} {op_system} {release}-{machine}")

    cpu_info = _subprocess.check_output("lscpu", shell=True).strip().decode()
    cpu_model = "Unknown"
    for line in cpu_info.split('\n'):
        if 'Model name' in line:
            cpu_model = line[36:]
            break
    CPU_model = _tc.colored("CPU model:", 'light_blue')
    print(f"{CPU_model} {cpu_model[2:]}")
    CPU_count = _tc.colored("CPU count:", 'light_blue')
    print(f"{CPU_count} {_os.cpu_count()}")

    memory = _psutil.virtual_memory()[0]
    Memory = _tc.colored("Memory:", 'light_blue')
    print(f"   {Memory} {_hr.memory(memory)}")

#-----------------------------------------------------------------------------
