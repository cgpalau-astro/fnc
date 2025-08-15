"""Run Bash commands on python scripts.

Example
-------
cmd("echo 1")
cmd(["echo 1", "echo 2"])"""

import subprocess as _subprocess
import sys as _sys

__all__ = ["cmd"]

#-----------------------------------------------------------------------------

def cmd(bash_command, output='return'):
    """Execute bash commands.

    Parameters
    ----------
    bash_command : str | list[str]
    output : {output='return': returns the output of the command.,
              output='stdout': prints on the sys.stdout the output of the
                               command.}

    Returns
    -------
    output : str | NoneType"""
    if output == 'return':
        stdout = _subprocess.PIPE
    elif output == 'stdout':
        stdout = _sys.stdout
    else:
        raise NameError("Available output: {'return', 'stdout'}")

    #Join list bash commands
    if isinstance(bash_command, list):
        bash_command = ';'.join(bash_command)

    with _subprocess.Popen(args=bash_command,
                           stderr=_sys.stderr,
                           stdout=stdout,
                           shell=True,
                           encoding='utf-8') as process:
        output = process.communicate()[0]

    return output

#-----------------------------------------------------------------------------
