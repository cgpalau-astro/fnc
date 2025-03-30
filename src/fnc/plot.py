"""Plot utilities."""

import numpy as _np

from fnc.utils import lazy as _lazy

_plt = _lazy.Import("matplotlib.pyplot")
_mpl = _lazy.Import("matplotlib")

__all__ = ["LATEX_PREAMBLE", "clr", "figure"]

#-----------------------------------------------------------------------------

LATEX_PREAMBLE = r"""\usepackage{amsmath}
\usepackage{amssymb}
\newcommand*{\BPRP}{{G_{\rm BP}\!-\!G_{\rm RP}}}
\newcommand*{\Sun}{\odot}
\newcommand*{\pd}[1]{\times\!10^{#1}}""".replace('\n', ' ')

#-----------------------------------------------------------------------------

clr = {
    'r': '#d62728',
    'b': '#0b73bb',
    'o': '#ff7f0e',
    'g': '#2ca02c',
    'p': '#9467bd',
    'br': '#8c564b',
    'm': '#e377c2',
    'gr': '#7f7f7f',
    'y': '#cfb114',
    'c': '#17becf',
    'k': 'k',
    'w': 'w'
}

#-----------------------------------------------------------------------------

def _set_margins(fig, margins):
    """Set figure margins as [left, right, top, bottom] in inches from the
    edges of the figure."""
    left, right, top, bottom = margins

    #Convert to figure coordinates
    right = 1.0 - right
    top = 1.0 - top

    #Get the layout engine and convert to its desired format
    engine = fig.get_layout_engine()
    if isinstance(engine, _mpl.layout_engine.TightLayoutEngine):
        rect = (left, bottom, right, top)
    elif isinstance(engine, _mpl.layout_engine.ConstrainedLayoutEngine):
        rect = (left, bottom, right - left, top - bottom)
    else:
        raise RuntimeError(
            "Cannot adjust margins of unsupported layout engine.")

    #Set and recompute the layout
    engine.set(rect=rect)
    engine.execute(fig)


def figure(nrows=1, ncols=1, fc=(1.4, 1.2), opt=None, latex=False):
    """Create figure and axis.

    Note
    ----
    1)  Size, margins, linewidths, etc. are set for display in a Jupyter
        notebook. They are not intended to be for plotting.

    2)  fig.savefig("name.png", dpi=300)"""
    #-------------------------------------------------
    def cm_to_inch(x):
        return x/2.54
    #-------------------------------------------------
    if opt is None:
        opt = {
            'width': 8.6,
            'height': 8.6/_np.sqrt(2.0),
            'left': 0.1,
            'right': 0.05,
            'top': 0.01,
            'bottom': 0.0,
            'h_pad': 0.0,
            'w_pad': 0.0
        }

    if isinstance(fc, int | float):
        fc = (fc, fc)
    #-------------------------------------------------
    #Latex
    if latex:
        _plt.rc('text', usetex=True)
        _plt.rc('font', family="DejaVu Sans")
        _plt.rc('font', serif="Computer Modern Sans")
        _plt.rc('font', size=7)
        _plt.rc('axes', linewidth=0.8)
        _plt.rc('lines', linewidth=0.8)
        _plt.rc('text.latex', preamble=LATEX_PREAMBLE)
    #-------------------------------------------------
    #Figure and axis
    fig, axis = _plt.subplots(nrows=nrows, ncols=ncols, layout="constrained")

    #Size
    width = cm_to_inch(opt['width'])*fc[0]
    height = cm_to_inch(opt['height'])*fc[1]

    fig.set_size_inches(width, height)
    _set_margins(fig, [opt['left'], opt['right'], opt['top'], opt['bottom']])
    fig.set_constrained_layout_pads(h_pad=opt['h_pad'], w_pad=opt['w_pad'])

    #Axis
    for ax in _np.asarray(axis).ravel():
        ax.minorticks_on()
        ax.grid(linestyle='-', linewidth=0.3)
        ax.ticklabel_format(useMathText=True)
        ax.get_xaxis().get_major_formatter().set_powerlimits((-2, 3))
        ax.get_yaxis().get_major_formatter().set_powerlimits((-2, 3))

    return fig, axis

#-----------------------------------------------------------------------------
