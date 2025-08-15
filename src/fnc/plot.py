"""Plot utilities."""

import numpy as _np
import tqdm as _tqdm
import fnc as _fnc

from fnc.utils import lazy as _lazy

_plt = _lazy.Import("matplotlib.pyplot")
_mpl = _lazy.Import("matplotlib")
_shutil = _lazy.Import("shutil")
_scipy = _lazy.Import("scipy")
_sklearn = _lazy.Import("sklearn")
_shapely = _lazy.Import('shapely')
_PIL_Image = _lazy.Import('PIL.Image')

__all__ = ["LATEX_PREAMBLE", "clr", "figure", "aspect_ratio",
           "remove_top_bottom_white_lines", "save_figure",
           "sigma_level", "limits",
           "mixed_scatter",
           "hist", "hist_area",
           "color_line", "polygon"]

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
    'dy': '#b89d12', #Dark yellow
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
        raise RuntimeError("Cannot adjust margins of unsupported layout engine.")

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


def aspect_ratio(ax):
    """Get aspect ratio given the axis of a plot."""

    #Total figure size
    fig_w, fig_h = ax.get_figure().get_size_inches()

    #Axis size on figure
    _, _, w, h = ax.get_position().bounds

    #Ratio of display units
    disp_ratio = (fig_h*h)/(fig_w*w)

    #Ratio of data units
    #Negative over negative because of the order of subtraction
    data_ratio = _np.subtract(*ax.get_ylim())/_np.subtract(*ax.get_xlim())

    return disp_ratio/data_ratio

#-----------------------------------------------------------------------------

def remove_top_bottom_white_lines(image_in, image_out):
    """Remove top and bottom white lines of a PNG image."""

    def is_white_line(line):
        #white = [255, 255, 255, 255]
        return _np.all(line == [255, 255, 255, 255])

    #Open a PNG image
    img = _PIL_Image.open(image_in)

    #Image dimensions
    width, height = img.size

    #From Image to array
    img_arr = _np.asarray(img)

    delete = [False]*height

    #Identify top white lines
    for i in range(height):
        delete[i] = is_white_line(img_arr[i])
        if not delete[i]:
            break

    #Identify bottom white lines
    for i in reversed(range(height)):
        delete[i] = is_white_line(img_arr[i])
        if not delete[i]:
            break

    #Lines to keep
    keep = _np.logical_not(delete)

    #From array to Image eliminating top and bottom white lines
    img_out = _PIL_Image.fromarray(img_arr[keep])

    #Save Image
    img_out.save(image_out)


def save_figure(directory, name, fast=False, remove_white_lines=True):
    """Save figures for papers."""
    if not fast:
        _plt.savefig(f"{name}_300.png", dpi=300)
        _plt.savefig(f"{name}_600.png", dpi=600)

        if remove_white_lines:
            remove_top_bottom_white_lines(f"{name}_300.png", f"{name}_300.png")
            remove_top_bottom_white_lines(f"{name}_600.png", f"{name}_600.png")

        _shutil.copy2(f"{name}_300.png", f"{directory}png_300/{name}_300.png")
        _shutil.copy2(f"{name}_600.png", f"{directory}png_600/{name}_600.png")
    else:
        _plt.savefig(f"{name}_300.png", dpi=300)
        _shutil.copy2(f"{name}_300.png", f"{directory}png_300/{name}_300.png")

#-----------------------------------------------------------------------------

def _minimum(eps, sigma_volume, bins, H):
    if eps[0] <= 0.0:
        return _np.inf

    vol = 0.0
    for i in range(bins):
        for j in range(bins):
            if H[i,j] >= eps[0]:
                vol += H[i,j]
    return _np.abs(vol - sigma_volume)


def sigma_level(x, sigma, n_resample, bins, limits_x=None, limits_y=None, seed=123, verbose=True):
    """Returns data to draw a contourn line that includes 'sigma' volume of a
    distribution of a sample of points 'x'.

    Note
    ----
    1)  When 'n_resample' > 0, the distribution of points is estimated with a
        KDE and a new distribution of points sampling the KDE is generated to
        determine the contourn level.

    2)  This fucntion is equivalent to:
        import seaborn
        vol_sigma = _volume_within_sigma(sigma)
        seaborn.kdeplot(x=x[0], y=x[1], levels=[1.0-vol_sigma])

    3)  This function is faster than seaborn for large samples.

    Example
    --------
    1)  sigma = 1 plots a contourn line including approximately 68 per cent of
        the distribution of points.

    2)  import scipy
        import seaborn
        import numpy as np
        import fnc

        #Definition Gaussian distribution
        mean = [1.0, 1.0]
        cov = [[0.05, 0.0], [0.0, 0.05]]
        norm = scipy.stats.multivariate_normal(mean=mean, cov=cov)

        #Random sample
        N = 1_000
        sample = norm.rvs(size=N, random_state=145).T

        #Plot
        fig, ax = fnc.plot.figure(1, 1)
        ax.scatter(sample[0], sample[1], s=0.1)

        #1-sigma level with seaborn
        vol_sigma = fnc.stats.norm.sigma_level(1)
        seaborn.kdeplot(x=sample[0], y=sample[1], levels=[1.0-vol_sigma])

        #1-sigma level with sigma_level
        n_resample = 10_000_000
        bins = 60
        limits_x = [0.0, 2.0]
        limits_y = [0.2, 1.8]

        #plot_limits(ax, limit_x, limit_y)
        X, Y, H, level = sigma_level(sample, 1.0, n_resample, bins, limits_x, limits_y)
        ax.contour(X, Y, H, levels=[level], colors='r', zorder=10)

    3)  import scipy
        import fnc
        import seaborn
        import numpy as np
        #-----------------------------------------------------------------------------
        mean = [0.0, 0.0]
        scale = 1.3199813663319262
        cov = [[scale**2, 0.0], [0.0, scale**2]]
        norm = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        sample = norm.rvs(10_000, random_state=123).T
        #-----------------------------------------------------------------------------
        fig, ax = fnc.plot.figure(2, 1, fc=(1,2))
        limit = 10.0

        mean = np.mean(sample[0])
        std = np.std(sample[0])
        seaborn.kdeplot(x=sample[0], ax=ax[0], color='r', bw_adjust=1, legend=False, fill=False)
        seaborn.kdeplot(x=sample[0], ax=ax[0], color='r', bw_adjust=1, legend=False, fill=True, clip=(mean - std, mean + std), linewidth=0.0)
        ax[0].set_xlim(-limit, limit)

        ax[1].scatter(sample[0], sample[1], s=0.1, color='b')
        levels = [1 - fnc.stats.norm.sigma_level(3),
                  1 - fnc.stats.norm.sigma_level(2),
                  1 - fnc.stats.norm.sigma_level(1),
                  1]
        seaborn.kdeplot(x=sample[0], y=sample[1], ax=ax[1], fill=False, levels=levels, color="r", alpha=1, bw_adjust=1, linewidths=1.0)
        ax[1].set_xlim(-limit, limit)
        ax[1].set_aspect(1)

        #level = fnc.stats.norm.sigma_level(1)
        #np.sqrt(-2.0*scale**2.0*np.log(1.0-level))"""

    #Print limits
    if limits_x is None:
        limits_x = [_np.min(x[0]), _np.max(x[0])]
        limits_y = [_np.min(x[1]), _np.max(x[1])]

    #Resample with KDE
    if n_resample != 0:
        kde = _scipy.stats.gaussian_kde(x, bw_method="scott")
        x_sample = kde.resample(n_resample, seed)
    else:
        print('No resample')
        x_sample = x

    #Histogram 2D
    h = _np.histogram2d(x=x_sample[0],
                        y=x_sample[1],
                        bins=bins,
                        range=[[limits_x[0], limits_x[1]], [limits_y[0], limits_y[1]]],
                        density=True)
    area = (h[1][1] - h[1][0])*(h[2][1] - h[2][0])
    H = h[0]*_np.abs(area)

    #Limit sigma
    x0 = H.max()/2.0
    sigma_volume = _fnc.stats.norm.sigma_level(sigma)
    res = _scipy.optimize.minimize(_minimum, x0=x0, args=(sigma_volume, bins, H), method='nelder-mead')
    level = res.x[0]

    #Meshgrid
    a = _np.linspace(limits_x[0], limits_x[1], bins)
    b = _np.linspace(limits_y[0], limits_y[1], bins)
    X, Y = _np.meshgrid(a, b)

    if verbose:
        print(f"Limits x : {limits_x}")
        print(f"Limits y : {limits_y}")
        print(f"Sample   : {len(x[0]):_}")
        if n_resample != 0:
            print(f"Resample : {n_resample:_}")
        print(f"Level {sigma}-sigma = {level}\n")

    return X, Y, H.T, level


def limits(ax, limits_x, limits_y):
    left, bottom, width, height = (limits_x[0], limits_y[0], limits_x[1]-limits_x[0], limits_y[1]-limits_y[0])
    rect = _plt.Rectangle((left, bottom), width, height, facecolor="k", alpha=0.25, zorder=0)
    ax.add_patch(rect)

#-----------------------------------------------------------------------------

def mixed_scatter(ax, a, b, color_a, color_b, **kargs):
    """Scatter plot where the points 'a' and 'b' are plotted mixed.

    Example
    -------
    import invi
    import fnc
    import scipy

    cov = [[1.0, 0.0], [0.0, 1.0]]
    norm_a = scipy.stats.multivariate_normal(mean=[1.0, 1.0], cov=cov)
    norm_b = scipy.stats.multivariate_normal(mean=[1.75, 1.75], cov=cov)

    size = 2_000
    a = norm_a.rvs(size=size, random_state=123).T
    b = norm_b.rvs(size=size, random_state=124).T

    fig, ax = fnc.plot.figure(1, 2, fc=(2,1))
    s = 5.0
    ax[0].scatter(a[0], a[1], s=s, c="k")
    ax[0].scatter(b[0], b[1], s=s, c="r")

    invi.plots.general.mixed_scatter(ax[1], a, b, 'k', 'r', s=s)"""
    #-------------------------------------------------------
    def plot(n, ax, a, b, color_a, color_b, **kargs):
        for i in _tqdm.tqdm(range(n), ncols=78):
            ax.scatter(a[0][i], a[1][i], c=color_a, **kargs)
            ax.scatter(b[0][i], b[1][i], c=color_b, **kargs)
    #-------------------------------------------------------
    #Shuffle points
    a[0], a[1] = _sklearn.utils.shuffle(a[0], a[1], random_state=123)
    b[0], b[1] = _sklearn.utils.shuffle(b[0], b[1], random_state=123)

    len_a = len(a[0])
    len_b = len(b[0])

    if len_a == len_b:
        plot(len_a, ax, a, b, color_a, color_b, **kargs)
    elif len_a > len_b:
        ax.scatter(a[0][len_b:], a[1][len_b:], c=color_a, **kargs)
        plot(len_b, ax, a, b, color_a, color_b, **kargs)
    else:
        mixed_scatter(ax, b, a, color_b, color_a, **kargs)

#-----------------------------------------------------------------------------

def hist(ax, counts, bins, **kwargs):
    """Plot histogram given the number of counts and bin limits.

    Note
    ----
    1) counts, bins = np.histogram(x, **kwargs)

    Example
    -------
    import numpy as np
    import scipy
    import fnc

    kwargs = {'bins': 100, 'range': [-0.5, 0.5], 'density': True}

    #Random sample
    norm = scipy.stats.norm(loc=0.0, scale=0.1)
    x = norm.rvs(size=1_000, random_state=111)

    #Plot histograms
    fig, ax = fnc.plot.figure()
    h = ax.hist(x, histtype='step', **kwargs)

    counts, bins = np.histogram(x, **kwargs)
    fnc.plot.hist(ax, counts, bins, color="r", linestyle="--")

    a = np.linspace(-0.5, 0.5, 10_000)
    b = fnc.numeric.eval_hist(a, counts, bins)
    ax.plot(a, b, c='k')"""

    ax.step(bins[1:len(bins)], counts, **kwargs)


def hist_area(ax, x, bins, range, area, **kwargs):
    """Plot histogram with specified area.

    Example
    -------
    import numpy as np
    import fnc

    x = [1, 2, 3, 4, 5, 5, 6]

    bins = 20
    rng = [0, 10]

    fig, ax = fnc.plot.figure(1,1, fc=(1.3,1.3))
    h = ax.hist(x, bins=bins, range=rng, density=True)
    fnc.plot.hist_area(ax, x, bins=bins, range=rng, area=0.8, color="r")"""

    normalized_counts, bins = _np.histogram(x, bins=bins, range=range, density=True)
    hist(ax, normalized_counts*area, bins, **kwargs)

#-----------------------------------------------------------------------------

def color_line(ax, x, y, z=None, cmap=None, norm=None, linewidth=3.0, alpha=1.0):
    """Plot a colored line with coordinates x and y. Optionally specify colors
    in the array z, colormap, norm function and a line width.

    Example
    -------
    import numpy as np
    import matplotlib.pyplot as plt
    import fnc
    import invi

    phi = np.linspace(0.0, 2.0*np.pi, 1_000)
    x = np.sin(phi)
    y = np.cos(phi)

    fig, ax = fnc.plot.figure()
    cb = invi.plots.general.color_line(ax, x, y, z=phi, cmap='turbo', norm=plt.Normalize(0.0, 2.0*np.pi), linewidth=10, alpha=1.0)
    cbar = fig.colorbar(cb, label="$(0, 2\\pi)$", fraction=0.1, pad=0.05, extend=None)"""
    #-----------------------------------------------------------
    def make_segments(x, y, con=1):
        """Create list of line segments from x and y coordinates, in the
        format for _mpl.collections.LineCollection: an array of the form:
        numlines x (points per line) x 2 (x and y) array"""
        points = _np.array([x, y]).T.reshape(-1, 1, 2)
        if con == 1:
            segments = _np.concatenate([points[:-1], points[1:]], axis=1)
        else:
            segments = _np.concatenate([points[:-2],points[1:-1], points[2:]], axis=1)
        return segments
    #-----------------------------------------------------------
    #Default colors equally spaced on [0,1]:
    if z is None:
        z = _np.linspace(0.0, 1.0, len(x))

    #Special case if a single number:
    if not hasattr(z, "__iter__"):  #Check for numerical input
        z = _np.array([z])

    z = _np.asarray(z)

    if cmap is None:
        cmap = _plt.get_cmap('copper')

    if norm is None:
        norm = _plt.Normalize(0.0, 1.0)
    #-----------------------------------------------------------
    kwargs = {'array': z, 'cmap': cmap, 'norm': norm, 'linewidth': linewidth, 'alpha': alpha}

    segments1 = make_segments(x, y, 1)
    lc1 = _mpl.collections.LineCollection(segments1, **kwargs)

    segments2 = make_segments(x, y, 2)
    lc2 = _mpl.collections.LineCollection(segments2, **kwargs)

    ax.plot(x, y, alpha=0.0)
    ax.add_collection(lc1)
    ax.add_collection(lc2)

    return lc1

#-----------------------------------------------------------------------------

def polygon(ax, points_polygon, **kwargs):
    """Plot polygon."""

    polygon = _shapely.geometry.Polygon(shell=points_polygon)

    path = _mpl.path.Path.make_compound_path(_mpl.path.Path(_np.asarray(polygon.exterior.coords)[:, :2]),
                                             *[_mpl.path.Path(_np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors])

    patch = _mpl.patches.PathPatch(path, **kwargs)
    collection = _mpl.collections.PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()

#-----------------------------------------------------------------------------
