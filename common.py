import collections
import IPython.display as IPd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sns
import six
import time

from scipy.stats import gaussian_kde
from seaborn.utils import despine
from svgutils.compose import Unit


try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections


def unit(val, to='px'):
    assert type(val) == str
    return Unit(val).to(to).value


def unit2px(val):
    return unit(val, to='px')


def unit2inches(val):
    return unit(val, to='pt')


def hex2rgb(hex):
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def rgb2hex(RGB):
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
                        "{0:x}".format(v) for v in RGB])


def get_colors(normalize=True):
    hex2rgb = lambda h: tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # RGB colors in [0, 255]
    col = {}

    #parameters and corresponding data
    col['POSTERIOR1']   = hex2rgb('2E7FB6')
    col['POSTERIOR2']   = hex2rgb('5DB784')
    col['POSTERIOR3']   = hex2rgb('EDE249')
    col['CONSISTENT1']  = hex2rgb('8D62BC')
    col['CONSISTENT2']  = hex2rgb('AF99EF')
    col['INCONSISTENT'] = hex2rgb('D73789')
    col['GT']           = hex2rgb('30C05D')

    # methods
    col['SNPE']         = hex2rgb('2E7FE8')
    col['MCMC']         = hex2rgb('FFDF50')
    col['SMC']          = hex2rgb('FC823E')

    # rarely used
    col['PRIOR']        = hex2rgb('2EDEE8')

    # Convert to RGB colors in [0, 1]
    if normalize:
        for k, v in col.items():
            col[k] = tuple([i/255 for i in v])

    return col


col = get_colors()


def probs2contours(probs, levels):
    """Takes an array of probabilities and produces an array of contours at specified percentile levels

    Parameters
    ----------
    probs : array
        Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    levels : list
        Percentile levels, have to be in [0.0, 1.0]

    Return
    ------
    Array of same shape as probs with percentile labels
    """
    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def plot_pdf(pdf1, lims=None, pdf2=None, gt=None, contours=False, levels=(0.68, 0.95),
             resolution=500, labels_params=None, ticks=False, diag_only=False,
             diag_only_cols=4, diag_only_rows=4, figsize=(5, 5), fontscale=1,
             partial=False, partial_dots=False, samples=None,
             sam1=None, sam2=None, sam3=None, samms=5,
             col1='k', col2='b', col3='g', col4='r', col5='k', col6='k', col7='k',
             axis_off=False):
    """Plots marginals of a pdf, for each variable and pair of variables.

    Parameters
    ----------
    pdf1 : object
    lims : array
    pdf2 : object (or None)
        If not none, visualizes pairwise marginals for second pdf on lower diagonal
    contours : bool
    levels : tuple
        For contours
    resolution
    labels_params : array of strings
    ticks: bool
        If True, includes ticks in plots
    diag_only : bool
    diag_only_cols : int
        Number of grid columns if only the diagonal is plotted
    diag_only_rows : int
        Number of grid rows if only the diagonal is plotted
    fontscale: int
    partial: bool
        If True, plots partial posterior with at the most 3 parameters.
        Only available if `diag_only` is False
    partial_dots : bool
        Plots ... next to plot
    samples: array
        If given, samples of a distribution are plotted along `pdf`.
        If given, `pdf` is plotted with default `levels` (0.68, 0.95), if provided `levels` is None.
        If given, `lims` is overwritten and taken to be the respective
        limits of the samples in each dimension.
    sam1 - sam3:
        Points to mark in plot
    col1 : str
        color 1 (samples)
    col2 : str
        color 2 (pdf1)
    col3 : str
        color 3 (pdf2)
    col4 : str
        color 4 (gt)
    col5 : str
        color 5 (sam1)
    col6 : str
        color 6 (sam2)
    col7 : str
        color 7 (sam3)
    """

    pdfs = (pdf1, pdf2)
    colrs = (col2, col3)

    if not (pdf1 is None or pdf2 is None):
        assert pdf1.ndim == pdf2.ndim

    if samples is not None:
        #contours = True
        if levels is None:
            levels = (0.68, 0.95)

    if samples is not None and lims is None:
        lims_min = np.min(samples, axis=1)
        lims_max = np.max(samples, axis=1)
        lims = np.asarray(lims)
        lims = np.concatenate(
            (lims_min.reshape(-1, 1), lims_max.reshape(-1, 1)), axis=1)
    else:
        lims = np.asarray(lims)
        lims = np.tile(lims, [pdf1.ndim, 1]) if lims.ndim == 1 else lims

    if pdf1.ndim == 1:

        fig, ax = plt.subplots(1, 1, facecolor='white', figsize=figsize)

        if samples is not None:
            ax.hist(samples[i, :], bins=100, normed=True,
                    color=col1,
                    edgecolor=col1)

        xx = np.linspace(lims[0, 0], lims[0, 1], resolution)

        for pdf, col in zip(pdfs, col):
            if pdf is not None:
                pp = pdf.eval(xx[:, np.newaxis], log=False)
                ax.plot(xx, pp, color=col)
        ax.set_xlim(lims[0])
        ax.set_ylim([0, ax.get_ylim()[1]])
        if gt is not None:
            ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

        if ticks:
            ax.get_yaxis().set_tick_params(which='both', direction='out')
            ax.get_xaxis().set_tick_params(which='both', direction='out')
            ax.set_xticks(np.linspace(lims[0, 0], lims[0, 1], 2))
            ax.set_yticks(np.linspace(min(pp), max(pp), 2))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        else:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    else:

        if not diag_only:
            if partial:
                rows = min(3, pdf1.ndim)
                cols = min(3, pdf1.ndim)
            else:
                rows = pdf1.ndim
                cols = pdf1.ndim
        else:
            cols = diag_only_cols
            rows = diag_only_rows
            r = 0
            c = -1

        fig, ax = plt.subplots(rows, cols, facecolor='white', figsize=figsize)
        ax = ax.reshape(rows, cols)

        for i in range(rows):
            for j in range(cols):

                if i == j:
                    if samples is not None:
                        ax[i, j].hist(samples[i, :], bins=100, normed=True,
                                      color=col1,
                                      edgecolor=col1)
                    xx = np.linspace(lims[i, 0], lims[i, 1], resolution)

                    for pdf, col in zip(pdfs, colrs):
                        if pdf is not None:
                            pp = pdf.eval(xx, ii=[i], log=False)

                            if diag_only:
                                c += 1
                            else:
                                r = i
                                c = j

                    for pdf, col in zip(pdfs, colrs):
                        if pdf is not None:
                            pp = pdf.eval(xx, ii=[i], log=False)
                            ax[r, c].plot(xx, pp, color=col, clip_on=False)

                    ax[r, c].set_xlim(lims[i])
                    ax[r, c].set_ylim([0, ax[r, c].get_ylim()[1]])

                    if gt is not None:
                        ax[r, c].vlines(
                            gt[i], 0, ax[r, c].get_ylim()[1], color=col4)

                    if sam1 is not None:
                        ax[r, c].vlines(
                            sam1[i], 0, ax[r, c].get_ylim()[1], color=col5)

                    if sam2 is not None:
                        ax[r, c].vlines(
                            sam2[i], 0, ax[r, c].get_ylim()[1], color=col6)

                    if sam3 is not None:
                        ax[r, c].vlines(
                            sam3[i], 0, ax[r, c].get_ylim()[1], color=col7)

                    if ticks:
                        ax[r, c].get_yaxis().set_tick_params(
                            which='both', direction='out')
                        ax[r, c].get_xaxis().set_tick_params(
                            which='both', direction='out')
                        ax[r, c].set_xticks(np.linspace(
                            lims[i, 0]+0.15*np.abs(lims[i, 0]-lims[j, 1]), lims[j, 1]-0.15*np.abs(lims[i, 0]-lims[j, 1]), 2))
                        ax[r, c].set_yticks(np.linspace(0+0.15*np.abs(0-max(pp)), max(pp)-0.15*np.abs(0-max(pp)), 2))
                        ax[r, c].xaxis.set_major_formatter(
                            mpl.ticker.FormatStrFormatter('%.1f'))
                        ax[r, c].yaxis.set_major_formatter(
                            mpl.ticker.FormatStrFormatter('%.1f'))
                    else:
                        ax[r, c].get_xaxis().set_ticks([])
                        ax[r, c].get_yaxis().set_ticks([])

                    if labels_params is not None:
                        ax[r, c].set_xlabel(
                            labels_params[i])
                    else:
                        ax[r, c].set_xlabel([])

                    x0, x1 = ax[r, c].get_xlim()
                    y0, y1 = ax[r, c].get_ylim()
                    ax[r, c].set_aspect((x1 - x0) / (y1 - y0))

                    if partial_dots and i == rows - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...')
                        plt.text(x1 + (x1 - x0) / 8.4, y0 - (y1 - y0) /
                                 6., '...', rotation=-45)

                else:
                    if diag_only:
                        continue

                    if i < j:
                        pdf = pdfs[0]
                    else:
                        pdf = pdfs[1]

                    if pdf is None:
                        ax[i, j].get_yaxis().set_visible(False)
                        ax[i, j].get_xaxis().set_visible(False)
                        ax[i, j].set_axis_off()
                        continue

                    if samples is not None:
                        H, xedges, yedges = np.histogram2d(
                            samples[i, :], samples[j, :], bins=30, range=[
                            [lims[i, 0], lims[i, 1]], [lims[j, 0], lims[j, 1]]], normed=True)
                        ax[i, j].imshow(H, origin='lower', extent=[
                                        yedges[0], yedges[-1], xedges[0], xedges[-1]])

                    xx = np.linspace(lims[i, 0], lims[i, 1], resolution)
                    yy = np.linspace(lims[j, 0], lims[j, 1], resolution)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate(
                        [X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                    pp = pdf.eval(xy, ii=[i, j], log=False)
                    pp = pp.reshape(list(X.shape))
                    if contours:
                        contour_cols = (col2, col2)
                        if i > j:
                            contour_cols = (col3, col3)
                        ax[i, j].contour(Y, X, probs2contours(
                            pp, levels), levels, colors=contour_cols)
                    else:
                        ax[i, j].imshow(pp.T, origin='lower',
                                        extent=[lims[j, 0], lims[j, 1], lims[i, 0], lims[i, 1]],
                                        aspect='auto', interpolation='none')
                    ax[i, j].set_xlim(lims[j])
                    ax[i, j].set_ylim(lims[i])

                    if gt is not None:
                        ax[i, j].plot(gt[j], gt[i], '.', color=col4, markersize=samms,
                                      markeredgewidth=0.0)

                    if sam1 is not None:
                        ax[i, j].plot(sam1[j], sam1[i], marker='.', color=col5, markersize=samms,
                                      markeredgewidth=0.0)

                    if sam2 is not None:
                        ax[i, j].plot(sam2[j], sam2[i], marker='.', color=col6, markersize=samms,
                                      markeredgewidth=0.0)

                    if sam3 is not None:
                        ax[i, j].plot(sam3[j], sam3[i], marker='.', color=col7, markersize=samms,
                                      markeredgewidth=0.0)

                    ax[i, j].get_xaxis().set_ticks([])
                    ax[i, j].get_yaxis().set_ticks([])
                    ax[i, j].set_axis_off()

                    x0, x1 = ax[i, j].get_xlim()
                    y0, y1 = ax[i, j].get_ylim()
                    ax[i, j].set_aspect((x1 - x0) / (y1 - y0))

                    if partial_dots and j == cols - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...')

                if diag_only and c == cols - 1:
                    c = -1
                    r += 1

                if axis_off:
                    ax[i, j].set_axis_off()

    return fig, ax


def img(img):
    IPd.display(IPd.HTML('<img src="{}?modified={}" / >'.format(img, time.time())))


svg = img



def _update(d, u):
    # https://stackoverflow.com/a/3233356
    for k, v in six.iteritems(u):
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = _update(dv, v)
        else:
            d[k] = v
    return d


def _format_axis(ax, xhide=True, yhide=True, xlabel='', ylabel='',
        tickformatter=None):
    for loc in ['right', 'top', 'left', 'bottom']:
        ax.spines[loc].set_visible(False)
    if xhide:
        ax.set_xlabel('')
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(labelbottom=False)
    if yhide:
        ax.set_ylabel('')
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_tick_params(labelleft=False)
    if not xhide:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(labelbottom=True)
        if tickformatter is not None:
            ax.xaxis.set_major_formatter(tickformatter)
        ax.spines['bottom'].set_visible(True)
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines['left'].set_visible(True)
    return ax


def samples_nd(samples, points=[], **kwargs):
    """Plot samples and points

    See `opts` below for available keyword arguments.
    """
    opts = {
        # what to plot on triagonal and diagonal subplots
        'upper': 'hist',   # hist/scatter/None/cond
        'diag': 'hist',    # hist/None/cond
        #'lower': None,     # hist/scatter/None  # TODO: implement

        # title and legend
        'title': None,
        'legend': False,

        # labels
        'labels': [],         # for dimensions
        'labels_points': [],  # for points
        'labels_samples': [], # for samples
        'labelpad': None,     # (int or None). If not None, the labels will be shifted downwards by labelpad

        # colors
        'samples_colors': plt.rcParams['axes.prop_cycle'].by_key()['color'],
        'points_colors': plt.rcParams['axes.prop_cycle'].by_key()['color'],

        # subset
        'subset': None,

        # conditional posterior requires condition and pdf1
        'pdfs': None,
        'condition': None,

        # axes limits
        'limits': [],

        # ticks
        'ticks': [],
        'tickformatter': mpl.ticker.FormatStrFormatter('%g'),
        'tick_labels': None,
        'tick_labelpad': None, # (None or int). If not None, the ticklabels will be shifted downwards by tick_labelpad

        # options for hist
        'hist_diag': {
            'alpha': 1.,
            'bins': 25,
            'density': False,
            'histtype': 'step'
        },
        'hist_offdiag': {
            #'edgecolor': 'none',
            #'linewidth': 0.0,
            'bins': 25,
        },

        # options for kde
        'kde_diag': {
            'bw_method': 'scott',
            'bins': 100,
            'color': 'black'
        },
        'kde_offdiag': {
            'bw_method': 'scott',
            'bins': 25
        },

        # options for contour
        'contour_offdiag': {
            'levels': [0.68],
            'percentile': True
        },

        # options for scatter
        'scatter_offdiag': {
            'alpha': 0.5,
            'edgecolor': 'none',
            'rasterized': False,
        },

        # options for plot
        'plot_offdiag': {},

        # formatting points (scale, markers)
        'points_diag': {
        },
        'points_offdiag': {
            'marker': '.',
            'markersize': 20,
        },

        # matplotlib style
        'style': os.path.join(os.path.dirname(__file__), '.matplotlibrc'),

        # other options
        'fig_size': (10, 10),
        'fig_bg_colors':
            {'upper': None,
             'diag': None,
             'lower': None},
        'fig_subplots_adjust': {
            'top': 0.9,
        },
        'subplots': {
        },
        'despine': {
            'offset': 5,
        },
        'title_format': {
            'fontsize': 16
        },
    }
    # TODO: add color map support
    # TODO: automatically determine good bin sizes for histograms
    # TODO: get rid of seaborn dependency for despine
    # TODO: add legend (if legend is True)

    samples_nd.defaults = opts.copy()
    opts = _update(opts, kwargs)

    # Prepare samples
    if type(samples) != list:
        samples = [samples]

    # Prepare points
    if type(points) != list:
        points = [points]
    points = [np.atleast_2d(p) for p in points]

    # Dimensions
    dim = samples[0].shape[1]
    num_samples = samples[0].shape[0]

    # TODO: add asserts checking compatiblity of dimensions

    # Prepare labels
    if opts['labels'] == [] or opts['labels'] is None:
        labels_dim = ['dim {}'.format(i+1) for i in range(dim)]
    else:
        labels_dim = opts['labels']

    # Prepare limits
    if opts['limits'] == [] or opts['limits'] is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(opts['limits']) == 1:
            limits = [opts['limits'][0] for _ in range(dim)]
        else:
            limits = opts['limits']

    # Prepare ticks
    if opts['ticks'] == [] or opts['ticks'] is None:
        ticks = None
    else:
        if len(opts['ticks']) == 1:
            ticks = [opts['ticks'][0] for _ in range(dim)]
        else:
            ticks = opts['ticks']

    # Prepare diag/upper/lower
    if type(opts['diag']) is not list:
        opts['diag'] = [opts['diag'] for _ in range(len(samples))]
    if type(opts['upper']) is not list:
        opts['upper'] = [opts['upper'] for _ in range(len(samples))]
    #if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts['lower'] = None

    # Style
    if opts['style'] in ['dark', 'light']:
        style = os.path.join(
            os.path.dirname(__file__),
            'matplotlib_{}.style'.format(opts['style']))
    else:
        style = opts['style'];

    # Apply custom style as context
    with mpl.rc_context(fname=style):

        # Figure out if we subset the plot
        subset = opts['subset']
        if subset is None:
            rows = cols = dim
            subset = [i for i in range(dim)]
        else:
            if type(subset) == int:
                subset = [subset]
            elif type(subset) == list:
                pass
            else:
                raise NotImplementedError
            rows = cols = len(subset)

        fig, axes = plt.subplots(rows, cols, figsize=opts['fig_size'], **opts['subplots'])
        axes = axes.reshape(rows, cols)

        # Style figure
        fig.subplots_adjust(**opts['fig_subplots_adjust'])
        fig.suptitle(opts['title'], **opts['title_format'])

        # Style axes
        row_idx = -1
        for row in range(dim):
            if row not in subset:
                continue
            else:
                row_idx += 1

            col_idx = -1
            for col in range(dim):
                if col not in subset:
                    continue
                else:
                    col_idx += 1

                if row == col:
                    current = 'diag'
                elif row < col:
                    current = 'upper'
                else:
                    current = 'lower'

                ax = axes[row_idx, col_idx]
                plt.sca(ax)

                # Background color
                if current in opts['fig_bg_colors'] and \
                    opts['fig_bg_colors'][current] is not None:
                    ax.set_facecolor(
                        opts['fig_bg_colors'][current])

                # Axes
                if opts[current] is None:
                    ax.axis('off')
                    continue

                # Limits
                if limits is not None:
                    ax.set_xlim(
                        (limits[col][0], limits[col][1]))
                    if current is not 'diag':
                        ax.set_ylim(
                            (limits[row][0], limits[row][1]))
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()

                # Ticks
                if ticks is not None:
                    ax.set_xticks(
                        (ticks[col][0], ticks[col][1]))
                    if current is not 'diag':
                        ax.set_yticks(
                            (ticks[row][0], ticks[row][1]))

                # Despine
                sns.despine(ax=ax, **opts['despine'])

                # Formatting axes
                if current == 'diag':  # off-diagnoals
                    if opts['lower'] is None or col == dim-1:
                        _format_axis(ax, xhide=False, xlabel=labels_dim[col],
                            yhide=True, tickformatter=opts['tickformatter'])
                        if opts['labelpad'] is not None: ax.xaxis.labelpad = opts['labelpad']
                    else:
                        _format_axis(ax, xhide=True, yhide=True)
                else:  # off-diagnoals
                    if row == dim-1:
                        _format_axis(ax, xhide=False, xlabel=labels_dim[col],
                            yhide=True, tickformatter=opts['tickformatter'])
                    else:
                        _format_axis(ax, xhide=True, yhide=True)
                if opts['tick_labels'] is not None:
                    ax.set_xticklabels(
                        (str(opts['tick_labels'][col][0]), str(opts['tick_labels'][col][1])))
                    if opts['tick_labelpad'] is not None:
                        ax.tick_params(axis='x', which='major', pad=opts['tick_labelpad'])

                # Diagonals
                if current == 'diag':
                    if len(samples) > 0:
                        for n, v in enumerate(samples):
                            if opts['diag'][n] == 'hist':
                                h = plt.hist(
                                    v[:, row],
                                    color=opts['samples_colors'][n],
                                    **opts['hist_diag']
                                )
                            elif opts['diag'][n] == 'kde':
                                density = gaussian_kde(
                                    v[:, row],
                                    bw_method=opts['kde_diag']['bw_method'])
                                xs = np.linspace(xmin, xmax, opts['kde_diag']['bins'])
                                ys = density(xs)
                                h = plt.plot(xs, ys,
                                    color=opts['samples_colors'][n],
                                )
                            elif opts['diag'][n] == 'cond':
                                p_vector = eval_conditional_density(opts['pdfs'][n], [opts['condition'][n]], opts['limits'], row, col, resolution=opts['hist_diag']['bins'], log=False)
                                p_vector = p_vector / np.max(p_vector) # just to scale it to 1
                                h = plt.plot(np.linspace(opts['limits'][row,0],opts['limits'][col,1],opts['hist_diag']['bins']), p_vector,
                                    c=opts['samples_colors'][n]
                                )
                            else:
                                pass

                    if len(points) > 0:
                        extent = ax.get_ylim()
                        for n, v in enumerate(points):
                            h = plt.plot(
                                [v[:, row], v[:, row]],
                                extent,
                                color=opts['points_colors'][n],
                                **opts['points_diag']
                            )

                # Off-diagonals
                else:

                    if len(samples) > 0:
                        for n, v in enumerate(samples):
                            if opts['upper'][n] == 'hist' or opts['upper'][n] == 'hist2d':
                                #h = plt.hist2d(
                                #     v[:, col], v[:, row],
                                #     range=(
                                #         [limits[col][0], limits[col][1]],
                                #         [limits[row][0], limits[row][1]]),
                                #     **opts['hist_offdiag']
                                #     )
                                hist, xedges, yedges = np.histogram2d(
                                    v[:, col], v[:, row], range=[
                                        [limits[col][0], limits[col][1]],
                                        [limits[row][0], limits[row][1]]],
                                    **opts['hist_offdiag'])
                                h = plt.imshow(hist.T,
                                    origin='lower',
                                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                    aspect='auto'
                                )

                            elif opts['upper'][n] in ['kde', 'kde2d', 'contour', 'contourf']:
                                density = gaussian_kde(v[:, [col, row]].T, bw_method=opts['kde_offdiag']['bw_method'])
                                X, Y = np.meshgrid(np.linspace(limits[col][0], limits[col][1], opts['kde_offdiag']['bins']),
                                                   np.linspace(limits[row][0], limits[row][1], opts['kde_offdiag']['bins']))
                                positions = np.vstack([X.ravel(), Y.ravel()])
                                Z = np.reshape(density(positions).T, X.shape)

                                if opts['upper'][n] == 'kde' or opts['upper'][n] == 'kde2d':
                                    h = plt.imshow(Z,
                                        extent=[limits[col][0], limits[col][1], limits[row][0], limits[row][1]],
                                        origin='lower',
                                        aspect='auto',
                                    )
                                elif opts['upper'][n] == 'contour':
                                    if opts['contour_offdiag']['percentile']:
                                        Z = probs2contours(Z, opts['contour_offdiag']['levels'])                                 
                                    else:
                                        Z = (Z - Z.min())/(Z.max() - Z.min())
                                    h = plt.contour(X, Y, Z,
                                        origin='lower',
                                        extent=[limits[col][0], limits[col][1], limits[row][0], limits[row][1]],
                                        colors=opts['samples_colors'][n],
                                        levels=opts['contour_offdiag']['levels']
                                    )
                                else:
                                    pass
                            elif opts['upper'][n] == 'scatter':
                                h = plt.scatter(
                                    v[:, col], v[:, row],
                                    color=opts['samples_colors'][n],
                                    **opts['scatter_offdiag']
                                )
                            elif opts['upper'][n] == 'plot':
                                h = plt.plot(
                                    v[:, col], v[:, row],
                                    color=opts['samples_colors'][n],
                                    **opts['plot_offdiag']
                                )
                            elif opts['upper'][n] == 'cond':
                                p_image = eval_conditional_density(opts['pdfs'][n],
                                [opts['condition'][n]], opts['limits'], row, col,
                                resolution=opts['hist_offdiag']['bins'], log=False)
                                h = plt.imshow(p_image, origin='lower',
                                                extent=[opts['limits'][row, 0], opts['limits'][row, 1], opts['limits'][col, 0], opts['limits'][col, 1]],
                                                aspect='auto'
                                )
                            else:
                                pass

                    if len(points) > 0:

                        for n, v in enumerate(points):
                            h = plt.plot(
                                v[:, col], v[:, row],
                                color=opts['points_colors'][n],
                                **opts['points_offdiag']
                            )

        if len(subset) < dim:
            for row in range(len(subset)):
                ax = axes[row, len(subset)-1]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                text_kwargs = {'fontsize': plt.rcParams['font.size']*2.}
                ax.text(x1 + (x1 - x0) / 8., (y0 + y1) / 2., '...', **text_kwargs)
                if row == len(subset)-1:
                    ax.text(x1 + (x1 - x0) / 12., y0 - (y1 - y0) / 1.5, '...', rotation=-45, **text_kwargs)

    return fig, axes


################################################################################
import numpy as np
from copy import deepcopy


def buildCondCovMatrix(posterior, lims, samples=None, num_samples=10, max_dim=None, resolution=20):

    if max_dim is None:
        max_dim = posterior.ndim
    if samples is None:
        samples = posterior.gen(num_samples) # TODO: rejection sampling!

    mdnewcounter = 0
    all_rho_images = []
    for theta in samples:
        print('mdnewcounter', mdnewcounter)
        rho_image_11 = np.zeros((max_dim, max_dim))
        for d1 in range(max_dim):
            for d2 in range(max_dim):
                if d1 < d2:
                    p_image = eval_conditional_density(posterior, [theta], lims, dim1=d1, dim2=d2,
                                                       resolution=resolution, log=False)
                    cc = conditional_correlation(p_image, lims[d1,0], lims[d1,1], lims[d2,0], lims[d2,1])
                    cc.calc_rhoXY()
                    rho_image_11[d1, d2] = cc.rho
        all_rho_images.append(rho_image_11)
        mdnewcounter += 1

    mean_conditional_correlation = np.nanmean(all_rho_images, axis=0)

    rho_symm = np.zeros((max_dim, max_dim))
    for d1 in range(max_dim):
        for d2 in range(max_dim):
            if d1 < d2:
                rho_symm[d1, d2] = mean_conditional_correlation[d1, d2]
            elif d1 > d2:
                rho_symm[d1, d2] = mean_conditional_correlation[d2, d1]
            else:
                rho_symm[d1, d2] = 1.0

    return rho_symm


def extractSpecificCondCorr(posterior, corrs, lims, samples=None, num_samples=10, max_dim=None, resolution=20):

    if max_dim is None:
        max_dim = posterior.ndim
    if samples is None:
        samples = posterior.gen(num_samples) # TODO: rejection sampling!

    all_lists_of_corrs = []
    for theta in samples:
        list_of_corrs = []

        for pair in corrs:
            d1 = pair[0]
            d2 = pair[1]
            if d1 < d2:
                p_image = eval_conditional_density(posterior, [theta], lims, dim1=d1, dim2=d2,
                                                   resolution=resolution, log=False)
                cc = conditional_correlation(p_image, lims[d1,0], lims[d1,1], lims[d2,0], lims[d2,1])
                cc.calc_rhoXY()
                list_of_corrs.append(cc.rho)
        all_lists_of_corrs.append(list_of_corrs)

    all_lists_of_corrs = np.asarray(all_lists_of_corrs).T

    return all_lists_of_corrs


def eval_conditional_density(pdf, theta, lims, dim1, dim2, resolution=20, log=True):

    if dim1 == dim2:
        gbar_dim1 = np.linspace(lims[dim1,0], lims[dim1,1], resolution)

        p_vector = np.zeros(resolution)

        list_of_current_point_eval = []

        for index_gbar1 in range(resolution):
            current_point_eval = deepcopy(theta)[0]
            current_point_eval[dim1] = gbar_dim1[index_gbar1]
            list_of_current_point_eval.append(current_point_eval)

        p = pdf.eval(np.asarray(list_of_current_point_eval))

        for index_gbar1 in range(resolution):
            p_vector[index_gbar1] = p[index_gbar1]

        if log:
            return p_vector
        else:
            return np.exp(p_vector)
    else:
        gbar_dim1 = np.linspace(lims[dim1,0], lims[dim1,1], resolution)
        gbar_dim2 = np.linspace(lims[dim2,0], lims[dim2,1], resolution)

        p_image = np.zeros((resolution, resolution))
        list_of_current_point_eval = []

        for index_gbar1 in range(resolution):
            for index_gbar2 in range(resolution):
                current_point_eval = deepcopy(theta)[0]
                current_point_eval[dim1] = gbar_dim1[index_gbar1]
                current_point_eval[dim2] = gbar_dim2[index_gbar2]
                list_of_current_point_eval.append(current_point_eval)

        p = pdf.eval(np.asarray(list_of_current_point_eval))
        i = 0
        for index_gbar1 in range(resolution):
            for index_gbar2 in range(resolution):
                p_image[index_gbar1, index_gbar2] = p[i]
                i += 1

        if log:
            return p_image
        else:
            return np.exp(p_image)


class conditional_correlation:
    def __init__(self, cPDF, lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y):
        self.lx = lower_bound_x
        self.ux = upper_bound_x
        self.ly = lower_bound_y
        self.uy = upper_bound_y
        self.resolution_y, self.resolution_x = np.shape(cPDF)
        self.pdfXY = self.normalize_pdf_2D(cPDF)
        self.pdfX  = None
        self.pdfY  = None
        self.EX    = None
        self.EY    = None
        self.EXY   = None
        self.VarX  = None
        self.VarY  = None
        self.CovXY = None
        self.rho   = None

    @staticmethod
    def normalize_pdf_1D(pdf, lower, upper, resolution):
        return pdf * resolution / (upper - lower) / np.sum(pdf)

    def normalize_pdf_2D(self, pdf):
        return pdf * self.resolution_x * self.resolution_y / (self.ux - self.lx) /\
               (self.uy - self.ly) / np.sum(pdf)

    def calc_rhoXY(self):
        self.calc_marginals()
        self.calc_EXY()
        self.EX  = conditional_correlation.calc_E_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.EY = conditional_correlation.calc_E_1D(self.pdfY, self.ly, self.uy, self.resolution_y)
        self.CovXY = self.EXY - self.EX * self.EY
        self.VarX = conditional_correlation.calc_var1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.VarY = conditional_correlation.calc_var1D(self.pdfY, self.ly, self.uy, self.resolution_y)
        self.rho = self.CovXY / np.sqrt(self.VarX * self.VarY)

    def calc_EXY(self):
        x_matrix = np.tile(np.linspace(self.lx, self.ux, self.resolution_x), (self.resolution_y, 1))
        y_matrix = np.tile(np.linspace(self.ly, self.uy, self.resolution_y), (self.resolution_x, 1)).T
        self.EXY = np.sum(  np.sum(  x_matrix * y_matrix * self.pdfXY  )  )
        self.EXY /= self.resolution_x * self.resolution_y / (self.ux - self.lx) / (self.uy - self.ly)

    @staticmethod
    def calc_E_1D(pdf, lower, upper, resolution):
        x_vector = np.linspace(lower, upper, resolution)
        E = np.sum(x_vector * pdf)
        E /= resolution / (upper - lower)
        return E

    @staticmethod
    def calc_var1D(pdf, lower, upper, resolution):
        x_vector = np.linspace(lower, upper, resolution)
        E2 = np.sum(x_vector**2 * pdf)
        E2 /= resolution / (upper - lower)
        var = E2 - conditional_correlation.calc_E_1D(pdf, lower, upper, resolution)**2
        return var

    def calc_marginals(self):
        self.pdfX = np.sum(self.pdfXY, axis=0)
        self.pdfY = np.sum(self.pdfXY, axis=1)

        self.pdfX = conditional_correlation.normalize_pdf_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.pdfY = conditional_correlation.normalize_pdf_1D(self.pdfY, self.ly, self.uy, self.resolution_y)




def partialCorrelation(posterior_samples, max_dim):
    import sklearn.linear_model as lm
    import scipy.stats

    rho_image_11 = np.zeros((max_dim, max_dim))
    significance_matrix = np.zeros((max_dim, max_dim))
    for d1 in range(max_dim):
        print('Starting new dimension:  ', d1)
        for d2 in range(max_dim):
            if d1 < d2:
                vec = np.ones(max_dim, dtype=int)
                vec[d1] = 0
                vec[d2] = 0
                condition_samples = posterior_samples[:,vec]
                samplesX = posterior_samples[:,d1]
                samplesY = posterior_samples[:,d2]
                regX = lm.LinearRegression().fit(condition_samples, samplesX)
                regY = lm.LinearRegression().fit(condition_samples, samplesY)

                predictedX = regX.predict(condition_samples)
                predictedY = regY.predict(condition_samples)

                residualsX = samplesX - predictedX
                residualsY = samplesY - predictedY

                partial_corr, p_val = scipy.stats.pearsonr(residualsX, residualsY)
                rho_image_11[d1, d2]        = partial_corr
                significance_matrix[d1, d2] = p_val

    rho_symm = np.zeros((max_dim, max_dim))
    for d1 in range(max_dim):
        for d2 in range(max_dim):
            if d1 < d2:
                rho_symm[d1, d2] = rho_image_11[d1, d2]
            elif d1 > d2:
                rho_symm[d1, d2] = rho_image_11[d2, d1]
            else:
                rho_symm[d1, d2] = 1.0

    return rho_symm, significance_matrix



class conditional_mutual_information:
    def __init__(self, cPDF, lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y):
        self.lx = lower_bound_x
        self.ux = upper_bound_x
        self.ly = lower_bound_y
        self.uy = upper_bound_y
        self.resolution_y, self.resolution_x = np.shape(cPDF)
        self.pdfXY = self.normalize_pdf_2D(cPDF)
        self.pdfX  = None
        self.pdfY  = None
        self.HX    = None
        self.IXY   = None

    @staticmethod
    def normalize_pdf_1D(pdf, lower, upper, resolution):
        return pdf * resolution / (upper - lower) / np.sum(pdf)

    def normalize_pdf_2D(self, pdf):
        return pdf * self.resolution_x * self.resolution_y / (self.ux - self.lx) /\
               (self.uy - self.ly) / np.sum(pdf)

    def calc_HX(self):
        self.pdfX = self.normalize_pdf_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.HX = -np.sum(self.pdfX * np.log(self.pdfX))
        self.HX /= self.resolution_x / (self.ux - self.lx)

    def calc_IXY(self):
        self.calc_marginals()
        pdfXmatrix = np.tile(self.pdfX, (self.resolution_y,1))
        pdfYmatrix = np.tile(self.pdfY, (self.resolution_x,1)).T
        self.IXY   = np.sum(self.pdfXY * np.log(self.pdfXY / pdfXmatrix / pdfYmatrix))
        self.IXY /= self.resolution_x * self.resolution_y / (self.ux - self.lx) / (self.uy - self.ly)

    def calc_marginals(self):
        self.pdfX = np.sum(self.pdfXY, axis=0)
        self.pdfY = np.sum(self.pdfXY, axis=1)

        self.pdfX = conditional_correlation.normalize_pdf_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.pdfY = conditional_correlation.normalize_pdf_1D(self.pdfY, self.ly, self.uy, self.resolution_y)
