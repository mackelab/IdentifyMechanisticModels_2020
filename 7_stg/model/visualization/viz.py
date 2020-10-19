import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from print_helper import build_string, conductance_to_value_exp, pick_synapse, scale_to_experimental
import netio
import seaborn as sns
from copy import deepcopy
import matplotlib.colors as mcolors
from decimal import Decimal
import importlib
import viz_samples
import scipy.stats as st
import scipy.signal as ss
from scipy.optimize import curve_fit
import matplotlib.patheffects as pe
#from delfi.utils.utils_prinzetal import inv_logistic_fct
import sys
sys.path.append("../utils")

def loss(losses, key='trn', loss_clipping=1000., title=''):
    """Given an info dict, plot loss"""

    x = np.array(losses[key + '_iter'])
    y = np.array(losses[key + '_val'])

    clip_idx = np.where(y > loss_clipping)[0]
    if len(clip_idx) > 0:
        print(
            'warning: loss exceeds threshold of {:.2f} in total {} time(s); values will be clipped'.format(
                loss_clipping,
                len(clip_idx)))

    y[clip_idx] = loss_clipping

    options = {}
    options['title'] = title
    options['xlabel'] = r'iteration'
    options['ylabel'] = r'loss'

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(x, y, 'b')
    ax.set_xlabel(options['xlabel'])
    ax.set_ylabel(options['ylabel'])

    return fig, ax


def dist(dist, title=''):
    """Given dist, plot histogram"""
    options = {}
    options['title'] = title
    options['xlabel'] = r'bin'
    options['ylabel'] = r'distance'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_samples = len(dist)
    ax.hist(dist, bins=int(np.sqrt(n_samples)))
    ax.set_xlabel(options['xlabel'])
    ax.set_ylabel(options['ylabel'])
    ax.set_title(options['title'])
    return fig, ax


def info(info, html=False, title=None):
    """Given info dict, produce info text"""
    if title is None:
        infotext = u''
    else:
        if html:
            infotext = u'<b>{}</b><br>'.format(title)
        else:
            infotext = u'{}\n'.format(title)

    for key, value in info.items():
        if key not in ['losses']:
            infotext += u'{} : {}'.format(key, value)
            if html:
                infotext += '<br>'
            else:
                infotext += '\n'

    return infotext


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


def gkern(kernlen=21):
    """Returns a 2D Gaussian kernel."""

    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def plot_pdf(pdf1=None, lims=None, pdf2=None, prior=None, gt=None, contours=False, levels=(0.68, 0.95),
             start_col='orange', end_col='b', path_col='w', num_dim_input=None, profile_posterior=False,
             conditional_posterior=False, log_histogram=False, label_mode='exponential',
             optimize_profile_posterior=False, num_profile_samples=100000, eval_pdf=False, sample_probs=None,
             resolution=500, labels_params=None, ticks=False, diag_only=False, linescale=1.0, pointscale=1.0,
             dimensions=None, diag_only_cols=4, diag_only_rows=4, figsize=(5, 5), fontscale=1.0, no_contours=False,
             partial=False, samples=None, start_point=None, end_point=None, current_point=None, smooth_MAF=False,
             path1=None, path2=None, path_steps1=5, path_steps2=5, col1='k', col2='b', col3='g', title=None, figname=None):
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
    samples: array
        If given, samples of a distribution are plotted along `pdf`.
        If given, `pdf` is plotted with default `levels` (0.68, 0.95), if provided `levels` is None.
        If given, `lims` is overwritten and taken to be the respective
        limits of the samples in each dimension.
    col1 : str
        color 1
    col2 : str
        color 2
    col3 : str
        color 3 (for pdf2 if provided)
    """

    mpl.rcParams['axes.linewidth'] = 10

    pdfs = (pdf1, pdf2)
    colrs = (col2, col3)
    if path1 is not None: path1 = np.transpose(path1)
    if path2 is not None: path2 = np.transpose(path2)

    params_mean = prior.mean
    params_std = prior.std

    num_dim = num_dim_input

    if dimensions is None:
        dimensions = np.arange(num_dim)

    if not (pdf1 is None or pdf2 is None):
        assert pdf1.ndim == pdf2.ndim

    if samples is not None:
        contours = True
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
        lims = np.tile(lims, [num_dim, 1]) if lims.ndim == 1 else lims

    if samples is not None and profile_posterior:
        if sample_probs is None:
            samples_trunc = samples.T[:num_profile_samples]
            sample_probs = pdf1.eval(samples_trunc, log=False)
        else:
            samples_trunc = samples.T[:len(sample_probs)]
        samplesAppendP = np.concatenate((samples_trunc, np.asarray([sample_probs]).T), axis=1)

    if num_dim == 1:

        fig, ax = plt.subplots(1, 1, facecolor='white', figsize=figsize)

        if samples is not None:
            ax.hist(samples[0], bins=100, density=True,
                    color=col1,
                    edgecolor=col1)

        xx = np.linspace(lims[0, 0], lims[0, 1], resolution)

        for pdf, col in zip(pdfs, colrs):
            if pdf is not None:
                pp = pdf.eval(xx[:, np.newaxis], log=False)
                ax.plot(xx, pp, color=col)
        ax.set_xlim(lims[0])
        ax.set_ylim([0, ax.get_ylim()[1]])
        if gt is not None:
            ax.vlines(gt, 0, ax.get_ylim()[1], color='r')
        if start_point is not None:
            ax.vlines(start_point, 0, ax.get_ylim()[1], color='g')
        if end_point is not None:
            ax.vlines(end_point, 0, ax.get_ylim()[1], color='r')
        if current_point is not None:
            ax.vlines(current_point, 0, ax.get_ylim()[1], color='b')

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
                rows = min(3, num_dim)
                cols = min(3, num_dim)
            else:
                rows = num_dim
                cols = num_dim
                rows = len(dimensions)
                cols = len(dimensions)
        else:
            cols = diag_only_cols
            rows = diag_only_rows
            r = 0
            c = -1

        fig, ax = plt.subplots(rows, cols, facecolor='white', figsize=figsize)
        #ax = ax.reshape(rows, cols)

        for i in range(rows):
            for j in range(cols):

                dim_i = dimensions[i]
                dim_j = dimensions[j]

                if dim_i == dim_j:
                    pdf = pdfs[0]
                    if samples is not None and not conditional_posterior:
                        if profile_posterior: # TODO : implement profile posterior here!!!
                            print('Warning: diagonal is no profile posterior yet')
                            hist_vals, x_pos = np.histogram(samples[dim_i, :], bins=100, density=True)
                            sigma = 1.5
                            gx = np.arange(-3 * sigma, 3 * sigma, 1.0)
                            gaussian = np.exp(-(gx / sigma) ** 2 / 2)
                            gaussian = gaussian / np.sum(gaussian)
                            hist_vals = np.convolve(hist_vals, gaussian, mode='same')
                            ax[i, j].plot(x_pos[:-1], hist_vals, color='k', lw=2.0 * linescale)
                        else:
                            hist_vals, x_pos = np.histogram(samples[dim_i, :], bins=100, density=True)
                            sigma = 1.5
                            gx = np.arange(-3 * sigma, 3 * sigma, 1.0)
                            gaussian = np.exp(-(gx / sigma) ** 2 / 2)
                            gaussian = gaussian / np.sum(gaussian)
                            hist_vals = np.convolve(hist_vals, gaussian, mode='same')
                            ax[i, j].plot(x_pos[:-1], hist_vals, color='b', lw=2.0*linescale)

                    if conditional_posterior:
                        assert start_point is not None, 'Conditional posterior requires current_point.'
                        p_vector = eval_conditional_density(pdf, [start_point], dim_i, dim_j, resolution=resolution, log=False)
                        ax[i, j].plot(np.linspace(lims[dim_i,0],lims[dim_i,1],resolution),
                                      p_vector, color='k', lw=2.0*linescale)

                    xx = np.linspace(lims[dim_i, 0], lims[dim_i, 1], resolution)

                    #for pdf, col in zip(pdfs, colrs):
                    #    if pdf is not None:
                    #        pp = pdf.eval(xx, ii=[dim_i], log=False)

                    if diag_only:
                        c += 1
                    else:
                        r = i
                        c = j

                    if eval_pdf:
                        for pdf, col in zip(pdfs, colrs):
                            if pdf is not None:
                                pp = pdf.eval(xx, ii=[i], log=False)
                                ax[r, c].plot(xx, pp, color=col)

                    ax[r, c].set_xlim(lims[dim_i])
                    ax[r, c].set_ylim([0, ax[r, c].get_ylim()[1]])

                    if gt is not None:
                        ax[r, c].vlines(
                            gt[dim_i], 0, ax[r, c].get_ylim()[1], color='r')
                    if start_point is not None:
                        ax[r, c].vlines(
                            start_point[dim_i], 0, ax[r, c].get_ylim()[1], color=start_col, lw=2.0*linescale, zorder=3)
                    if end_point is not None:
                        ax[r, c].vlines(
                            end_point[dim_i], 0, ax[r, c].get_ylim()[1], color=end_col, lw=2.0*linescale, zorder=4)
                    if current_point is not None:
                        ax[r, c].vlines(
                            current_point[dim_i], 0, ax[r, c].get_ylim()[1], color='b', lw=2.0*linescale)

                    fs_ = 4.0
                    if ticks:
                        factor = np.ones(num_dim)
                        factor = np.asarray([1.0, 10.0, 100, 10000, 100, 10000])
                        lims_here = deepcopy(lims)
                        lims_here = lims_here
                        ax[r, c].get_yaxis().set_tick_params(
                            which='both', direction='out', labelsize=fontscale * 8.0)
                        ax[r, c].get_xaxis().set_tick_params(
                            which='both', direction='out', labelsize=fontscale * 8.0)
#                         ax[r, c].locator_params(nbins=3)
                        ax[r, c].set_xticks(np.linspace(
                            lims_here[dim_i, 0]+0.0*np.abs(lims_here[dim_i, 0]-lims_here[dim_j, 1]),
                            lims_here[j, 1]-0.0*np.abs(lims_here[dim_i, 0]-lims_here[dim_j, 1]), 2))
                        if dim_i > len(params_mean) - 7.5: # synapses
                            if label_mode == 'exponential':
                                #labels = [round(Decimal((np.exp(lims[dim_i, 0] * params_std[dim_i] + params_mean[dim_i])) * 1e-3 * 1e9), 2),
                                #          round(Decimal((np.exp(lims[dim_i, 1] * params_std[dim_i] + params_mean[dim_i])) * 1e-3 * 1e9), 0)]
                                pos_of_labels = (np.log(np.asarray([1e-11, 1e-6])/1e-3) - params_mean[dim_i]) / params_std[dim_i]
                                #labels = ['1e-11', '1e-6  ']
                                ax[r,c].xaxis.set_major_locator(plt.FixedLocator(pos_of_labels))

                                val1 = 1e-11
                                val2 = 1e-6
                                str1 = build_string(conductance_to_value_exp([val1]), include_multiplier=False)
                                str2 = build_string(conductance_to_value_exp([val2]), include_multiplier=False)
                                ax[r, c].set_xticklabels([r'$%s$'%str1, r'$%s$    '%str2]) # blankspaces
                            else:
                                pos_of_labels = (np.log(np.asarray([1e-11, 1e-6]) / 1e-3) - params_mean[dim_i]) / \
                                                params_std[dim_i]
                                ax[r, c].xaxis.set_major_locator(plt.FixedLocator(pos_of_labels))
                                ax[r, c].set_xticklabels(['0.01', '1000   '])  # blankspaces
                        else: # membrane conductances
                            if label_mode=='exponential':
                                labels = [round(Decimal(factor[dim_i]*(lims[dim_i, num_tmp] * params_std[dim_i] + params_mean[dim_i])/0.628e-3), 0) for num_tmp in range(2)]
                                ax[r, c].set_xticklabels(labels)
                                ax[r, c].text(0.45, -0.95, 'x', fontsize=fontscale * 8.0, transform=ax[r, c].transAxes)
                                if factor[dim_i] == 1.0:
                                    ax[r, c].text(0.43, -1.17, r'$%s$' % str(int(factor[dim_i])), fontsize=fontscale * 8.0,
                                                  transform=ax[r, c].transAxes)
                                elif factor[dim_i] == 10.0:
                                    ax[r, c].text(0.38, -1.17, r'$%s$' % str(int(factor[dim_i])), fontsize=fontscale * 8.0,
                                                  transform=ax[r, c].transAxes)
                                else:
                                    ax[r, c].text(0.33, -1.17, r'$%s$' % build_string(conductance_to_value_exp([factor[dim_i]]), include_multiplier=False, negative_num=False), fontsize=fontscale * 8.0, transform=ax[r, c].transAxes)
                            else:
                                num_after_digits = -int(np.log10(lims[dim_i, 1] * params_std[dim_i] + params_mean[dim_i]))
                                if num_after_digits > 2:
                                    num_after_digits=2
                                labels = [round(Decimal((lims[dim_i, num_tmp] * params_std[dim_i] + params_mean[dim_i]) / 0.628e-3), num_after_digits)
                                          for num_tmp in range(2)]
                                new_labels = []
                                counter=0
                                for l in labels:
                                    if counter == 0:
                                        new_labels.append('   '+str(l))
                                    else:
                                        new_labels.append(str(l)+'   ')
                                    counter+=1

                                ax[r, c].set_xticklabels(new_labels)

                        ax[r, c].tick_params(width=2.0*linescale, length=5.0*linescale)
                        #ax[r, c].set_yticks(np.linspace(0+0.15*np.abs(0-max(pp)), max(pp)-0.15*np.abs(0-max(pp)), 2))
                        ax[r, c].get_yaxis().set_ticks([])
                    else:
                        ax[r, c].get_xaxis().set_ticks([])
                        ax[r, c].get_yaxis().set_ticks([])

                    if labels_params is not None:
                        if label_mode == 'exponential':
                            if ticks:
                                if dim_i > 5:
                                    ax[r, c].set_xlabel(
                                        r'$\mathdefault{%s}$ ' %labels_params[dim_i], fontsize=fontscale * 8.0)
                                else:
                                    ax[r, c].set_xlabel(
                                        r'$\mathdefault{%s}$' % labels_params[dim_i], fontsize=fontscale * 8.0)
                                ax[r, c].xaxis.labelpad = 3.0*fontscale
                            else:
                                if dim_i > 5:
                                    ax[r, c].set_xlabel(
                                        r'$\mathdefault{%s}$  ' % labels_params[dim_i], fontsize=fontscale * 8.0)
                                else:
                                    ax[r, c].set_xlabel(
                                        r'$\mathdefault{%s}$' % labels_params[dim_i], fontsize=fontscale * 8.0)
                                ax[r, c].xaxis.labelpad = 3.0*fontscale
                        else:
                            if ticks:
                                if dim_i > 5:
                                    ax[r, c].set_xlabel(
                                        r'$\mathdefault{%s}$ ' %labels_params[dim_i], fontsize=fontscale * 8.0)
                                else:
                                    ax[r, c].set_xlabel(
                                        '%s' % labels_params[dim_i], fontsize=fontscale * 8.0)
                                ax[r, c].xaxis.labelpad = 3.0*fontscale
                            else:
                                if dim_i > 5:
                                    ax[r, c].set_xlabel(
                                        r'$\mathdefault{%s}$ ' % labels_params[dim_i], fontsize=fontscale * 8.0)
                                else:
                                    ax[r, c].set_xlabel(
                                        '%s' % labels_params[dim_i], fontsize=fontscale * 8.0)
                                ax[r, c].xaxis.labelpad = 3.0*fontscale
                    else:
                        ax[r, c].set_xlabel([])

                    x0, x1 = ax[r, c].get_xlim()
                    y0, y1 = ax[r, c].get_ylim()
                    ax[r, c].set_aspect((x1 - x0) / (y1 - y0))

                    if partial and dim_i == rows - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...', fontsize=fontscale * 25)
                        plt.text(x1 + (x1 - x0) / 8.4, y0 - (y1 - y0) /
                                 6., '...', fontsize=fontscale * 25, rotation=-45)

                    ax[i, j].spines['right'].set_visible(False)
                    ax[i, j].spines['top'].set_visible(False)
                    ax[i, j].spines['left'].set_visible(False)
                    ax[i, j].spines['bottom'].set_linewidth(2.0*linescale)
                    ax[i, j].spines['bottom'].set_position(('axes', -0.2))
                else:

                    if diag_only:
                        continue

                    if dim_i < dim_j:
                        pdf = pdfs[0]
                    else:
                        pdf = pdfs[1]

                    if pdf is None and dim_i > dim_j:
                        ax[i, j].get_yaxis().set_visible(False)
                        ax[i, j].get_xaxis().set_visible(False)
                        ax[i, j].set_axis_off()
                        continue

                    if samples is not None and not profile_posterior and not conditional_posterior:
                        H, xedges, yedges = np.histogram2d(
                            samples[dim_i, :], samples[dim_j, :], bins=resolution, range=[
                            [lims[dim_i, 0], lims[dim_i, 1]], [lims[dim_j, 0], lims[dim_j, 1]]], density=True)
                        if smooth_MAF:
                            gkernel = gkern(kernlen=3)
                            H = ss.convolve2d(H, gkernel, mode='valid')
                        if dim_i > 50 and dim_j > 50:
                            difference = np.max(H) - np.min(H)
                            ax[i, j].imshow(H, origin='lower', extent=[
                                yedges[0], yedges[-1], xedges[0], xedges[-1]],
                                            clim=[np.min(H)-3*difference, np.max(H)+3*difference])
                        else:
                            if log_histogram:
                                H[H==0] = np.min(H[H>0.0])
                                ax[i, j].imshow(np.log(H), origin='lower', extent=[
                                                yedges[0], yedges[-1], xedges[0], xedges[-1]])
                            else:
                                ax[i, j].imshow(H, origin='lower', extent=[
                                    yedges[0], yedges[-1], xedges[0], xedges[-1]])
                    elif profile_posterior:
                        data_in_2D_bins = vu.bin_data_2D(samplesAppendP.T, dim_i, dim_j, num_bins=resolution)
                        best_sample = np.asarray(vu.binsToMaxSample(data_in_2D_bins, data_dim=num_dim))
                        best_sample_probs = best_sample[:, :, -1]
                        print('Got non-optimized profile posterior')
                        if optimize_profile_posterior:
                            best_sample_probs = vu.get_profile_likelihood(best_sample, pdf, dim_i, dim_j, 'maf')
                            print('optimized profile posterior')
                        ax[i, j].imshow(best_sample_probs, origin='lower', extent=[lims[dim_i,0], lims[dim_i,1], lims[dim_j,0], lims[dim_j,1]])

                    if conditional_posterior:
                        p_image = eval_conditional_density(pdf, [start_point], dim_i, dim_j, resolution=resolution, log=False)
                        ax[i, j].imshow(p_image, origin='lower',
                                        extent=[lims[dim_i, 0], lims[dim_i, 1], lims[dim_j, 0], lims[dim_j, 1]])

                    xx = np.linspace(lims[dim_i, 0], lims[dim_i, 1], resolution)
                    yy = np.linspace(lims[dim_j, 0], lims[dim_j, 1], resolution)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate(
                        [X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                    if eval_pdf and pdf is not None:
                        pp = pdf.eval(xy, ii=[dim_i, dim_j], log=False)
                        pp = pp.reshape(list(X.shape))
                        if contours:
                            if no_contours:
                                pass
                            else:
                                ax[i, j].contour(Y, X, probs2contours(
                                    pp, levels), levels, colors=('w', 'y'))
                        else:
                            if no_contours:
                                pass
                            else:
                                ax[i, j].imshow(pp.T, origin='lower',
                                                extent=[lims[dim_j, 0], lims[dim_j, 1], lims[dim_i, 0], lims[dim_i, 1]],
                                                aspect='auto', interpolation='none')
                                ax[i, j].contour(Y, X, probs2contours(
                                    pp, levels), levels, colors=('w', 'y'))

                    if path1 is not None:
                        ax[i, j].plot(path1[dim_j][0::path_steps1], path1[dim_i][0::path_steps1],
                                      color=path_col, lw=2.8*pointscale,
                                      path_effects=[pe.Stroke(linewidth=3.8*pointscale, foreground='k'), pe.Normal()])

                    if path2 is not None:
                        ax[i, j].plot(path2[dim_j][0::path_steps2], path2[dim_i][0::path_steps2],
                                      color=path_col, lw=2.8 * pointscale,
                                      path_effects=[pe.Stroke(linewidth=3.8 * pointscale, foreground='k'), pe.Normal()])

                    if gt is not None:
                        ax[i, j].plot(gt[dim_j], gt[dim_i], 'r.', ms=10*pointscale,
                                      markeredgewidth=0.0)
                    if start_point is not None:
                        ax[i, j].plot(start_point[dim_j], start_point[dim_i], color=start_col, marker='o', markeredgecolor='w', ms=7*pointscale,
                                      markeredgewidth=1.0*pointscale, path_effects=[pe.Stroke(linewidth=2.5*pointscale, foreground='k'), pe.Normal()])
                    if end_point is not None:
                        ax[i, j].plot(end_point[dim_j], end_point[dim_i], color=end_col, marker='o', markeredgecolor='w', ms=7*pointscale,
                                      markeredgewidth=1.0*pointscale, path_effects=[pe.Stroke(linewidth=2.5*pointscale, foreground='k'), pe.Normal()])
                    if current_point is not None:
                        ax[i, j].plot(current_point[dim_j], current_point[dim_i], 'bo', ms=15*pointscale,
                                      markeredgewidth=0.0*pointscale)

                    ax[i, j].set_xlim(lims[dim_j])
                    ax[i, j].set_ylim(lims[dim_i])

                    ax[i, j].get_xaxis().set_ticks([])
                    ax[i, j].get_yaxis().set_ticks([])
                    ax[i, j].set_axis_off()

                    x0, x1 = ax[i, j].get_xlim()
                    y0, y1 = ax[i, j].get_ylim()
                    ax[i, j].set_aspect((x1 - x0) / (y1 - y0))

                    if partial and dim_j == cols - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...', fontsize=fontscale * 25)

                if diag_only and c == cols - 1:
                    c = -1
                    r += 1

    FIGS_DIR = '../../thesis_results'

    #fig.subplots_adjust(hspace=0.1, wspace=0.5)

    if title is not None:
        fig.suptitle(title, fontsize=36)
        fig.subplots_adjust(top=0.965)

    '''
    for i, row in enumerate(ax):
        ax = row[i]
        ax.set_yticks([])
        val1 = 1e-8
        val2 = 1e-3
        str1 = build_string(conductance_to_value_exp([val1]))
        str2 = build_string(conductance_to_value_exp([val2]))
        str1 = str1[-4:]
        str2 = str2[-4:]
        xticks = [3, 8]
        if i > rows - 7.5:
            ax.set_xticks([np.log(0.1 ** j) for j in xticks])
            ax.set_xticklabels([r'$%s$'%str1, r'$%s$'%str2], fontsize=14.0)
        #else:
        #    ax.set_xticks(lims[i])
        #    ax.set_xticklabels([r'$%s$' % str1, r'$%s$' % str2], fontsize=14.0)
        ax.set_aspect('auto')
    '''

    if figname is not None:
        fig.savefig("{}/pdf/{}.pdf".format(FIGS_DIR, figname), bbox_inches='tight')
        fig.savefig("{}/svg/{}.svg".format(FIGS_DIR, figname), bbox_inches='tight')
        fig.savefig("{}/png/{}.png".format(FIGS_DIR, figname), bbox_inches='tight', dpi=200)

    return fig, ax



def plot_single_marginal_pdf(pdf1=None, lims=None, pdf2=None, prior=None, contours=False, levels=(0.68, 0.95), pointscale=1.0,
             start_col='orange', end_col='b', path_col='w', path_col2='w', current_col='g', current_col1='g', current_col2='g', num_dim_input=None,
             optimize_profile_posterior=False, num_profile_samples=100000, eval_pdf=False, sample_probs=None, log_profile=False,
             params_mean_log=None, params_std_log=None, resolution=500, labels_params=None, ticks=False, diag_only=False, smooth_MAF=True,
             diag_only_cols=4, diag_only_rows=4, figsize=(5, 5), fontscale=1, no_contours=False, profile_posterior=False,
             partial=False, samples=None, start_point=None, end_point=None, current_point=None, current_point2=None,
             labels=None,display_axis_lims=True, p1g=None, p2g=None, p3g=None, p1b=None, p2b=None, p3b=None,
             path1=None, path2=None, path3=None, path_steps1=5, path_steps2=5, col1='k', col2='b', col3='g', pdf_type='MAF',
             dimensions=None, title=None, figname=None, ax=None):
    """Plots marginals of a pdf, for each variable and pair of variables.
ax.plot(path1[dim_j][0:-1:path_steps1], path1[dim_i][0:-1:path_steps1],
                      color=path_col, lw=2.8,
                      path_effects=[pe.Stroke(linewidth=3.8, foreground='k'), pe.Normal()])
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
    samples: array
        If given, samples of a distribution are plotted along `pdf`.
        If given, `pdf` is plotted with default `levels` (0.68, 0.95), if provided `levels` is None.
        If given, `lims` is overwritten and taken to be the respective
        limits of the samples in each dimension.
    col1 : str
        color 1
    col2 : str
        color 2
    col3 : str
        color 3 (for pdf2 if provided)
    """

    pdfs = (pdf1, pdf2)
    colrs = (col2, col3)
    if path1 is not None: path1 = np.transpose(path1)
    if path2 is not None: path2 = np.transpose(path2)
    if path3 is not None: path3 = np.transpose(path3)

    params_mean = prior.mean
    params_std = prior.std

    num_dim = num_dim_input

    if not (pdf1 is None or pdf2 is None):
        assert pdf1.ndim == pdf2.ndim

    if samples is not None:
        contours = True
        if levels is None:
            levels = (0.68, 0.95)


    if samples is not None and lims is None:
        lims_min = np.min(samples, axis=1)
        lims_max = np.max(samples, axis=1)
        lims = np.concatenate(
            (lims_min.reshape(-1, 1), lims_max.reshape(-1, 1)), axis=1)
    else:
        lims = np.asarray(lims)
        lims = np.tile(lims, [num_dim, 1]) if lims.ndim == 1 else lims

    if samples is not None and profile_posterior:
        if sample_probs is None:
            samples_trunc = samples.T[:num_profile_samples]
            sample_probs = pdf1.eval(samples_trunc, log=False)
        else:
            samples_trunc = samples.T[:len(sample_probs)]
        samplesAppendP = np.concatenate((samples_trunc, np.asarray([sample_probs]).T), axis=1)

    rows = 1
    cols = 1

    if ax is None:
        fig, ax = plt.subplots(rows, cols, facecolor='white', figsize=figsize)

    dim_i = dimensions[0]
    dim_j = dimensions[1]

    pdf = pdfs[0]

    if pdf is None:
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_axis_off()

    best_sample_probs = np.zeros((resolution, resolution))
    if samples is not None and not profile_posterior:
        if samples is not None:
            H, xedges, yedges = np.histogram2d(
                samples[dim_i, :], samples[dim_j, :], bins=resolution, range=[
                [lims[dim_i, 0], lims[dim_i, 1]], [lims[dim_j, 0], lims[dim_j, 1]]], density=True)
            if smooth_MAF:
                gkernel = gkern(kernlen=10)
                H = ss.convolve2d(H, gkernel, mode='valid')
            ax.imshow(H, origin='lower', extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
                      vmin=0.0, vmax=np.min(H)+(np.max(H)-np.min(H))*1.0)
            best_sample_probs = H
    elif profile_posterior:
        data_in_2D_bins = vu.bin_data_2D(samplesAppendP.T, dim_i, dim_j, num_bins=resolution)
        best_sample = np.asarray(vu.binsToMaxSample(data_in_2D_bins, data_dim=num_dim))
        best_sample_probs = best_sample[:, :, -1]
        if optimize_profile_posterior: # run a numerical optimization for every pixel
            best_sample_probs = vu.get_profile_likelihood(best_sample, pdf, dim_i, dim_j, 'maf')
        if log_profile: # take the log? If yes, take the 0.0 probability states and set them to the minimum
            minimum_observed = np.min(best_sample_probs[best_sample_probs>0.0])
            best_sample_probs[best_sample_probs==0.0] = minimum_observed
            best_sample_probs = np.log(best_sample_probs)
        ax.imshow(best_sample_probs, origin='lower',
                        extent=[lims[dim_i, 0], lims[dim_i, 1], lims[dim_j, 0], lims[dim_j, 1]])


    if path1 is not None:
        ax.plot(path1[dim_j][::path_steps1], path1[dim_i][::path_steps1],
                      color=path_col, lw=5.5*pointscale,
                      path_effects=[pe.Stroke(linewidth=7.1*pointscale, foreground='k'), pe.Normal()])

    if path2 is not None:
        ax.plot(path2[dim_j][0::path_steps2], path2[dim_i][0::path_steps2],
                color=path_col2, lw=5.5*pointscale,
                path_effects=[pe.Stroke(linewidth=7.1*pointscale, foreground='k'), pe.Normal()])
    if start_point is not None:
        ax.plot(start_point[dim_j], start_point[dim_i], color=start_col, marker='o', markeredgecolor='w', ms=16*pointscale,
                      markeredgewidth=2.7*pointscale, path_effects=[pe.Stroke(linewidth=4.0*pointscale, foreground='k'), pe.Normal()])
    if end_point is not None:
        ax.plot(end_point[dim_j], end_point[dim_i], color=end_col, marker='o', markeredgecolor='w', ms=16*pointscale,
                      markeredgewidth=2.7*pointscale, path_effects=[pe.Stroke(linewidth=4.0*pointscale, foreground='k'), pe.Normal()])
    if current_point is not None:
        for current_p in current_point:
            ax.plot(current_p[dim_j], current_p[dim_i], color=current_col, marker='o', markeredgecolor='w', ms=27*pointscale,
                          markeredgewidth=3.5*pointscale, path_effects=[pe.Stroke(linewidth=4.9*pointscale, foreground='k'), pe.Normal()])
    if current_point2 is not None:
        for current_p in current_point2:
            ax.plot(current_p[dim_j], current_p[dim_i], color=current_col2, marker='o', markeredgecolor='w', ms=27*pointscale,
                          markeredgewidth=3.5*pointscale, path_effects=[pe.Stroke(linewidth=4.9*pointscale, foreground='k'), pe.Normal()])

    if p1g is not None:
        ax.plot(p1g[dim_j], p1g[dim_i], color=current_col, marker='o', markeredgecolor='k', ms=8*pointscale, markeredgewidth=0.5*pointscale)
    if p2g is not None:
        ax.plot(p2g[dim_j], p2g[dim_i], color=current_col1, marker='o', markeredgecolor='k', ms=8*pointscale, markeredgewidth=0.5*pointscale)
    if p3g is not None:
        ax.plot(p3g[dim_j], p3g[dim_i], color=current_col1, marker='o', markeredgecolor='k', ms=8*pointscale, markeredgewidth=0.5*pointscale)
    if p1b is not None:
        ax.plot(p1b[dim_j], p1b[dim_i], color=current_col2, marker='o', markeredgecolor='k', ms=8*pointscale, markeredgewidth=0.5*pointscale)
    if p2b is not None:
        ax.plot(p2b[dim_j], p2b[dim_i], color=current_col2, marker='o', markeredgecolor='k', ms=8*pointscale, markeredgewidth=0.5*pointscale)
    if p3b is not None:
        ax.plot(p3b[dim_j], p3b[dim_i], color=current_col2, marker='o', markeredgecolor='k', ms=8*pointscale, markeredgewidth=0.5*pointscale)

    ax.set_xlim(lims[dim_j])
    ax.set_ylim(lims[dim_i])
    ax.tick_params(width=2.0 * 0.666, length=5.0 * 0.666)
    ax.tick_params(width=2.0 * 0.3, length=5.0 * 0.4, which='minor')
    mpl.rcParams['axes.linewidth'] = 10

    if display_axis_lims:
        ax.spines['left'].set_position(('axes', -0.03))
        ax.spines['bottom'].set_position(('axes', -0.03))

        # set major ticks position
        ax.set_xticks(np.linspace(-np.sqrt(3), np.sqrt(3), 6))
        ax.set_yticks(np.linspace(-np.sqrt(3), np.sqrt(3), 7))

        # create minor ticks in log-positions for x
        vecs = np.log10(np.linspace(1, 10, 11)) * 2 * np.sqrt(3) / 5
        vec = []
        for k in [0.0, 1, 2, 3, 4]:
            kk = k / 5 * 2 * np.sqrt(3)
            for v in vecs:
                vec.append(kk + v)
        vec -= np.sqrt(3)
        ax.set_xticks(vec, minor=True)

        # create minor ticks in log-positions for y
        vecs = np.log10(np.linspace(1, 10, 11)) * 2 * np.sqrt(3) / 6
        vec = []
        for k in [0.0, 1, 2, 3, 4, 5]:
            kk = k / 6 * 2 * np.sqrt(3)
            for v in vecs:
                vec.append(kk + v)
        vec -= np.sqrt(3)
        ax.set_yticks(vec, minor=True)

        # set major ticks labels
        ax.set_xticklabels(['0.01','','','','', '1000'], fontsize=8.0)
        ax.set_yticklabels(['0.01','','','','','', '10000'], fontsize=8.0)

        ax.set_ylabel(r'$\mathdefault{AB-LP \;\;[nS]}$', fontsize=8.0)
        ax.set_xlabel(r'$\mathdefault{PD-LP \;\;[nS]}$', fontsize=8.0)
        ax.xaxis.labelpad = -0
        ax.yaxis.labelpad = -10
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.spines['bottom'].set_color('w')
        ax.spines['left'].set_color('w')

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))

    return ax, best_sample_probs



def plot_posterior_over_path(path1, steps, test_idcs, x_axis_scaling='dist', x_axis_labels=True, cols=None,
                             path2=None, normalize_yaxis=True, title=None, case='high_p', use_sns=True,
                             xval_ortho_start=0, verbose=False, date_today=None, save_fig=False):

    fig, ax = plt.subplots(1,1,figsize=(4.0, 1.7))

    for index in test_idcs:

        path1.get_probability_along_path(log=True, normalize=False)

        if verbose:
            print("Mean value:   ", np.mean(path1.path_probs))
            print("min value:    ", np.min(path1.path_probs))

        if x_axis_scaling == 'dist':
            path1.get_travelled_distance()
            plt.plot(path1.dists, path1.path_probs, color=cols[0], label=r'$\mathdefault{High \,\: probability \,\: path}$')
        else:
            time_ = np.linspace(0, 1, len(path1.path_probs))
            plt.plot(time_, path1.path_probs, color=cols[0], label=r'$\mathdefault{High \,\: probability \,\: path}$')

        if path2 is not None:
            k = 0
            for current_path2 in path2:
                current_path2.get_probability_along_path(log=True, normalize=False)
                if x_axis_scaling == 'dist':
                    current_path2.get_travelled_distance()
                    plt.plot(current_path2.dists+path1.dists[xval_ortho_start[k]], current_path2.path_probs, color=cols[-1], label=r'$\mathdefault{Orthogonal \,\: path}$')
                else:
                    time_ = np.linspace(0, 1, len(current_path2.path_probs))
                    plt.plot(time_, current_path2.path_probs, color=cols[-1], label=r'$\mathdefault{Orthogonal \,\: path}$')
                #plt.scatter(path1.dists[xval_ortho_start[k]], path1.path_probs[xval_ortho_start[k]], s=85.0,
                #            marker='o', zorder=5, color=cols[1])
                k+=1

        # plot start and end points
        pointscale = 0.6
        plt.plot(path1.dists[0], path1.path_probs[0], color=cols[0], marker='o',
                markeredgecolor='w', ms=16 * pointscale,
                markeredgewidth=2.7 * pointscale,
                path_effects=[pe.Stroke(linewidth=4.0 * pointscale, foreground='k'),
                              pe.Normal()])
        # plt.scatter(path1.dists[-1],
        #             path1.path_probs[-1], s=85.0,
        #             marker='o', zorder=5, color=cols[0])
        plt.plot(path1.dists[-1], path1.path_probs[-1], color=cols[1], marker='o',
                 markeredgecolor='w', ms=16 * pointscale,
                 markeredgewidth=2.7 * pointscale,
                 path_effects=[pe.Stroke(linewidth=4.0 * pointscale, foreground='k'),
                               pe.Normal()])

        # plot point labelled as 1 (good intermediate)
        plt.scatter(path1.dists[45],
                    path1.path_probs[45], s=55.0,
                    marker='o', zorder=5, color=cols[2])

        # plot point labelled as 2 (bad intermediate)
        plt.scatter(path1.dists[xval_ortho_start[0]]+path2[0].dists[-1],
                    path2[0].path_probs[-1], s=55.0,
                    marker='o', zorder=5, color=cols[3])

        steps = np.round(steps)
        ax.set_xlabel(r'$\mathdefault{Normalized \,\:distance \,\:along \,\:path}$')
        ax.set_ylabel(r'$\mathdefault{log}(p_{\theta|x}(\theta|x))$')
        #ax.set_yscale('log')
        if path2 is not None:
            plt.legend(loc='lower right')
        ax.set_title(title)
        ttl = ax.title
        ttl.set_position([.5, 1.08])
        if normalize_yaxis: ax.set_ylim([-0.02, 1.0])
        ax.set_xlim([-0.3, path1.dists[-1]+0.3])
        ax.set_ylim([-60.0, -10.0])
        if not x_axis_labels:
            ax.set_xticklabels([])
            ax.set_xticks([])

        if x_axis_labels:
            plt.xticks(range(1 + int(np.floor(path1.dists[-1]))))


        if title is not None:
            ax.set_title(title, fontsize=15)
            ttl = ax.title
            ttl.set_position([.5, 1.08])
            if save_fig:
                plt.savefig('../../thesis_results/pdf/'+date_today+'_path_prob_over_time_' + case + '_{}_title.pdf'.format(
                        index), bbox_inches='tight')
                plt.savefig('../../thesis_results/png/'+date_today+'_path_prob_over_time_' + case + '_{}_title.png'.format(
                        index), bbox_inches='tight')
                plt.savefig('../../thesis_results/svg/'+date_today+'_path_prob_over_time_' + case + '_{}_title.svg'.format(
                        index), bbox_inches='tight')

        ax.set_title('')
        if save_fig:
            plt.savefig('../../thesis_results/pdf/'+date_today+'_path_prob_over_time_'+case+'_{}.pdf'.format(
                    index), bbox_inches='tight')
            plt.savefig('../../thesis_results/png/'+date_today+'_path_prob_over_time_'+case+'_{}.png'.format(
                    index), bbox_inches='tight')
            plt.savefig('../../thesis_results/svg/'+date_today+'_path_prob_over_time_'+case+'_{}.svg'.format(
                    index), bbox_inches='tight')



def viz_path_and_samples_abstract(posterior_MoG, high_p_path, prior, lims, samples, figsize, ticks,
                                  no_contours, labels_params, start_point, end_point, path1,
                                  path_steps1, dimensions_to_use, indizes, hyperparams, ortho_path=None,
                                  ortho_p_indizes=None, high_p_indizes=None, offsets=0, linescale=1.0,
                                  seeds=None, path2=None, path_steps2=5, date_today='xx',
                                  stat_mean=None, stat_std=None,
                                  case='None', save_fig=False):
    pyloric_sim = netio.create_simulators(hyperparams)
    summ_stats = netio.create_summstats(hyperparams)
    counter = 0

    cols1 = ['#005824', '#238b45', '#41ae76']
    cols2 = ['#034e7b', '#0570b0', '#3690c0']
    cols3 = ['#990000', '#d7301f', '#ef6548']
    colsk = ['k', 'k', 'k']
    cols = [cols1, cols2, cols3]
    cols = [colsk, colsk, colsk]

    if seeds is None:
        seeds = np.ones_like(indizes)

    fig, ax = plt.subplots(3, 3, facecolor='white', figsize=figsize)

    p1g = high_p_path.path_coords[int(high_p_indizes[0])]
    p2g = high_p_path.path_coords[int(high_p_indizes[1])]
    p3g = high_p_path.path_coords[int(high_p_indizes[2])]

    p1b = ortho_path.path_coords[int(ortho_p_indizes[0])]
    p2b = ortho_path.path_coords[int(ortho_p_indizes[1])]
    p3b = ortho_path.path_coords[int(ortho_p_indizes[2])]

    for row in range(3):
        for col in range(3):
            sns.set(style="ticks", font_scale=5)
            sns.despine()
            importlib.reload(mpl)
            importlib.reload(sns)
            if row == 0:
                target_params = deepcopy(high_p_path.path_coords[int(high_p_indizes[col])]) * prior.std + prior.mean
            elif row == 1:
                if col == 0: target_params = deepcopy(high_p_path.path_coords[-1]) * prior.std + prior.mean
                if col == 2: target_params = deepcopy(high_p_path.path_coords[0]) * prior.std + prior.mean
                if col == 1:
                    if ortho_path is not None:
                        _ = plot_single_marginal_pdf(pdf1=posterior_MoG, prior=prior, resolution=200,
                                                          lims=lims, samples=np.transpose(samples), figsize=(5.0, 5.0),
                                                          ticks=False, no_contours=True, labels_params=labels_params,
                                                          start_point=high_p_path.start_point, end_point=high_p_path.end_point,
                                                          path1=high_p_path.path_coords,display_axis_lims=False,
                                                          path2=ortho_path.path_coords,
                                                          p1g=p1g, p2g=p2g, p3g=p3g,
                                                          p1b=p1b, p2b=p2b, p3b=p3b,
                                                          path_steps1=1,  path_steps2=path_steps2,
                                                          dimensions=dimensions_to_use, ax=ax[col,row])
                    else:
                        _ = plot_single_marginal_pdf(pdf1=posterior_MoG, prior=prior, resolution=200,
                                                      lims=lims, samples=np.transpose(samples), figsize=(5.0, 5.0),
                                                      ticks=False, no_contours=True, labels_params=labels_params,
                                                      p1g=p1g, p2g=p2g, p3g=p3g,
                                                      p1b=p1b, p2b=p2b, p3b=p3b,
                                                      start_point=high_p_path.start_point, end_point=high_p_path.end_point,
                                                      path1=high_p_path.path_coords, path_steps1=1, display_axis_lims=False,
                                                      dimensions=dimensions_to_use, ax=ax[col,row])
            elif row == 2:
                target_params = deepcopy(ortho_path.path_coords[int(ortho_p_indizes[col])]) * prior.std + prior.mean

            if col != 1 or row != 1:
                out_target = pyloric_sim[0].gen_single(deepcopy(target_params), seed_sim=True,
                                                       to_seed=seeds[counter])

                _ = viz_samples.vis_sample_plain(pyloric_sim[0], summ_stats, target_params, axV=ax[row,col], time_len=130000,
                                                   voltage_trace=out_target, offset=offsets[counter], col=cols[row],
                                                   test_idx=[0], case='high_p', legend=False, linescale=linescale,
                                                   title='Sample along the path of high probability in Prinz format',
                                                   date_today=date_today, counter=counter, save_fig=False)
            counter+=1

    if save_fig:
        plt.savefig('png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                    bbox_inches='tight')
        plt.savefig('svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                    bbox_inches='tight')

    return fig




def viz_path_and_samples_abstract_singleRow(posterior_MoG, high_p_path, prior, lims, samples, figsize, ticks,
                                  no_contours, labels_params, start_point, end_point, path1,
                                  path_steps1, dimensions_to_use, indizes, hyperparams, ortho_path=None,
                                  ortho_p_indizes=None, high_p_indizes=None, offsets=0, linescale=1.0,
                                  seeds=None, path2=None, path_steps2=5, date_today='xx',
                                  stat_mean=None, stat_std=None,
                                  case='None', save_fig=False):
    pyloric_sim = netio.create_simulators(hyperparams)
    summ_stats = netio.create_summstats(hyperparams)
    counter = 0

    cols1 = ['#005824', '#238b45', '#41ae76']
    cols2 = ['#034e7b', '#0570b0', '#3690c0']
    cols3 = ['#990000', '#d7301f', '#ef6548']
    colsk = ['k', 'k', 'k']
    cols = [cols1, cols2, cols3]
    cols = [colsk, colsk, colsk, colsk, colsk, colsk]

    if seeds is None:
        seeds = np.ones_like(indizes)

    gridspec = dict(hspace=0.0, wspace=0.1, width_ratios=[1, 0.05, 1, 1, 0.05, 1, 1])
    fig, ax = plt.subplots(1, 7, facecolor='white', figsize=figsize, gridspec_kw=gridspec)
    ax[1].set_visible(False)
    ax[4].set_visible(False)

    p1g = high_p_path.path_coords[int(high_p_indizes[0])]
    p2g = high_p_path.path_coords[int(high_p_indizes[1])]

    p1b = ortho_path.path_coords[int(ortho_p_indizes[0])]
    p2b = ortho_path.path_coords[int(ortho_p_indizes[1])]

    col_access = 0

    for col in range(5):
        if col_access == 1 or col_access == 4: col_access += 1

        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(sns)
        if col == 0:
            if ortho_path is not None:
                _ = plot_single_marginal_pdf(pdf1=posterior_MoG, prior=prior, resolution=200,
                                             lims=lims, samples=np.transpose(samples), figsize=(5.0, 5.0),
                                             ticks=False, no_contours=True, labels_params=labels_params,
                                             start_point=high_p_path.start_point, end_point=high_p_path.end_point,
                                             path1=high_p_path.path_coords, display_axis_lims=False,
                                             path2=ortho_path.path_coords,
                                             p1g=p1g, p2g=p2g,
                                             p1b=p1b, p2b=p2b,
                                             path_steps1=1, path_steps2=path_steps2,
                                             dimensions=dimensions_to_use, ax=ax[col_access])
            else:
                _ = plot_single_marginal_pdf(pdf1=posterior_MoG, prior=prior, resolution=200,
                                             lims=lims, samples=np.transpose(samples), figsize=(5.0, 5.0),
                                             ticks=False, no_contours=True, labels_params=labels_params,
                                             p1g=p1g, p2g=p2g,
                                             p1b=p1b, p2b=p2b,
                                             start_point=high_p_path.start_point, end_point=high_p_path.end_point,
                                             path1=high_p_path.path_coords, path_steps1=1, display_axis_lims=False,
                                             dimensions=dimensions_to_use, ax=ax[col_access])

        elif col < 3:
            target_params = deepcopy(high_p_path.path_coords[int(high_p_indizes[col-1])]) * prior.std + prior.mean
        else:
            target_params = deepcopy(ortho_path.path_coords[int(ortho_p_indizes[col-3])]) * prior.std + prior.mean

        if col > 0:
            out_target = pyloric_sim[0].gen_single(deepcopy(target_params), seed_sim=True,
                                                   to_seed=seeds[counter])
            _ = viz_samples.vis_sample_plain(pyloric_sim[0], summ_stats, target_params, axV=ax[col_access], time_len=130000,
                                               voltage_trace=out_target, offset=offsets[counter], col=cols[col],
                                               test_idx=[0], case='high_p', legend=False, linescale=linescale,
                                               title='Sample along the path of high probability in Prinz format',
                                               date_today=date_today, counter=counter, save_fig=False)
        counter+=1
        col_access += 1

    if save_fig:
        plt.savefig('png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                    bbox_inches='tight')
        plt.savefig('svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                    bbox_inches='tight')

    return fig






def viz_path_and_samples_abstract_singleRowReduced(posterior_MoG, high_p_path, prior, lims, samples, figsize, ticks,
                                  no_contours, labels_params, start_point, end_point, path1,
                                  path_steps1, dimensions_to_use, indizes, hyperparams, ortho_path=None, mycols=None,
                                  ortho_p_indizes=None, high_p_indizes=None, offsets=0, linescale=1.0,
                                  seeds=None, path2=None, path_steps2=5, date_today='xx',
                                  stat_mean=None, stat_std=None,
                                  case='None', save_fig=False):
    pyloric_sim = netio.create_simulators(hyperparams)
    summ_stats = netio.create_summstats(hyperparams)
    counter = 0

    #cols1 = ['#005824', '#238b45', '#41ae76']
    #cols2 = ['#034e7b', '#0570b0', '#3690c0']
    #cols3 = ['#990000', '#d7301f', '#ef6548']
    #colsk = ['k', 'k', 'k']
    #cols = [cols1, cols2, cols3]

    color_mixture = 0.5 * (np.asarray(list(mycols['CONSISTENT1'])) + np.asarray(list(mycols['CONSISTENT2'])))

    cols = [mycols['CONSISTENT2'], mycols['CONSISTENT2'], color_mixture, mycols['INCONSISTENT']]

    if seeds is None:
        seeds = np.ones_like(indizes)

    gridspec = dict(hspace=0.0, wspace=0.08, width_ratios=[1, 1, 1, 1])
    fig, ax = plt.subplots(1, 4, facecolor='white', figsize=figsize, gridspec_kw=gridspec)
    #ax[3].set_visible(False)

    p1g = high_p_path.path_coords[int(high_p_indizes[0])]
    p2g = high_p_path.path_coords[int(high_p_indizes[1])]

    p1b = ortho_path.path_coords[int(ortho_p_indizes[0])]

    col_access = 0

    for col in range(4):
        #if col_access == 3: col_access += 1

        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(sns)
        if col == 0:
            if ortho_path is not None:
                _ = plot_single_marginal_pdf(pdf1=posterior_MoG, prior=prior, resolution=200,
                                             lims=lims, samples=np.transpose(samples), figsize=(5.0, 5.0),
                                             ticks=False, no_contours=True, labels_params=labels_params,
                                             start_point=high_p_path.start_point, end_point=high_p_path.end_point,
                                             path1=high_p_path.path_coords, display_axis_lims=False,
                                             path2=ortho_path.path_coords,
                                             p1g=p1g, p2g=p2g, start_col=mycols['GT'], end_col=mycols['CONSISTENT1'],
                                             p1b=p1b, current_col1=color_mixture,current_col=mycols['CONSISTENT2'],
                                             current_col2=mycols['INCONSISTENT'],
                                             path_steps1=1, path_steps2=path_steps2,
                                             dimensions=dimensions_to_use, ax=ax[col_access])
            else:
                _ = plot_single_marginal_pdf(pdf1=posterior_MoG, prior=prior, resolution=200,
                                             lims=lims, samples=np.transpose(samples), figsize=(5.0, 5.0),
                                             ticks=False, no_contours=True, labels_params=labels_params,
                                             p1g=p1g, p2g=p2g, start_col=mycols['GT'], end_col=mycols['CONSISTENT1'],
                                             p1b=p1b, current_col1=color_mixture, current_col=mycols['CONSISTENT2'],
                                             current_col2=mycols['INCONSISTENT'],
                                             start_point=high_p_path.start_point, end_point=high_p_path.end_point,
                                             path1=high_p_path.path_coords, path_steps1=1, display_axis_lims=False,
                                             dimensions=dimensions_to_use, ax=ax[col_access])

        elif col < 3:
            target_params = deepcopy(high_p_path.path_coords[int(high_p_indizes[col-1])]) * prior.std + prior.mean
        else:
            target_params = deepcopy(ortho_path.path_coords[int(ortho_p_indizes[col-3])]) * prior.std + prior.mean

        if col > 0:
            out_target = pyloric_sim[0].gen_single(deepcopy(target_params), seed_sim=True,
                                                   to_seed=seeds[counter])
            ss = summ_stats.calc([out_target])[0]
            _ = viz_samples.vis_sample_plain(pyloric_sim[0], summ_stats, target_params, axV=ax[col_access], time_len=165000,
                                               voltage_trace=out_target, offset=offsets[counter], col=cols[col],
                                               test_idx=[0], case='high_p', legend=False, linescale=linescale,
                                               title='Sample along the path of high probability in Prinz format',
                                               date_today=date_today, counter=counter, save_fig=False)
        counter+=1
        col_access += 1

    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)

    if save_fig:
        plt.savefig('png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                    bbox_inches='tight')
        plt.savefig('svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                    bbox_inches='tight')

    return fig





def viz_path_and_samples_abstract_twoRows(posterior_MoG, high_p_path, prior, lims, samples, figsize, ticks,
                                  no_contours, labels_params, start_point, end_point, path1,
                                  path_steps1, dimensions_to_use, indizes, hyperparams, ortho_path=None, mycols=None,
                                  ortho_p_indizes=None, high_p_indizes=None, offsets=0, linescale=1.0,
                                  seeds=None, path2=None, path_steps2=5, date_today='xx', time_len=165000,
                                  stat_mean=None, stat_std=None,
                                  case='None', save_fig=False):
    pyloric_sim = netio.create_simulators(hyperparams)
    summ_stats = netio.create_summstats(hyperparams)
    counter = 0

    #cols1 = ['#005824', '#238b45', '#41ae76']
    #cols2 = ['#034e7b', '#0570b0', '#3690c0']
    #cols3 = ['#990000', '#d7301f', '#ef6548']
    #colsk = ['k', 'k', 'k']
    #cols = [cols1, cols2, cols3]

    scalebar = [False, True, False, False]

    color_mixture = 0.5 * (np.asarray(list(mycols['CONSISTENT1'])) + np.asarray(list(mycols['CONSISTENT2'])))

    cols = [mycols['CONSISTENT1'], mycols['CONSISTENT2'], color_mixture, mycols['INCONSISTENT']]

    if seeds is None:
        seeds = np.ones_like(indizes)

    gridspec = dict(hspace=0.03, wspace=0.03, width_ratios=[1, 1], height_ratios=[1, 1])
    fig, ax = plt.subplots(2, 2, facecolor='white', figsize=figsize, gridspec_kw=gridspec)

    p1g = high_p_path.path_coords[int(high_p_indizes[0])]
    p2g = high_p_path.path_coords[int(high_p_indizes[1])]

    p1b = ortho_path.path_coords[int(ortho_p_indizes[0])]

    col_access = 0
    row_access = 0
    print_label = True

    for col in range(4):

        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(sns)
        col_access = col % 2
        if col < 3:
            target_params = deepcopy(high_p_path.path_coords[int(high_p_indizes[col])]) * prior.std + prior.mean
        else:
            target_params = deepcopy(ortho_path.path_coords[int(ortho_p_indizes[col-3])]) * prior.std + prior.mean

        out_target = pyloric_sim[0].gen_single(deepcopy(target_params), seed_sim=True,
                                               to_seed=seeds[counter])
        ss = summ_stats.calc([out_target])[0]
        if col > 1: print_label=False

        _ = viz_samples.vis_sample_plain(pyloric_sim[0], summ_stats, target_params, axV=ax[row_access, col_access], time_len=time_len,
                                           voltage_trace=out_target, offset=offsets[counter], col=cols[col], scale_bar=scalebar[counter],
                                           test_idx=[0], case='high_p', legend=False, linescale=linescale, print_label=print_label,
                                           title='Sample along the path of high probability in Prinz format',
                                           date_today=date_today, counter=counter, save_fig=False)
        if counter == 1:
            row_access += 1

        counter+=1

    for aa in ax:
        for a in aa:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)

    if save_fig:
        plt.savefig('png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                    bbox_inches='tight')
        plt.savefig('svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                    bbox_inches='tight')

    return fig



def viz_path_and_samples_abstract_twoRows_31DSynthetic(posterior_MoG, high_p_path, prior, lims, samples, figsize, ticks,
                                  no_contours, labels_params, start_point, end_point, path1,
                                  path_steps1, dimensions_to_use, indizes, hyperparams, ortho_path=None, mycols=None,
                                  ortho_p_indizes=None, high_p_indizes=None, offsets=0, linescale=1.0,
                                  seeds=None, path2=None, path_steps2=5, date_today='xx', time_len=165000,
                                  stat_mean=None, stat_std=None,
                                  case='None', save_fig=False):
    pyloric_sim = netio.create_simulators(hyperparams)
    summ_stats = netio.create_summstats(hyperparams)
    counter = 0

    #cols1 = ['#005824', '#238b45', '#41ae76']
    #cols2 = ['#034e7b', '#0570b0', '#3690c0']
    #cols3 = ['#990000', '#d7301f', '#ef6548']
    #colsk = ['k', 'k', 'k']
    #cols = [cols1, cols2, cols3]

    scalebar = [False, False, False, False]
    draw_patch = [False, False, False, False]

    color_mixture = 0.5 * (np.asarray(list(mycols['CONSISTENT1'])) + np.asarray(list(mycols['CONSISTENT2'])))

    cols = [mycols['CONSISTENT1'], mycols['CONSISTENT2'], color_mixture, mycols['INCONSISTENT']]

    if seeds is None:
        seeds = np.ones_like(indizes)

    gridspec = dict(hspace=0.03, wspace=0.03, width_ratios=[1, 1], height_ratios=[1, 1])
    fig, ax = plt.subplots(2, 2, facecolor='white', figsize=figsize, gridspec_kw=gridspec)

    p1g = high_p_path.path_coords[int(high_p_indizes[0])]
    p2g = high_p_path.path_coords[int(high_p_indizes[1])]

    p1b = ortho_path.path_coords[int(ortho_p_indizes[0])]

    col_access = 0
    row_access = 0
    print_label = True

    for col in range(4):

        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(sns)
        col_access = col % 2
        if col < 3:
            target_params = deepcopy(high_p_path.path_coords[int(high_p_indizes[col])]) * prior.std + prior.mean
        else:
            target_params = deepcopy(ortho_path.path_coords[int(ortho_p_indizes[col-3])]) * prior.std + prior.mean

        out_target = pyloric_sim[0].gen_single(deepcopy(target_params), seed_sim=True,
                                               to_seed=seeds[counter])
        ss = summ_stats.calc([out_target])[0]
        if col > 0: print_label=False

        _ = viz_samples.vis_sample_plain_31DSynthetic(pyloric_sim[0], summ_stats, target_params, axV=ax[row_access, col_access], time_len=time_len,
                                           voltage_trace=out_target, offset=offsets[counter], col=cols[col], scale_bar=scalebar[counter],
                                           test_idx=[0], case='high_p', legend=False, linescale=linescale, print_label=print_label, draw_patch=draw_patch[counter],
                                           title='Sample along the path of high probability in Prinz format',
                                           date_today=date_today, counter=counter, save_fig=False)
        if counter == 1:
            row_access += 1

        counter+=1

    for aa in ax:
        for a in aa:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)

    if save_fig:
        plt.savefig('png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                    bbox_inches='tight')
        plt.savefig('svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                    bbox_inches='tight')

    return fig





def calc_lims(params):

    if params.comp_neurons is not None:
        neuron_1 = np.asarray(netio.create_neurons(params.comp_neurons[0]))
        neuron_2 = np.asarray(netio.create_neurons(params.comp_neurons[1]))
        membrane_cond_mins = np.minimum(neuron_1, neuron_2)
        membrane_cond_maxs = np.maximum(neuron_1, neuron_2)
        membrane_cond_mins[membrane_cond_mins==0.0]   += 1e-20
        membrane_cond_maxs[membrane_cond_maxs == 0.0] += 1e-20
    else:
        low_val = 0.0
        membrane_cond_mins = np.asarray([[100, 2.5,     2,       10, 5,       50,  0.01,    low_val ], # PM
                                         [100, low_val, 4,       20, low_val, 25,  low_val, 0.02    ],   # LP
                                         [100, 2.5,     low_val, 40, low_val, 75,  low_val, low_val ]]) * 0.628e-3 # PY

        # contains the maximal values that were used by Prinz et al.
        membrane_cond_maxs = np.asarray([[400, 5.0,     6,       50, 10,      125, 0.01,    low_val ], # PM
                                         [100, low_val, 10,      50, 5,       100, 0.05,    0.03    ],   # LP
                                         [500, 10,      2,       50, low_val, 125, 0.05,    0.03    ]]) * 0.628e-3 # PY

    security_factor = 1.25 # factor that we increase the margin used by Prinz et al. with
    membrane_cond_mins /= security_factor
    membrane_cond_maxs *= security_factor

    use_membrane = np.asarray(params.use_membrane)
    membrane_used_mins = membrane_cond_mins[use_membrane == True].flatten()
    membrane_used_maxs = membrane_cond_maxs[use_membrane == True].flatten()

    syn_dim_mins = np.ones_like(params.true_params) * params.syn_min # syn_min is the start of uniform interval
    syn_dim_maxs = np.ones_like(params.true_params) * params.syn_max # syn_max is the end of uniform interval

    syn_dim_mins = np.log(syn_dim_mins)
    syn_dim_maxs = np.log(syn_dim_maxs)

    membrane_and_sny_mins = np.concatenate((membrane_used_mins, syn_dim_mins))
    membrane_and_sny_maxs = np.concatenate((membrane_used_maxs, syn_dim_maxs))

    lims = (membrane_and_sny_mins.tolist(), membrane_and_sny_maxs.tolist())
    lims = np.asarray(lims)
    dimensions = np.sum(params.use_membrane) + 7
    lims = np.tile(lims, [dimensions, 1]) if lims.ndim == 1 else lims
    lims = np.transpose(lims)
    return lims


def get_labels(params, mathmode=False, include_q10=True):
    if params.comp_neurons is None:
        membrane_conds = np.asarray(params.use_membrane)
        membrane_names = [['AB_{Na}', 'AB_{CaT}', 'AB_{CaS}', 'AB_{A}', 'AB_{KCa}', 'AB_{Kd}', 'AB_{H}', 'AB_{leak}'],
                          ['LP_{Na}', 'LP_{CaT}', 'LP_{CaS}', 'LP_{A}', 'LP_{KCa}', 'LP_{Kd}', 'LP_{H}', 'LP_{leak}'],
                          ['PY_{Na}', 'PY_{CaT}', 'PY_{CaS}', 'PY_{A}', 'PY_{KCa}', 'PY_{Kd}', 'PY_{H}', 'PY_{leak}']]
        if mathmode:
            membrane_names = [
                [r'$\mathrm{AB}_\mathrm{Na}$', r'$\mathrm{AB}_\mathrm{CaT}$', r'$\mathrm{AB}_\mathrm{CaS}$', r'$\mathrm{AB}_\mathrm{A}$', r'$\mathrm{AB}_\mathrm{KCa}$', r'$\mathrm{AB}_\mathrm{Kd}$', r'$\mathrm{AB}_\mathrm{H}$', r'$\mathrm{AB}_\mathrm{leak}$'],
                [r'$\mathrm{LP}_\mathrm{Na}$', r'$\mathrm{LP}_\mathrm{CaT}$', r'$\mathrm{LP}_\mathrm{CaS}$', r'$\mathrm{LP}_\mathrm{A}$', r'$\mathrm{LP}_\mathrm{KCa}$', r'$\mathrm{LP}_\mathrm{Kd}$', r'$\mathrm{LP}_\mathrm{H}$', r'$\mathrm{LP}_\mathrm{leak}$'],
                [r'$\mathrm{PY}_\mathrm{Na}$', r'$\mathrm{PY}_\mathrm{CaT}$', r'$\mathrm{PY}_\mathrm{CaS}$', r'$\mathrm{PY}_\mathrm{A}$', r'$\mathrm{PY}_\mathrm{KCa}$', r'$\mathrm{PY}_\mathrm{Kd}$', r'$\mathrm{PY}_\mathrm{H}$', r'$\mathrm{PY}_\mathrm{leak}$']]
        membrane_names = np.asarray(membrane_names)
        relevant_membrane_names = membrane_names[membrane_conds]
        synapse_names = np.asarray([pick_synapse(num) for num in range(7)])
        relevant_labels = np.concatenate((relevant_membrane_names, synapse_names))
        #q10_names = [u'Q_{10} g\u0305_{glut}', u'Q_{10} g\u0305_{chol}', r'Q_{10} \tau_{glut}', r'Q_{10} \tau_{chol}']
        if include_q10:
            q10_names = ['Q_{10} Na', 'Q_{10} CaT', 'Q_{10} CaS', 'Q_{10} CaA', 'Q_{10} KCa', 'Q_{10} Kd', 'Q_{10} H', 'Q_{10} leak', u'Q_{10} g\u0305_{glut}', u'Q_{10} g\u0305_{chol}']
            relevant_labels = np.concatenate((relevant_labels, q10_names))
    else:
        assert 'This case is not implemented yet.'
    return relevant_labels


def get_labels_8pt(params, mathmode=False, include_q10=True):
    if params.comp_neurons is None:
        membrane_conds = np.asarray(params.use_membrane)
        membrane_names = [['AB-Na', 'AB-CaT', 'AB-CaS', 'AB-A', 'AB-KCa', 'AB-Kd', 'AB-H', 'AB-leak'],
                          ['LP-Na', 'LP-CaT', 'LP-CaS', 'LP-A', 'LP-KCa', 'LP-Kd', 'LP-H', 'LP-leak'],
                          ['PY-Na', 'PY-CaT', 'PY-CaS', 'PY-A', 'PY-KCa', 'PY-Kd', 'PY-H', 'PY-leak']]
        if mathmode:
            membrane_names = [
                [r'$\mathrm{AB}_\mathrm{Na}$', r'$\mathrm{AB}_\mathrm{CaT}$', r'$\mathrm{AB}_\mathrm{CaS}$', r'$\mathrm{AB}_\mathrm{A}$', r'$\mathrm{AB}_\mathrm{KCa}$', r'$\mathrm{AB}_\mathrm{Kd}$', r'$\mathrm{AB}_\mathrm{H}$', r'$\mathrm{AB}_\mathrm{leak}$'],
                [r'$\mathrm{LP}_\mathrm{Na}$', r'$\mathrm{LP}_\mathrm{CaT}$', r'$\mathrm{LP}_\mathrm{CaS}$', r'$\mathrm{LP}_\mathrm{A}$', r'$\mathrm{LP}_\mathrm{KCa}$', r'$\mathrm{LP}_\mathrm{Kd}$', r'$\mathrm{LP}_\mathrm{H}$', r'$\mathrm{LP}_\mathrm{leak}$'],
                [r'$\mathrm{PY}_\mathrm{Na}$', r'$\mathrm{PY}_\mathrm{CaT}$', r'$\mathrm{PY}_\mathrm{CaS}$', r'$\mathrm{PY}_\mathrm{A}$', r'$\mathrm{PY}_\mathrm{KCa}$', r'$\mathrm{PY}_\mathrm{Kd}$', r'$\mathrm{PY}_\mathrm{H}$', r'$\mathrm{PY}_\mathrm{leak}$']]
        membrane_names = np.asarray(membrane_names)
        relevant_membrane_names = membrane_names[membrane_conds]
        synapse_names = np.asarray([pick_synapse(num, True) for num in range(7)])
        relevant_labels = np.concatenate((relevant_membrane_names, synapse_names))
        #q10_names = [u'Q_{10} g\u0305_{glut}', u'Q_{10} g\u0305_{chol}', r'Q_{10} \tau_{glut}', r'Q_{10} \tau_{chol}']
        if include_q10:
            q10_names = ['Q_{10} Na', 'Q_{10} CaT', 'Q_{10} CaS', 'Q_{10} CaA', 'Q_{10} KCa', 'Q_{10} Kd', 'Q_{10} H', 'Q_{10} leak', u'Q_{10} g\u0305_{glut}', u'Q_{10} g\u0305_{chol}']
            relevant_labels = np.concatenate((relevant_labels, q10_names))
    else:
        assert 'This case is not implemented yet.'
    return relevant_labels

def get_labels_8pt_supp(params, mathmode=False, include_q10=True):
    if params.comp_neurons is None:
        membrane_conds = np.asarray(params.use_membrane)
        membrane_names = [[r'$\mathdefault{AB-Na}\;$  ', r'$\mathdefault{AB-CaT}\;$  ', r'$\mathdefault{AB-CaS}\;$  ', r'$\mathdefault{AB-A}\;$  ', r'$\mathdefault{AB-KCa}\;$  ', r'$\mathdefault{AB-Kd}\;$  ', r'$\mathdefault{AB-H}\;$  ', r'$\mathdefault{AB-leak}\;$  '],
                          [r'$\mathdefault{LP-Na}\;$  ', r'$\mathdefault{LP-CaT}\;$  ', r'$\mathdefault{LP-CaS}\;$  ', r'$\mathdefault{LP-A}\;$  ', r'$\mathdefault{LP-KCa}\;$  ', r'$\mathdefault{LP-Kd}\;$  ', r'$\mathdefault{LP-H}\;$  ', r'$\mathdefault{LP-leak}\;$  '],
                          [r'$\mathdefault{PY-Na}\;$  ', r'$\mathdefault{PY-CaT}\;$  ', r'$\mathdefault{PY-CaS}\;$  ', r'$\mathdefault{PY-A}\;$  ', r'$\mathdefault{PY-KCa}\;$  ', r'$\mathdefault{PY-Kd}\;$  ', r'$\mathdefault{PY-H}\;$  ', r'$\mathdefault{PY-leak}\;$  ']]
        if mathmode:
            membrane_names = [
                [r'$\mathrm{AB}_\mathrm{Na}$', r'$\mathrm{AB}_\mathrm{CaT}$', r'$\mathrm{AB}_\mathrm{CaS}$', r'$\mathrm{AB}_\mathrm{A}$', r'$\mathrm{AB}_\mathrm{KCa}$', r'$\mathrm{AB}_\mathrm{Kd}$', r'$\mathrm{AB}_\mathrm{H}$', r'$\mathrm{AB}_\mathrm{leak}$'],
                [r'$\mathrm{LP}_\mathrm{Na}$', r'$\mathrm{LP}_\mathrm{CaT}$', r'$\mathrm{LP}_\mathrm{CaS}$', r'$\mathrm{LP}_\mathrm{A}$', r'$\mathrm{LP}_\mathrm{KCa}$', r'$\mathrm{LP}_\mathrm{Kd}$', r'$\mathrm{LP}_\mathrm{H}$', r'$\mathrm{LP}_\mathrm{leak}$'],
                [r'$\mathrm{PY}_\mathrm{Na}$', r'$\mathrm{PY}_\mathrm{CaT}$', r'$\mathrm{PY}_\mathrm{CaS}$', r'$\mathrm{PY}_\mathrm{A}$', r'$\mathrm{PY}_\mathrm{KCa}$', r'$\mathrm{PY}_\mathrm{Kd}$', r'$\mathrm{PY}_\mathrm{H}$', r'$\mathrm{PY}_\mathrm{leak}$']]
        membrane_names = np.asarray(membrane_names)
        relevant_membrane_names = membrane_names[membrane_conds]
        synapse_names = np.asarray([pick_synapse(num, True) for num in range(7)])
        relevant_labels = np.concatenate((relevant_membrane_names, synapse_names))
        #q10_names = [u'Q_{10} g\u0305_{glut}', u'Q_{10} g\u0305_{chol}', r'Q_{10} \tau_{glut}', r'Q_{10} \tau_{chol}']
        if include_q10:
            q10_names = ['Q_{10} Na', 'Q_{10} CaT', 'Q_{10} CaS', 'Q_{10} CaA', 'Q_{10} KCa', 'Q_{10} Kd', 'Q_{10} H', 'Q_{10} leak', u'Q_{10} g\u0305_{glut}', u'Q_{10} g\u0305_{chol}']
            relevant_labels = np.concatenate((relevant_labels, q10_names))
    else:
        assert 'This case is not implemented yet.'
    return relevant_labels


def get_labels_asterisk(params, mathmode=False):
    if params.comp_neurons is None:
        membrane_conds = np.asarray(params.use_membrane)
        membrane_names = [['AB_{Na}', 'AB_{CaT}', 'AB_{CaS}', 'AB_{A}', 'AB_{KCa}', 'AB_{Kd}', 'AB_{H}', 'AB_{leak}'],
                          ['LP_{Na}', 'LP_{CaT}', 'LP_{CaS}', 'LP_{A}', 'LP_{KCa}', 'LP_{Kd}', 'LP_{H}', 'LP_{leak}'],
                          ['PY_{Na}', 'PY_{CaT}', 'PY_{CaS}', 'PY_{A}', 'PY_{KCa}', 'PY_{Kd}', 'PY_{H}', 'PY_{leak}']]
        if mathmode:
            membrane_names = [
                [r'$\mathrm{AB}_\mathrm{Na}$', r'$\mathrm{AB}_\mathrm{CaT}^{*}$', r'$\mathrm{AB}_\mathrm{CaS}^{+}$', r'$\mathrm{AB}_\mathrm{A}^{*}$', r'$\mathrm{AB}_\mathrm{KCa}^{+}$', r'$\mathrm{AB}_\mathrm{Kd}$', r'$\mathrm{AB}_\mathrm{H}^{-}$', r'$\mathrm{AB}_\mathrm{leak}^{-}$'],
                [r'$\mathrm{LP}_\mathrm{Na}$', r'$\mathrm{LP}_\mathrm{CaT}^{*}$', r'$\mathrm{LP}_\mathrm{CaS}^{+}$', r'$\mathrm{LP}_\mathrm{A}^{*}$', r'$\mathrm{LP}_\mathrm{KCa}^{+}$', r'$\mathrm{LP}_\mathrm{Kd}$', r'$\mathrm{LP}_\mathrm{H}^{-}$', r'$\mathrm{LP}_\mathrm{leak}^{-}$'],
                [r'$\mathrm{PY}_\mathrm{Na}$', r'$\mathrm{PY}_\mathrm{CaT}^{*}$', r'$\mathrm{PY}_\mathrm{CaS}^{+}$', r'$\mathrm{PY}_\mathrm{A}^{*}$', r'$\mathrm{PY}_\mathrm{KCa}^{+}$', r'$\mathrm{PY}_\mathrm{Kd}$', r'$\mathrm{PY}_\mathrm{H}^{-}$', r'$\mathrm{PY}_\mathrm{leak}^{-}$']]
        membrane_names = np.asarray(membrane_names)
        relevant_membrane_names = membrane_names[membrane_conds]
        synapse_names = np.asarray([pick_synapse(num, mathmode) for num in range(7)])
        relevant_labels = np.concatenate((relevant_membrane_names, synapse_names))
    else:
        assert 'This case is not implemented yet.'
    return relevant_labels



def viz_path_and_samples_presentation_Nmar(high_p_path, prior, lims, samples, figsize, ticks,
                                          no_contours, labels_params, start_point, end_point, path1,
                                          path_steps1, dimensions_to_use, indizes, hyperparams,
                                          num_dim_input=None,posterior_MoG=None, ortho_path=None,
                                          profile_posterior=False, log_profile=False, mode='13D',
                                          optimize_profile_posterior=False, sample_probs=None,
                                          params_mean_log=None, show_membrane_conds=None,offsets=None,
                                          params_std_log=None, resolution=40, pdf_type=None,
                                          scale_bar=True, stat_scale=None, current_col='g',
                                          seeds=None, path2=None, path_steps2=5, date_today='xx',
                                          stat_mean=None, stat_std=None, tf=False, mode_for_membrane_height=None,
                                          case='None', save_fig=False):

    if stat_scale == 'dataset': assert (stat_mean is not None) and (stat_std is not None), \
        'Provide mean and std of dataset for scaling (or use other scaling method).'

    if tf: hyperparams.transform = False
    pyloric_sim = netio.create_simulators(hyperparams)
    summ_stats = netio.create_summstats(hyperparams)
    counter = 0
    final_indizes = []
    #final_indizes.append(indizes[0])
    for current_ind in indizes:
        final_indizes.append(current_ind)
    #final_indizes.append(indizes[-1])
    if seeds is None:
        seeds = np.ones_like(final_indizes)
    for index in final_indizes:
        #if counter == 0:
        #    curr_col = 'orange'
        #elif counter == len(final_indizes)-1:
        #    curr_col = 'b'
        #else:
        curr_col = current_col
        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(plt)
        importlib.reload(sns)
        import matplotlib.gridspec as gridspec

        if case=='high_p': target_params = deepcopy(high_p_path.path_coords[int(index)])
        if case=='ortho_p': target_params = deepcopy(ortho_path.path_coords[int(index)])
        if tf: target_params_sim = target_params * (prior.upper - prior.lower) + prior.lower
        else: target_params_sim = target_params
        target_params_sim = target_params * prior.std + prior.mean
        out_target = pyloric_sim[0].gen_single(deepcopy(target_params_sim), seed_sim=True,
                                               to_seed=seeds[counter])  # params.true_params gives the synaptic strengths

        fig = plt.figure(figsize=(12, 5.86))
        outer = gridspec.GridSpec(1, 2, wspace=0.08, hspace=0.2, width_ratios=[0.5, 0.5])

        params_ = out_target['params']
        stats = summ_stats.calc([out_target])[0]

        if show_membrane_conds is not None:
            params_trunc = params_[show_membrane_conds].tolist()
            params_trunc += params_[-7:].tolist()
            params_ = np.asarray(params_trunc)

        if hyperparams.include_plateau:
            stats = stats[:-4]
        if stat_scale == 'dataset':
            stats = (stats - stat_mean) / stat_std
        if stat_scale == 'experimental':
            stats = scale_to_experimental(stats)
        if index == indizes[0]:
            max_stats = np.asarray(deepcopy(stats))
            min_stats = np.asarray(deepcopy(stats))
            max_conds = np.asarray(deepcopy(params_))
            min_conds = np.asarray(deepcopy(params_))
            max_stats[np.isnan(max_stats)] = 0.0
            min_stats[np.isnan(min_stats)] = 0.0
        else:
            stats = np.asarray(stats)
            params_ = np.asarray(params_)
            max_stats[max_stats < stats] = stats[max_stats < stats]
            min_stats[min_stats > stats] = stats[min_stats > stats]
            max_conds[max_conds < params_] = params_[max_conds < params_]
            min_conds[min_conds > params_] = params_[min_conds > params_]

        n_d_to_use = len(dimensions_to_use)

        for i in range(2):

            if i == 0:
                #ax = plt.Subplot(fig, path_ax[0])
                path_ax = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[0], height_ratios=[0.33,0.33,0.34],
                                                      width_ratios=[0.33,0.33,0.34],
                                                      wspace=0.1, hspace=0.1)
                names = get_labels(hyperparams, mathmode=True, include_q10=False)#[:-7]
                if ortho_path is not None:
                    path_ax, im = plot_N_2D_marginals(pdf1=posterior_MoG, prior=prior, resolution=resolution,
                                                  start_col='orange', end_col='b', path_col='w', current_col=curr_col,
                                                  lims=lims, samples=np.transpose(samples), figsize=(5.0, 5.0),
                                                  ticks=False, no_contours=True, labels_params=labels_params,
                                                  start_point=start_point, end_point=end_point,
                                                  profile_posterior=profile_posterior,
                                                  optimize_profile_posterior=optimize_profile_posterior,
                                                  sample_probs=sample_probs, log_profile=log_profile,
                                                  pdf_type=pdf_type, smooth_MAF=False,
                                                  labels=names[dimensions_to_use], fig=fig,
                                                  params_mean_log=params_mean_log, params_std_log=params_std_log,
                                                  path1=high_p_path.path_coords,
                                                  path2=ortho_path.path_coords, num_dim_input=num_dim_input,
                                                  path_steps1=1,  path_steps2=path_steps2,
                                                  dimensions=dimensions_to_use, current_point=[target_params], ax=path_ax)
                else:
                    path_ax, im = plot_N_2D_marginals(pdf1=posterior_MoG, prior=prior, resolution=resolution,
                                                  start_col='orange', end_col='b', path_col='w', current_col=curr_col,
                                                  lims=lims, samples=np.transpose(samples), figsize=(5.0, 5.0),
                                                  ticks=False, no_contours=True, labels_params=labels_params,
                                                  start_point=start_point, end_point=end_point,
                                                  profile_posterior=profile_posterior,
                                                  optimize_profile_posterior=optimize_profile_posterior,
                                                  sample_probs=sample_probs, log_profile=log_profile,
                                                  pdf_type=pdf_type, smooth_MAF=False,
                                                  labels=names[dimensions_to_use], fig=fig,
                                                  params_mean_log=params_mean_log, params_std_log=params_std_log,
                                                  path1=high_p_path.path_coords, path_steps1=1, num_dim_input=num_dim_input,
                                                  dimensions=dimensions_to_use, current_point=[target_params], ax=path_ax)
            else:
                axV = plt.Subplot(fig, outer[1])

                axV = \
                    viz_samples.vis_sample_subfig_twitter(pyloric_sim[0], summ_stats,
                                                  target_params, stats=stats, test_idx=[0],
                                                  case='ortho_p_short', legend=False,
                                                  max_stats=max_stats, min_stats=min_stats,
                                                  max_conds=max_conds, min_conds=min_conds,
                                                  hyperparams=hyperparams, stat_scale=stat_scale,
                                                  stat_mean=stat_mean, stat_std=stat_std,
                                                  mem_dimensions=show_membrane_conds,
                                                  voltage_trace=out_target, axV=axV, mode=mode,
                                                  mode_for_membrane_height=mode_for_membrane_height,
                                                  scale_bar=scale_bar, offset=offsets[counter],
                                                  date_today=date_today, current_col=curr_col,
                                                  counter=counter, save_fig=False)
                fig.add_subplot(axV)

        if save_fig:
            plt.savefig('tweet/sample_prinz_' + case + '_{}.png'.format(counter),
                        bbox_inches='tight', dpi=300)

        counter += 1
        plt.show()
    return im




def viz_barplots_over_path(high_p_path, prior, lims, samples, figsize, ticks,
                                          no_contours, labels_params, start_point, end_point, path1,
                                          path_steps1, dimensions_to_use, indizes, hyperparams,
                                          num_dim_input=None,posterior_MoG=None, ortho_path=None,
                                          profile_posterior=False, log_profile=False, mode='13D',
                                          optimize_profile_posterior=False, sample_probs=None,
                                          params_mean_log=None, show_membrane_conds=None, labels_=True,
                                          params_std_log=None, resolution=40, pdf_type=None, param_names=True,
                                          scale_bar=True, stat_scale=None, current_col='g', ss_names=True,
                                          seeds=None, path2=None, path_steps2=5, date_today='xx',
                                          stat_mean=None, stat_std=None, tf=False, mode_for_membrane_height=None,
                                          case='None', save_fig=False):

    if stat_scale == 'dataset': assert (stat_mean is not None) and (stat_std is not None), \
        'Provide mean and std of dataset for scaling (or use other scaling method).'

    if tf: hyperparams.transform = False
    pyloric_sim = netio.create_simulators(hyperparams)
    summ_stats = netio.create_summstats(hyperparams)
    counter = 0
    final_indizes = []
    #final_indizes.append(indizes[0])
    for current_ind in indizes:
        final_indizes.append(current_ind)
    #final_indizes.append(indizes[-1])
    if seeds is None:
        seeds = np.ones_like(final_indizes)
    for index in final_indizes:
        if counter == 0:
            curr_col = 'orange'
        elif counter == len(final_indizes)-1:
            curr_col = 'b'
        else:
            curr_col = current_col
        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(plt)
        importlib.reload(sns)
        import matplotlib.gridspec as gridspec

        if case=='high_p': target_params = deepcopy(high_p_path.path_coords[int(index)])
        if case=='ortho_p': target_params = deepcopy(ortho_path.path_coords[int(index)])
        if tf: target_params_sim = target_params * (prior.upper - prior.lower) + prior.lower
        else: target_params_sim = target_params
        target_params_sim = target_params * prior.std + prior.mean
        out_target = pyloric_sim[0].gen_single(deepcopy(target_params_sim), seed_sim=True,
                                               to_seed=seeds[counter])  # params.true_params gives the synaptic strengths

        fig = plt.figure(figsize=(10, 1.5))
        outer = gridspec.GridSpec(1, 2, wspace=0.13, hspace=0.2, width_ratios=[0.335, 0.665])

        params_ = out_target['params']
        stats = summ_stats.calc([out_target])[0]

        if show_membrane_conds is not None:
            params_trunc = params_[show_membrane_conds].tolist()
            params_trunc += params_[-7:].tolist()
            params_ = np.asarray(params_trunc)

        if hyperparams.include_plateau:
            stats = stats[:-4]
        if stat_scale == 'dataset':
            stats = (stats - stat_mean) / stat_std
        if stat_scale == 'experimental':
            stats = scale_to_experimental(stats)
        if index == indizes[0]:
            max_stats = np.asarray(deepcopy(stats))
            min_stats = np.asarray(deepcopy(stats))
            max_conds = np.asarray(deepcopy(params_))
            min_conds = np.asarray(deepcopy(params_))
            max_stats[np.isnan(max_stats)] = 0.0
            min_stats[np.isnan(min_stats)] = 0.0
        else:
            stats = np.asarray(stats)
            params_ = np.asarray(params_)
            max_stats[max_stats < stats] = stats[max_stats < stats]
            min_stats[min_stats > stats] = stats[min_stats > stats]
            max_conds[max_conds < params_] = params_[max_conds < params_]
            min_conds[min_conds > params_] = params_[min_conds > params_]

        if index == final_indizes[-1]:
            gs = gridspec.GridSpec(1, 3, width_ratios=[len(stats), len(params_[:-7]), len(params_[-7:])], wspace=0.3, hspace=0.1)

            axss = plt.Subplot(fig, gs[0])
            axmemparams = plt.Subplot(fig, gs[1])
            axsynparams = plt.Subplot(fig, gs[2])
            axss, axmemparams, axsynparams = \
                viz_samples.vis_sample_subfig_no_voltage(pyloric_sim[0], summ_stats,
                                              target_params, stats=stats, test_idx=[0],
                                              case='ortho_p_short', legend=False,
                                              max_stats=max_stats, min_stats=min_stats,
                                              max_conds=max_conds, min_conds=min_conds,
                                              hyperparams=hyperparams, stat_scale=stat_scale,
                                              stat_mean=stat_mean, stat_std=stat_std,ss_names=ss_names,
                                              mem_dimensions=show_membrane_conds, param_names=param_names,
                                              voltage_trace=out_target, mode=mode, labels_=labels_,
                                              axss=axss, axmemparams=axmemparams, mode_for_membrane_height=mode_for_membrane_height,
                                              axsynparams=axsynparams, scale_bar=scale_bar,
                                              date_today=date_today, current_col=curr_col,
                                              counter=counter, save_fig=False)
            fig.add_subplot(axss)
            fig.add_subplot(axmemparams)
            fig.add_subplot(axsynparams)

            if save_fig:
                plt.savefig('../../thesis_results/pdf/' + date_today + '_sample_prinz_' + case + '_{}.pdf'.format(counter),
                            bbox_inches='tight')
                plt.savefig('../../thesis_results/png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                            bbox_inches='tight', dpi=300)
                plt.savefig('../../thesis_results/svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                            bbox_inches='tight')

        counter += 1


def viz_ss_over_samples(all_ss,
          indizes, hyperparams, mode='13D', figsize=(3.5*1.35, 1.6),
          color_input='k', show_membrane_conds=None, labels_=True,
          param_names=True, scale_bar=True, stat_scale=None, ss_names=True,
          date_today='xx', stat_mean=None, stat_std=None, tf=False,
          mode_for_membrane_height=None, case='None', save_fig=False):

    if stat_scale == 'dataset': assert (stat_mean is not None) and (stat_std is not None), \
        'Provide mean and std of dataset for scaling (or use other scaling method).'

    summ_stats = netio.create_summstats(hyperparams)
    if tf: hyperparams.transform = False
    counter = 0

    for index, ss in zip(indizes, all_ss):
        curr_col = 'k'
        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(plt)
        importlib.reload(sns)
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(1, 2, wspace=0.13, hspace=0.2, width_ratios=[0.335, 0.665])

        stats = ss
        if np.any(np.isnan(stats)):
            print('Warning: NaN in stats in iteration', counter)

        if hyperparams.include_plateau:
            stats = stats[:-4]
        if stat_scale == 'dataset':
            stats = (stats - stat_mean) / stat_std
        if stat_scale == 'experimental':
            stats = scale_to_experimental(stats)
        if index == indizes[0]:
            max_stats = np.asarray(deepcopy(stats))
            min_stats = np.asarray(deepcopy(stats))
            max_stats[np.isnan(max_stats)] = 0.0
            min_stats[np.isnan(min_stats)] = 0.0
        else:
            stats = np.asarray(stats)
            max_stats[max_stats < stats] = stats[max_stats < stats]
            min_stats[min_stats > stats] = stats[min_stats > stats]

        if index == indizes[-1]:
            gs = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.1)

            axss = plt.Subplot(fig, gs[0])
            axss = \
                viz_samples.vis_ss_barplot(np.zeros(1), summ_stats,
                                              np.zeros(31), stats=stats, test_idx=[0],
                                              case='ortho_p_short', legend=False,
                                              max_stats=max_stats, min_stats=min_stats,
                                              hyperparams=hyperparams, stat_scale=stat_scale,
                                              stat_mean=stat_mean, stat_std=stat_std,ss_names=ss_names,
                                              mem_dimensions=show_membrane_conds, param_names=param_names,
                                              mode=mode, labels_=labels_,
                                              with_ss=True, with_params=False,
                                              axss=axss,  mode_for_membrane_height=mode_for_membrane_height,
                                              scale_bar=scale_bar, color_input=color_input,
                                              date_today=date_today, current_col=curr_col,
                                              counter=counter, save_fig=False)
            fig.add_subplot(axss)

            if save_fig:
                plt.savefig('../../thesis_results/pdf/' + date_today + '_sample_prinz_' + case + '_{}.pdf'.format(counter),
                            bbox_inches='tight')
                plt.savefig('../../thesis_results/png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                            bbox_inches='tight', dpi=300)
                plt.savefig('../../thesis_results/svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                            bbox_inches='tight')

        counter += 1



def viz_barplots_over_path_v2ss(high_p_path, out_targets, prior, lims, samples, ticks,
                                          no_contours, labels_params, start_point, end_point, path1,
                                          path_steps1, dimensions_to_use, indizes, hyperparams,
                                          num_dim_input=None,posterior_MoG=None, ortho_path=None,
                                          profile_posterior=False, log_profile=False, mode='13D', figsize=(3.5*1.35, 1.6),
                                          optimize_profile_posterior=False, sample_probs=None, color_input='k',
                                          params_mean_log=None, show_membrane_conds=None, labels_=True,
                                          params_std_log=None, resolution=40, pdf_type=None, param_names=True,
                                          scale_bar=True, stat_scale=None, current_col='g', ss_names=True,
                                          seeds=None, path2=None, path_steps2=5, date_today='xx',
                                          stat_mean=None, stat_std=None, tf=False, mode_for_membrane_height=None,
                                          case='None', save_fig=False):

    if stat_scale == 'dataset': assert (stat_mean is not None) and (stat_std is not None), \
        'Provide mean and std of dataset for scaling (or use other scaling method).'

    summ_stats = netio.create_summstats(hyperparams)
    if tf: hyperparams.transform = False
    counter = 0

    for index, out_target in zip(indizes, out_targets):
        curr_col = 'k'
        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(plt)
        importlib.reload(sns)
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(1, 2, wspace=0.13, hspace=0.2, width_ratios=[0.335, 0.665])

        params_ = out_target['params']
        stats = summ_stats.calc([out_target])[0]
        if np.any(np.isnan(stats)):
            print('Warning: NaN in stats in iteration', counter)

        if show_membrane_conds is not None:
            params_trunc = params_[show_membrane_conds].tolist()
            params_trunc += params_[-7:].tolist()
            params_ = np.asarray(params_trunc)

        if hyperparams.include_plateau:
            stats = stats[:-4]
        if stat_scale == 'dataset':
            stats = (stats - stat_mean) / stat_std
        if stat_scale == 'experimental':
            stats = scale_to_experimental(stats)
        if index == indizes[0]:
            max_stats = np.asarray(deepcopy(stats))
            min_stats = np.asarray(deepcopy(stats))
            max_conds = np.asarray(deepcopy(params_))
            min_conds = np.asarray(deepcopy(params_))
            max_stats[np.isnan(max_stats)] = 0.0
            min_stats[np.isnan(min_stats)] = 0.0
        else:
            stats = np.asarray(stats)
            params_ = np.asarray(params_)
            max_stats[max_stats < stats] = stats[max_stats < stats]
            min_stats[min_stats > stats] = stats[min_stats > stats]
            max_conds[max_conds < params_] = params_[max_conds < params_]
            min_conds[min_conds > params_] = params_[min_conds > params_]

        if index == indizes[-1]:
            gs = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.1)

            axss = plt.Subplot(fig, gs[0])
            axss = \
                viz_samples.vis_sample_subfig_no_voltage(np.zeros(1), summ_stats,
                                              np.zeros(31), stats=stats, test_idx=[0],
                                              case='ortho_p_short', legend=False,
                                              max_stats=max_stats, min_stats=min_stats,
                                              max_conds=max_conds, min_conds=min_conds,
                                              hyperparams=hyperparams, stat_scale=stat_scale,
                                              stat_mean=stat_mean, stat_std=stat_std,ss_names=ss_names,
                                              mem_dimensions=show_membrane_conds, param_names=param_names,
                                              voltage_trace=out_target, mode=mode, labels_=labels_,
                                              with_ss=True, with_params=False,
                                              axss=axss,  mode_for_membrane_height=mode_for_membrane_height,
                                              scale_bar=scale_bar, color_input=color_input,
                                              date_today=date_today, current_col=curr_col,
                                              counter=counter, save_fig=False)
            fig.add_subplot(axss)

            if save_fig:
                plt.savefig('../../thesis_results/pdf/' + date_today + '_sample_prinz_' + case + '_{}.pdf'.format(counter),
                            bbox_inches='tight')
                plt.savefig('../../thesis_results/png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                            bbox_inches='tight', dpi=300)
                plt.savefig('../../thesis_results/svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                            bbox_inches='tight')

        counter += 1



def viz_barplots_over_path_v2params(high_p_path, prior, lims, samples, figsize, ticks,
                                          no_contours, labels_params, start_point, end_point, path1,
                                          path_steps1, dimensions_to_use, indizes, hyperparams,
                                          num_dim_input=None,posterior_MoG=None, ortho_path=None,
                                          profile_posterior=False, log_profile=False, mode='13D',
                                          optimize_profile_posterior=False, sample_probs=None, color_input='k',
                                          params_mean_log=None, show_membrane_conds=None, labels_=True,
                                          params_std_log=None, resolution=40, pdf_type=None, param_names=True,
                                          scale_bar=True, stat_scale=None, current_col='g', ss_names=True,
                                          seeds=None, path2=None, path_steps2=5, date_today='xx',
                                          stat_mean=None, stat_std=None, tf=False, mode_for_membrane_height=None,
                                          case='None', save_fig=False):

    if stat_scale == 'dataset': assert (stat_mean is not None) and (stat_std is not None), \
        'Provide mean and std of dataset for scaling (or use other scaling method).'

    if tf: hyperparams.transform = False
    pyloric_sim = netio.create_simulators(hyperparams)
    summ_stats = netio.create_summstats(hyperparams)
    counter = 0
    final_indizes = []
    #final_indizes.append(indizes[0])
    for current_ind in indizes:
        final_indizes.append(current_ind)
    #final_indizes.append(indizes[-1])
    if seeds is None:
        seeds = np.ones_like(final_indizes)
    for index in final_indizes:
        if counter == 0:
            curr_col = 'orange'
        elif counter == len(final_indizes)-1:
            curr_col = 'b'
        else:
            curr_col = current_col
        sns.set(style="ticks", font_scale=5)
        sns.despine()
        importlib.reload(mpl)
        importlib.reload(plt)
        importlib.reload(sns)
        import matplotlib.gridspec as gridspec

        if case=='high_p': target_params = deepcopy(high_p_path.path_coords[int(index)])
        if case=='ortho_p': target_params = deepcopy(ortho_path.path_coords[int(index)])
        if tf: target_params_sim = target_params * (prior.upper - prior.lower) + prior.lower
        else: target_params_sim = target_params
        target_params_sim = target_params * prior.std + prior.mean

        fig = plt.figure(figsize=figsize)

        params_ = target_params_sim
        target_params_sim[-7:] = np.exp(target_params_sim[-7:])

        if show_membrane_conds is not None:
            params_trunc = params_[show_membrane_conds].tolist()
            params_trunc += params_[-7:].tolist()
            params_ = np.asarray(params_trunc)

        if index == indizes[0]:
            max_conds = np.asarray(deepcopy(params_))
            min_conds = np.asarray(deepcopy(params_))
        else:
            params_ = np.asarray(params_)
            max_conds[max_conds < params_] = params_[max_conds < params_]
            min_conds[min_conds > params_] = params_[min_conds > params_]

        if index == final_indizes[-1]:
            gs = gridspec.GridSpec(1, 2, width_ratios=[len(params_[:-7]), len(params_[-7:])], wspace=0.20, hspace=0.1)

            axmemparams = plt.Subplot(fig, gs[0])
            axsynparams = plt.Subplot(fig, gs[1])
            axmemparams, axsynparams = \
                viz_samples.vis_sample_subfig_no_voltage(pyloric_sim[0], summ_stats,
                                              target_params, stats=np.zeros(15), test_idx=[0],
                                              case='ortho_p_short', legend=False,
                                              max_stats=np.zeros(15), min_stats=np.zeros(15),
                                              max_conds=max_conds, min_conds=min_conds,
                                              hyperparams=hyperparams, stat_scale=stat_scale,
                                              stat_mean=stat_mean, stat_std=stat_std,ss_names=ss_names,
                                              mem_dimensions=show_membrane_conds, param_names=param_names,
                                              voltage_trace=np.zeros(15), mode=mode, labels_=labels_,
                                              with_ss=False, with_params=True, color_input=color_input,
                                              axmemparams=axmemparams, mode_for_membrane_height=mode_for_membrane_height,
                                              axsynparams=axsynparams, scale_bar=scale_bar,
                                              date_today=date_today, current_col=curr_col,
                                              counter=counter, save_fig=False)
            fig.add_subplot(axmemparams)
            fig.add_subplot(axsynparams)

            if save_fig:
                plt.savefig('../../thesis_results/pdf/' + date_today + '_sample_prinz_' + case + '_{}.pdf'.format(counter),
                            bbox_inches='tight')
                plt.savefig('../../thesis_results/png/' + date_today + '_sample_prinz_' + case + '_{}.png'.format(counter),
                            bbox_inches='tight', dpi=300)
                plt.savefig('../../thesis_results/svg/' + date_today + '_sample_prinz_' + case + '_{}.svg'.format(counter),
                            bbox_inches='tight')

        counter += 1


def plot_N_2D_marginals(pdf1=None, lims=None, pdf2=None, prior=None, contours=False, levels=(0.68, 0.95),
             start_col='orange', end_col='b', path_col='w', path_col2='w', current_col='g', current_col2='g', num_dim_input=None,
             optimize_profile_posterior=False, num_profile_samples=100000, eval_pdf=False, sample_probs=None, log_profile=False,
             params_mean_log=None, params_std_log=None, resolution=500, labels_params=None, ticks=False, diag_only=False, smooth_MAF=True,
             diag_only_cols=4, diag_only_rows=4, figsize=(5, 5), fontscale=1, no_contours=False, profile_posterior=False,
             partial=False, samples=None, start_point=None, end_point=None, current_point=None, current_point2=None,
             labels=None, fig=None,
             path1=None, path2=None, path3=None, path_steps1=5, path_steps2=5, col1='k', col2='b', col3='g', pdf_type='MAF',
             dimensions=None, title=None, figname=None, ax=None):
    """Plots marginals of a pdf, for each variable and pair of variables.
ax.plot(path1[dim_j][0:-1:path_steps1], path1[dim_i][0:-1:path_steps1],
                      color=path_col, lw=2.8,
                      path_effects=[pe.Stroke(linewidth=3.8, foreground='k'), pe.Normal()])
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
    samples: array
        If given, samples of a distribution are plotted along `pdf`.
        If given, `pdf` is plotted with default `levels` (0.68, 0.95), if provided `levels` is None.
        If given, `lims` is overwritten and taken to be the respective
        limits of the samples in each dimension.
    col1 : str
        color 1
    col2 : str
        color 2
    col3 : str
        color 3 (for pdf2 if provided)
    """

    pdfs = (pdf1, pdf2)
    colrs = (col2, col3)
    if path1 is not None: path1 = np.transpose(path1)
    if path2 is not None: path2 = np.transpose(path2)
    if path3 is not None: path3 = np.transpose(path3)

    plots_2D = len(dimensions) - 1

    params_mean = prior.mean
    params_std = prior.std

    num_dim = num_dim_input

    if not (pdf1 is None or pdf2 is None):
        assert pdf1.ndim == pdf2.ndim

    if samples is not None:
        contours = True
        if levels is None:
            levels = (0.68, 0.95)


    if samples is not None and lims is None:
        lims_min = np.min(samples, axis=1)
        lims_max = np.max(samples, axis=1)
        lims = np.concatenate(
            (lims_min.reshape(-1, 1), lims_max.reshape(-1, 1)), axis=1)
    else:
        lims = np.asarray(lims)
        lims = np.tile(lims, [num_dim, 1]) if lims.ndim == 1 else lims

    if samples is not None and profile_posterior:
        if sample_probs is None:
            samples_trunc = samples.T[:num_profile_samples]
            sample_probs = pdf1.eval(samples_trunc, log=False)
        else:
            samples_trunc = samples.T[:len(sample_probs)]
        samplesAppendP = np.concatenate((samples_trunc, np.asarray([sample_probs]).T), axis=1)

    rows = 1
    cols = 1

    if ax is None:
        fig, ax = plt.subplots(rows, cols, facecolor='white', figsize=figsize)

    pdf = pdfs[0]

    if pdf is None:
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_axis_off()

    best_sample_probs = np.zeros((resolution, resolution))
    ax_i = 0
    for i in range(plots_2D+1):
        dim_i = dimensions[i]
        ax_j = 0
        for j in range(plots_2D+1):
            dim_j = dimensions[j]
            if i != j and i < j:

                ax_plot = plt.Subplot(fig, ax[i, j-1])

                if samples is not None:
                    H, xedges, yedges = np.histogram2d(
                        samples[dim_i, :], samples[dim_j, :], bins=resolution, range=[
                        [lims[dim_i, 0], lims[dim_i, 1]], [lims[dim_j, 0], lims[dim_j, 1]]], density=True)
                    if smooth_MAF:
                        gkernel = gkern(kernlen=10)
                        H = ss.convolve2d(H, gkernel, mode='valid')
                    ax_plot.imshow(H, origin='lower', extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
                              vmin=0.0, vmax=np.min(H)+(np.max(H)-np.min(H))*1.0)
                    best_sample_probs = H

                mymarker = 'o'
                myfactor = 1.2
                mymarkersize = 9*myfactor
                mymarkeredge = 1.2*myfactor
                strokelinewidth = 2.5*myfactor
                path_size = 4.0*1.0  # 5.5
                path_stroke = 5.5*1.0
                if path1 is not None:
                    ax_plot.plot(path1[dim_j][::path_steps1], path1[dim_i][::path_steps1],
                                  color=path_col, lw=path_size,
                                  path_effects=[pe.Stroke(linewidth=path_stroke, foreground='k'), pe.Normal()])
                if path2 is not None:
                    ax_plot.plot(path2[dim_j][0::path_steps2], path2[dim_i][0::path_steps2],
                            color=path_col2, lw=path_size,
                            path_effects=[pe.Stroke(linewidth=path_stroke, foreground='k'), pe.Normal()])
                if path3 is not None:
                    ax_plot.plot(path3[dim_j][0::path_steps2], path3[dim_i][0::path_steps2],
                            color=path_col2, lw=path_size,
                            path_effects=[pe.Stroke(linewidth=path_stroke, foreground='k'), pe.Normal()])
                if start_point is not None:
                    ax_plot.plot(start_point[dim_j], start_point[dim_i], color=start_col, marker=mymarker, markeredgecolor='w', ms=mymarkersize,
                                  markeredgewidth=mymarkeredge, path_effects=[pe.Stroke(linewidth=strokelinewidth, foreground='k'), pe.Normal()])
                if end_point is not None:
                    ax_plot.plot(end_point[dim_j], end_point[dim_i], color=end_col, marker=mymarker, markeredgecolor='w', ms=mymarkersize,
                                  markeredgewidth=mymarkeredge, path_effects=[pe.Stroke(linewidth=strokelinewidth, foreground='k'), pe.Normal()])
                if current_point is not None:
                    for current_p in current_point:
                        ax_plot.plot(current_p[dim_j], current_p[dim_i], color=current_col, marker=mymarker, markeredgecolor='w', ms=mymarkersize,
                                      markeredgewidth=mymarkeredge, path_effects=[pe.Stroke(linewidth=strokelinewidth, foreground='k'), pe.Normal()])
                if current_point2 is not None:
                    for current_p in current_point2:
                        ax_plot.plot(current_p[dim_j], current_p[dim_i], color=current_col2, marker=mymarker, markeredgecolor='w', ms=mymarkersize,
                                      markeredgewidth=mymarkeredge, path_effects=[pe.Stroke(linewidth=strokelinewidth, foreground='k'), pe.Normal()])

                ax_plot.set_xlim(lims[dim_j])
                ax_plot.set_ylim(lims[dim_i])

                axis_dim_i = dimensions[i + 1]
                axis_dim_j = dimensions[j-1]


                ax_plot.get_xaxis().set_ticks([])
                ax_plot.get_yaxis().set_ticks([])
                ax_plot.spines['bottom'].set_color('w')
                ax_plot.spines['left'].set_color('w')

                x0, x1 = ax_plot.get_xlim()
                y0, y1 = ax_plot.get_ylim()
                ax_plot.set_aspect((x1 - x0) / (y1 - y0))

                ax_j += 1

                fig.add_subplot(ax_plot)
        ax_i += 1

    return ax, best_sample_probs
