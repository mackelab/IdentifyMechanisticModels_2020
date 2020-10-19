import delfi.utils.colormaps as cmaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import time


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

def plot_multipdf(pdfs, lims=None, gt=None,
             resolution=500, labels_params=None, ticks=False, diag_only=False,
             diag_only_cols=4, diag_only_rows=4, figsize=(5, 5), fontscale=1,
             partial=False,partial_ls=None, samples=None, col1='k', colrs=('c'),alpha=1,imageshow=True):
    """Plots marginals of a pdf, for each variable and pair of variables.

    Parameters
    ----------
    pdfs : list of objects
    lims : array
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
    partial_ls: list of integers
        Indices of parameters plotted if `partial` is set to True. Minimum of 3 indices.
    samples: array
        If given, samples of a distribution are plotted along `pdf`.
        If given, `lims` is overwritten and taken to be the respective
        limits of the samples in each dimension.
    col1 : str
        color 1
    col2 : list of str
        colors 2
    imageshow : bool
        If True, imshow is active
    """

    pdf1 = pdfs[-1]

    levels = [0.95]
            
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
            #ax.get_yaxis().set_tick_params(which='both', direction='out')
            ax.get_xaxis().set_tick_params(which='both', direction='out')
            ax.set_xticks(np.linspace(lims[0, 0], lims[0, 1], 2))
            #ax.set_yticks(np.linspace(min(pp), max(pp), 2))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            #ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            ax.get_yaxis().set_ticks([])
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
                
                if partial and partial_ls is not None:
                    p_i = partial_ls[i]
                    p_j = partial_ls[j]
                else:
                    p_i = i
                    p_j = j


                if i == j:
                    if samples is not None:
                        ax[i, j].hist(samples[p_i, :], bins=100, normed=True,
                                      color=col1,
                                      edgecolor=col1)
                    xx = np.linspace(lims[p_i, 0], lims[p_i, 1], resolution)

                    for pdf, col in zip(pdfs, colrs):
                        if pdf is not None:
                            pp = pdf.eval(xx, ii=[p_i], log=False)

                            if diag_only:
                                c += 1
                            else:
                                r = i
                                c = j
                    pp_ls = []
                    for pdf, col in zip(pdfs, colrs):
                        if pdf is not None:
                            pp = pdf.eval(xx, ii=[p_i], log=False)
                            ax[r, c].plot(xx, pp,color=col,alpha=alpha)
                            pp_ls.append(pp)

                    ax[r, c].set_xlim(lims[p_i])
                    ax[r, c].set_ylim([0, ax[r, c].get_ylim()[1]])

                    if gt is not None:
                        ax[r, c].vlines(
                            gt[p_i], 0, ax[r, c].get_ylim()[1], color='r')

                    if ticks:
                        #ax[r, c].get_yaxis().set_tick_params(
                        #    which='both', direction='out', labelsize=fontscale * 15)
                        ax[r, c].get_xaxis().set_tick_params(
                            which='both', direction='out', labelsize=fontscale * 15)
#                         ax[r, c].locator_params(nbins=3)
                        ax[r, c].set_xticks(np.linspace(
                            lims[p_i, 0]+0.15*np.abs(lims[p_i, 0]-lims[p_j, 1]), lims[p_j, 1]-0.15*np.abs(lims[p_i, 0]-lims[p_j, 1]), 2))
                        #ax[r, c].set_yticks(np.linspace(0+0.15*np.abs(0-np.max(pp_ls)), np.max(pp_ls)-0.15*np.abs(0-np.max(pp_ls)), 2))
                        ax[r, c].xaxis.set_major_formatter(
                            mpl.ticker.FormatStrFormatter('%.1f'))
                        if lims[p_j, 1]>1e3:
                            ax[r, c].xaxis.set_major_formatter(
                            mpl.ticker.FormatStrFormatter('%.0f'))
                        
                        #ax[r, c].yaxis.set_major_formatter(
                        #    mpl.ticker.FormatStrFormatter('%.1f'))
                        ax[r, c].get_yaxis().set_ticks([])
                    else:
                        ax[r, c].get_xaxis().set_ticks([])
                        ax[r, c].get_yaxis().set_ticks([])

                    if labels_params is not None:
                        ax[r, c].set_xlabel(
                            labels_params[p_i], fontsize=fontscale * 20)
                    else:
                        ax[r, c].set_xlabel([])

                    x0, x1 = ax[r, c].get_xlim()
                    y0, y1 = ax[r, c].get_ylim()
                    ax[r, c].set_aspect((x1 - x0) / (y1 - y0))

                    if partial and i == rows - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...', fontsize=fontscale * 25)
                        plt.text(x1 + (x1 - x0) / 8.4, y0 - (y1 - y0) /
                                 6., '...', fontsize=fontscale * 25, rotation=-45)

                else:
                    if diag_only:
                        continue

                    if i < j:
                        pdf = pdfs[-1]
                    else:
                        pdf = None

                    if pdf is None:
                        ax[i, j].get_yaxis().set_visible(False)
                        ax[i, j].get_xaxis().set_visible(False)
                        ax[i, j].set_axis_off()
                        continue

                    if samples is not None:
                        H, xedges, yedges = np.histogram2d(
                            samples[p_i, :], samples[p_j, :], bins=30, range=[
                            [lims[p_i, 0], lims[p_i, 1]], [lims[p_j, 0], lims[p_j, 1]]], normed=True)
                        ax[i, j].imshow(H, origin='lower', extent=[
                                        yedges[0], yedges[-1], xedges[0], xedges[-1]])

                    xx = np.linspace(lims[p_i, 0], lims[p_i, 1], resolution)
                    yy = np.linspace(lims[p_j, 0], lims[p_j, 1], resolution)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate(
                        [X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                    pp = pdf.eval(xy, ii=[p_i, p_j], log=False)
                    pp = pp.reshape(list(X.shape))
                    
                    if imageshow==True:
                        ax[i, j].imshow(pp.T, origin='lower',
                                            extent=[lims[p_j, 0], lims[p_j, 1], lims[p_i, 0], lims[p_i, 1]],
                                            aspect='auto', interpolation='none')
                    
                    for pdf,col in zip(pdfs[0:-1],colrs[0:-1]):
                        pp = pdf.eval(xy, ii=[p_i, p_j], log=False)
                        pp = pp.reshape(list(X.shape))
                        ax[i, j].contour(Y, X, probs2contours(pp, levels), levels, colors=[col],alpha=alpha)

                    ax[i, j].set_xlim(lims[p_j])
                    ax[i, j].set_ylim(lims[p_i])

                    if gt is not None:
                        ax[i, j].plot(gt[p_j], gt[p_i], 'r.', ms=10,
                                      markeredgewidth=0.0)

                    ax[i, j].get_xaxis().set_ticks([])
                    ax[i, j].get_yaxis().set_ticks([])
                    if imageshow==True:
                        ax[i, j].set_axis_off()
                    else:
                        ax[i, j].set_axis_on()

                    x0, x1 = ax[i, j].get_xlim()
                    y0, y1 = ax[i, j].get_ylim()
                    ax[i, j].set_aspect((x1 - x0) / (y1 - y0))

                    if partial and j == cols - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...', fontsize=fontscale * 25)

                if diag_only and c == cols - 1:
                    c = -1
                    r += 1

    return fig, ax

def plot_pdf_multipts(pdf1, lims, pdf2=None, gt=None, param1=None, contours=False, levels=(0.68, 0.95),
             resolution=500, labels_params=None, ticks=False, diag_only=False,
             diag_only_cols=4, diag_only_rows=4, figsize=(5, 5), fontscale=1,
             partial=False,partial_ls=None, samples=None, col1='k', col2='b', col3='g',col_samp=None):
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
    partial_ls: list of integers
        Indices of parameters plotted if `partial` is set to True. Minimum of 3 indices.
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
    col_samp : list
        If defined, colormap for samples provided in gt
    """

    pdfs = (pdf1, pdf2)
    colrs = (col2, col3)
    
    if col_samp is None:
        col_min = 1
        num_colors = 20+col_min
        cm1 = mpl.cm.Reds
        col_samp = [cm1(1.*i/num_colors) for i in range(col_min,num_colors)]
        col_samp[0] = col_samp[13]
        col_samp[1:] = [cm1(1.*6/num_colors) for i in range(col_min,num_colors)]

    if not (pdf1 is None or pdf2 is None):
        assert pdf1.ndim == pdf2.ndim

    if samples is not None:
        contours = True
        if levels is None:
            levels = (0.68, 0.95)
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
            for samp in range(len(gt[0])-1,-1,-1):
                ax.vlines(gt[0][samp], 0, ax.get_ylim()[1], color=col_samp[samp])

        if ticks:
            #ax.get_yaxis().set_tick_params(which='both', direction='out')
            ax.get_xaxis().set_tick_params(which='both', direction='out')
            ax.set_xticks(np.linspace(lims[0, 0], lims[0, 1], 2))
            #ax.set_yticks(np.linspace(min(pp), max(pp), 2))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            #ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            ax.get_yaxis().set_ticks([])
        else:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    else:

        if not diag_only:
            if partial:
                rows = min(max(3,len(partial_ls)), pdf1.ndim)
                cols = min(max(3,len(partial_ls)), pdf1.ndim)
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
                
                if partial and partial_ls is not None:
                    p_i = partial_ls[i]
                    p_j = partial_ls[j]
                else:
                    p_i = i
                    p_j = j

                if i == j:
                    if samples is not None:
                        ax[i, j].hist(samples[p_i, :], bins=100, normed=True,
                                      color=col1,
                                      edgecolor=col1)
                    xx = np.linspace(lims[p_i, 0], lims[p_i, 1], resolution)

                    for pdf, col in zip(pdfs, colrs):
                        if pdf is not None:
                            pp = pdf.eval(xx, ii=[p_i], log=False)

                            if diag_only:
                                c += 1
                            else:
                                r = i
                                c = j

                    for pdf, col in zip(pdfs, colrs):
                        if pdf is not None:
                            pp = pdf.eval(xx, ii=[p_i], log=False)
                            ax[r, c].plot(xx, pp, color=col)

                    ax[r, c].set_xlim(lims[p_i])
                    ax[r, c].set_ylim([0, ax[r, c].get_ylim()[1]])

                    if gt is not None:
                        for samp in range(len(gt[0])-1,-1,-1):
                            ax[r, c].vlines(
                                gt[p_i][samp], 0, ax[r, c].get_ylim()[1], color=col_samp[samp])
                        
                    if param1 is not None:
                        ax[r, c].vlines(
                            param1[p_i], 0, ax[r, c].get_ylim()[1], color=(244/255, 152/255, 25/255))

                    if ticks:
                        #ax[r, c].get_yaxis().set_tick_params(
                        #    which='both', direction='out', labelsize=fontscale * 15)
                        ax[r, c].get_xaxis().set_tick_params(
                            which='both', direction='out', labelsize=fontscale * 15)
#                         ax[r, c].locator_params(nbins=3)
                        ax[r, c].set_xticks(np.linspace(
                            lims[p_i, 0]+0.15*np.abs(lims[p_i, 0]-lims[p_j, 1]), lims[p_j, 1]-0.15*np.abs(lims[p_i, 0]-lims[p_j, 1]), 2))
                        #ax[r, c].set_yticks(np.linspace(0+0.15*np.abs(0-max(pp)), max(pp)-0.15*np.abs(0-max(pp)), 2))
                        ax[r, c].xaxis.set_major_formatter(
                            mpl.ticker.FormatStrFormatter('%.1f'))
                        if lims[p_j, 1]>1e3:
                            ax[r, c].xaxis.set_major_formatter(
                            mpl.ticker.FormatStrFormatter('%.0f'))
                        
                        #ax[r, c].yaxis.set_major_formatter(
                        #    mpl.ticker.FormatStrFormatter('%.1f'))
                        ax[r, c].get_yaxis().set_ticks([])
                    else:
                        ax[r, c].get_xaxis().set_ticks([])
                        ax[r, c].get_yaxis().set_ticks([])

                    if labels_params is not None:
                        ax[r, c].set_xlabel(
                            labels_params[p_i], fontsize=fontscale * 20)
                    else:
                        ax[r, c].set_xlabel([])

                    x0, x1 = ax[r, c].get_xlim()
                    y0, y1 = ax[r, c].get_ylim()
                    ax[r, c].set_aspect((x1 - x0) / (y1 - y0))

                    if partial and i == rows - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...', fontsize=fontscale * 25)
                        plt.text(x1 + (x1 - x0) / 8.4, y0 - (y1 - y0) /
                                 6., '...', fontsize=fontscale * 25, rotation=-45)

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
                            samples[p_i, :], samples[p_j, :], bins=30, normed=True)
                        ax[i, j].imshow(H, origin='lower', extent=[
                                        yedges[0], yedges[-1], xedges[0], xedges[-1]])

                    xx = np.linspace(lims[p_i, 0], lims[p_i, 1], resolution)
                    yy = np.linspace(lims[p_j, 0], lims[p_j, 1], resolution)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate(
                        [X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                    pp = pdf.eval(xy, ii=[p_i, p_j], log=False)
                    pp = pp.reshape(list(X.shape))
                    if contours:
                        ax[i, j].contour(Y, X, probs2contours(
                            pp, levels), levels, colors=('w', 'y'))
                    else:
                        ax[i, j].imshow(pp.T, origin='lower',
                                        extent=[lims[p_j, 0], lims[p_j, 1], lims[p_i, 0], lims[p_i, 1]],
                                        aspect='auto', interpolation='none')
                    ax[i, j].set_xlim(lims[p_j])
                    ax[i, j].set_ylim(lims[p_i])

                    if gt is not None:
                        for samp in range(len(gt[0])-1,-1,-1):
                            ax[i, j].plot(gt[p_j][samp], gt[p_i][samp],color=col_samp[samp], marker='.',linestyle = 'none', ms=10,
                                          markeredgewidth=0.0)
                    
                    if param1 is not None:
                        ax[i, j].plot(param1[p_j], param1[p_i], '.', color=(244/255, 152/255, 25/255), ms=10,
                                      markeredgewidth=0.0)

                    ax[i, j].get_xaxis().set_ticks([])
                    ax[i, j].get_yaxis().set_ticks([])
                    ax[i, j].set_axis_off()

                    x0, x1 = ax[i, j].get_xlim()
                    y0, y1 = ax[i, j].get_ylim()
                    ax[i, j].set_aspect((x1 - x0) / (y1 - y0))

                    if partial and j == cols - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...', fontsize=fontscale * 25)

                if diag_only and c == cols - 1:
                    c = -1
                    r += 1

    return fig, ax