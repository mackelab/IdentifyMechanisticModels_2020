import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from print_helper import conductance_to_value_exp, build_string, build_string_gen
import seaborn as sns
from matplotlib import lines
import matplotlib.gridspec as gridspec
import prinzdb
from print_helper import get_summ_stat_name, get_summ_stat_name_text, get_synapse_name, get_summ_stat_name_asterisk, scale_to_experimental
import sys
sys.path.append("../visualization")
import viz
from copy import deepcopy
import matplotlib.ticker
import matplotlib.patheffects as pe


def vis_sample(m, s, sample, hyperparams, t_on=None, t_off=None, with_ss=True, with_params=True,
               mem_dimensions=None,mode2=None,
               voltage_trace=None, time_len=None, fontscale=1.0, linescale=1.0, offset=0.0,
               test_idx=None, case=None, title=None, date_today=None, counter=0, offset_labels=0.0, legend=True,
               multiplier_cond_shift = 0.0, vis_legend=True, scale_bar=True,
               ss_names=True, param_names=True, save_fig=False):
    """
    Function of Kaan, modified by Michael. Used for plotting fig 5b Prinz.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    font_size = 15.0
    if voltage_trace is None:
        data = m.gen_single(sample)
    else:
        data = voltage_trace

    Vx = data['data']
    params = data['params']

    stats = s.calc([data])[0]
    stats_nan = deepcopy(stats)
    stats[np.isnan(stats)] = 0.0
    if hyperparams.include_plateau:
        stats = stats[:-4]
    stats = scale_to_experimental(stats)

    bar_scaling_factors = [1.0, 10, 100, 10, 100, 1, 10000, 10000]
    bar_scaling_factors = np.reshape(np.tile(bar_scaling_factors, 3), (3, 8))
    bar_vals = bar_scaling_factors[np.asarray(hyperparams.use_membrane)]

    if mem_dimensions is not None:
        params_trunc = params[mem_dimensions].tolist()
        params_trunc += params[-7:].tolist()
        bar_vals = bar_vals[mem_dimensions]
        params = np.asarray(params_trunc)

    if with_params and with_ss:
        fig = plt.figure(figsize=(11.3, 6))
        gs = gridspec.GridSpec(2, 3, width_ratios=[len(stats), len(params[:-7]), len(params[-7:])],
                               wspace=0.25, height_ratios=[0.7, 0.3])
        axV = plt.subplot(gs[0, :])
        axss = plt.subplot(gs[1, 0])
        axmemparams = plt.subplot(gs[1, 1])
        axsynparams = plt.subplot(gs[1, 2])
    elif with_params:
        fig = plt.figure(figsize=(6, 7.5))
        gs = gridspec.GridSpec(2, 2, width_ratios=[len(params[:-7]), len(params[-7:])],
                               hspace=0.1, wspace=0.38, height_ratios=[0.65, 0.35])
        axV = plt.subplot(gs[0, :])
        axmemparams = plt.subplot(gs[1, 0])
        axsynparams = plt.subplot(gs[1, 1])
    elif with_ss:
        fig, (axV, axss) = plt.subplots(2, figsize=(14, 6))
    else:
        fig, axV = plt.subplots(1, figsize=(14, 3))

    cols = ['#034e7b', '#0570b0', '#3690c0']
    #cols = ['k', 'k', 'k']
    current_col = 0

    scale_bar_breadth = 1000.0
    scale_bar_voltage_breadth = 50.0

    if time_len is not None:
        m.t = m.t * len(m.t) / time_len
        scale_bar_breadth = scale_bar_breadth * len(m.t) / time_len

    for j in range(len(prinzdb.neutypes)):
        if time_len is not None:
            axV.plot(m.t[10000+offset:10000+offset+time_len], Vx[j, 10000+offset:10000+offset+time_len] + 120.0 * (2 - j),
                     label=prinzdb.neutypes[j], lw=0.75, c='k', rasterized=True)
        else:
            axV.plot(m.t, Vx[j] + 120.0 * (2 - j), label=prinzdb.neutypes[j], lw=0.75, c='k')
        current_col += 1
    if scale_bar:
        if mode2 == 'small':
            axV.plot(10860 + np.arange(scale_bar_breadth), 318 * np.ones_like(np.arange(scale_bar_breadth)), lw=1.0,
                     color='k', zorder=5, rasterized=True)
            axV.text(10905, 324, '1 sec', fontsize=font_size)

            import matplotlib.patches as patches
            rect = patches.Rectangle((11890, 234), 2000, 100, linewidth=1, facecolor='w', zorder=3)
            axV.add_patch(rect)

            axV.plot(13490 * np.ones_like(np.arange(scale_bar_voltage_breadth)),
                     318 - scale_bar_voltage_breadth + np.arange(scale_bar_voltage_breadth), lw=1.0, color='k', zorder=6, rasterized=True)
            axV.text(11770, 270, '50 mV', fontsize=font_size)
        else:
            axV.plot(10860 + np.arange(scale_bar_breadth), 318 * np.ones_like(np.arange(scale_bar_breadth)), lw=1.0,
                     color='k', rasterized=True)
            axV.text(10905, 324, '1 sec', fontsize=font_size)

            import matplotlib.patches as patches
            rect = patches.Rectangle((10900, 264), 700, 50, linewidth=1, facecolor='w', zorder=3)
            axV.add_patch(rect)

            axV.plot(11860 * np.ones_like(np.arange(scale_bar_voltage_breadth)),
                     318 - scale_bar_voltage_breadth + np.arange(scale_bar_voltage_breadth), lw=1.0, color='k')
            axV.text(10930, 270, '50 mV', fontsize=font_size)



    if not legend and vis_legend:
        if mode2=='small':
            axV.text(-0.15, 0.75, 'AB/PD', fontsize=font_size, transform=axV.transAxes)
            axV.text(-0.1, 0.45, 'LP', fontsize=font_size, transform=axV.transAxes)
            axV.text(-0.1, 0.15, 'PY', fontsize=font_size, transform=axV.transAxes)
        else:
            axV.text(-1540+offset_labels, 220, 'AB/PD', fontsize=font_size)
            axV.text(-1050+offset_labels,  95, 'LP', fontsize=font_size)
            axV.text(-1080+offset_labels, -30, 'PY', fontsize=font_size)

    box = axV.get_position()

    if t_on is not None:
        axV.axvline(t_on, c='r', ls='--')

    if t_on is not None:
        axV.axvline(t_off, c='r', ls='--')

    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.axes.get_yaxis().set_ticks([])
    axV.axes.get_xaxis().set_ticks([])

    if legend: axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                          ncol=len(prinzdb.neutypes), fontsize=font_size*fontscale)
    axV.xaxis.set_tick_params(labelsize=font_size*fontscale)
    axV.yaxis.set_tick_params(labelsize=font_size*fontscale)
    axV.spines['left'].set_linewidth(2.0 * linescale)
    axV.spines['bottom'].set_linewidth(2.0 * linescale)

    col1 = 'r'
    col2 = 'r'
    col3 = 'r'

    if with_params:
        lticks = np.arange(len(params[:-7]))
        width = 0.35*linescale  # the width of the bars

        axmemparams.bar(lticks + width / 2, bar_vals * params[:-7] / 0.628e-3, width,
                        bottom=min(1e-8, np.min(params[:-7])), color='k')

        names = viz.get_labels(hyperparams, mathmode=True, include_q10=False)[:-7]
        if mem_dimensions is not None: names = names[mem_dimensions]
        axmemparams.set_ylim((0, 700))
        # axmemparams.set_ylabel('Membrane', fontsize=font_size)
        axmemparams.set_xticks(lticks + width / 2)
        if param_names:
            axmemparams.set_xticklabels(names, rotation='vertical', fontsize=font_size*fontscale)
        else:
            axmemparams.axes.get_xaxis().set_visible(False)
            axmemparams.axes.get_yaxis().set_visible(False)
        axmemparams.xaxis.set_tick_params(labelsize=font_size*fontscale)
        axmemparams.yaxis.set_tick_params(labelsize=font_size*fontscale)

        small_offset = [0.00, -0.0, -0.0, 0.0, -0.00, 0.0]
        font_decrease = 1.7
        mode = '13D'
        if mode == '13D':
            if mode2 == 'small':
                for i in range(6):  # 520 or so
                    axmemparams.text(-0.0 + i * 1.03, -360, 'x', fontsize=font_size / 2)
                axmemparams.set_ylim((0, 700))  # 850
                small_offset = [0.15, -0.04, -0.1, 0.0, -0.02, 0.0]
                for i in range(6):
                    if i == 2 or i == 3 or i == 4 or i == 5:  # -620
                        axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -410,
                                         r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                                include_multiplier=False, negative_num=False),
                                         fontsize=font_size / font_decrease)
                    else:
                        axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -410, r'$%s$' % str(int(bar_vals[i])),
                                         fontsize=font_size / font_decrease)
                axmemparams.text(0.11, -0.73, r'Membrane $\mathregular{\bar g}$', fontsize=font_size,
                                 transform=axmemparams.transAxes)
                axmemparams.text(0.22, -0.85, '[mS/cm' + chr(0x00b0 + 2) + ']', fontsize=font_size,
                                 transform=axmemparams.transAxes)
            else:

                for i in range(6):  # 520 or so
                    axmemparams.text(-0.0 + i * 1.03, -390, 'x', fontsize=font_size / 2)
                axmemparams.set_ylim((0, 600))  # 850
                small_offset = [0.15, -0.04, -0.1, 0.0, -0.02, 0.0]
                for i in range(6):
                    if i == 2 or i == 3 or i == 4 or i == 5:  # -620
                        axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -450,
                                         r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                                include_multiplier=False, negative_num=False),
                                         fontsize=font_size / font_decrease)
                    else:
                        axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -450, r'$%s$' % str(int(bar_vals[i])),
                                         fontsize=font_size / font_decrease)
                axmemparams.text(0.11, -0.95, r'Membrane $\mathregular{\bar g}$', fontsize=font_size,
                                 transform=axmemparams.transAxes)
                axmemparams.text(0.22, -1.12, '[mS/cm' + chr(0x00b0 + 2) + ']', fontsize=font_size,
                                 transform=axmemparams.transAxes)
        else:
            for i in range(6):  # 520 or so
                axmemparams.text(-0.0 + i * 1.03, -470, 'x', fontsize=font_size / 2)
            for i in range(6):
                if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5:  # -620
                    axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -530,
                                     r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                            include_multiplier=False, negative_num=False),
                                     fontsize=font_size / font_decrease)
                else:
                    axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -530, r'$%s$' % str(int(bar_vals[i])),
                                     fontsize=font_size / font_decrease)

        lticks = np.arange(len(params[-7:]))
        axsynparams.bar(lticks + width / 2, params[-7:] * 1e-3, width,
                        bottom=min(1e-8 * 1e-3, np.min(params[-7:] * 1e-3)), color='k')

        if mode2 == 'small':
            axsynparams.text(0.22, -0.73, r'Synaptic $\mathregular{\bar g}$', fontsize=font_size,
                             transform=axsynparams.transAxes)
            axsynparams.text(0.4, -0.85, '[nS]', fontsize=font_size, transform=axsynparams.transAxes)
        else:
            axsynparams.text(0.22, -0.95, r'Synaptic $\mathregular{\bar g}$', fontsize=font_size,
                             transform=axsynparams.transAxes)
            axsynparams.text(0.4, -1.12, '[nS]', fontsize=font_size, transform=axsynparams.transAxes)

        names = viz.get_labels(hyperparams, include_q10=False)[-7:]
        # axsynparams.set_ylabel('Synapses', fontsize=font_size)
        axsynparams.set_yscale('log')
        axsynparams.set_ylim((1e-8 * 1e-3, 1e-3 * 1e-3))
        axsynparams.set_xticks(lticks + width / 2)
        axsynparams.set_yticks([1e-11, 1e-9, 1e-7])
        if param_names:
            axsynparams.set_xticklabels(names, rotation='vertical', fontsize=font_size*fontscale)
        else:
            axsynparams.axes.get_xaxis().set_visible(False)
            axsynparams.axes.get_yaxis().set_visible(False)
        axsynparams.xaxis.set_tick_params(labelsize=font_size*fontscale)
        axsynparams.yaxis.set_tick_params(labelsize=font_size*fontscale)

        axsynparams.spines['left'].set_linewidth(2.0 * linescale)
        axsynparams.spines['bottom'].set_linewidth(2.0 * linescale)
        axmemparams.spines['left'].set_linewidth(2.0 * linescale)
        axmemparams.spines['bottom'].set_linewidth(2.0 * linescale)

    if with_ss:
        lticks = np.arange(len(stats))
        width = 0.35  # the width of the bars

        #stats[8:] *= 2000
        axss.bar(lticks + width / 2, stats, width, color='k')
        nan_pos = np.where(np.isnan(stats_nan))[0]
        axss.scatter(nan_pos + width / 2, 50 * np.ones_like(nan_pos), c='b', s=70.0, zorder=2, marker='x')

        # add some text for labels, title and axes ticks
        names = []
        for num in range(15):
            names.append(get_summ_stat_name(num))
        # axss.set_ylabel('Summary Statistics', fontsize=font_size)
        axss.set_xticks(lticks + width / 2)
        if ss_names:
            axss.set_xticklabels(names, rotation='vertical', fontsize=font_size*fontscale)
        else:
            axss.axes.get_xaxis().set_visible(False)
            axss.axes.get_yaxis().set_visible(False)
        axss.xaxis.set_tick_params(labelsize=font_size*fontscale)
        axss.yaxis.set_tick_params(labelsize=font_size*fontscale)
        axss.set_ylim([-4, 4])
        axss.set_yticks([-4, -2, 0, 2, 4])
        axss.set_yticklabels([r'$-4 \sigma$', r'$-2 \sigma$', '0', '$2 \sigma$', '$4 \sigma$'])

        axss.text(0.27, -0.95, 'Summary statistics', fontsize=font_size, transform=axss.transAxes)
        axss.text(0.145, -1.12, '[st. dev. of samples]', fontsize=font_size, transform=axss.transAxes)

        axss.spines['right'].set_visible(False)
        axss.spines['top'].set_visible(False)
        #axss.axes.get_yaxis().set_ticks([])

    axV.spines['right'].set_visible(False)
    axV.spines['top'].set_visible(False)
    axsynparams.spines['right'].set_visible(False)
    axsynparams.spines['top'].set_visible(False)
    axmemparams.spines['right'].set_visible(False)
    axmemparams.spines['top'].set_visible(False)

    sns.set(style="ticks", font_scale=1)
    sns.despine()

    axV.set_title('')
    if save_fig:
        plt.savefig(
            'png/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.png'.format(test_idx[0],
                                                                                                     counter),
            bbox_inches='tight', dpi=500)
        plt.savefig(
            'svg/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.svg'.format(test_idx[0],
                                                                                                     counter),
            bbox_inches='tight')

    return fig



def vis_sample_plain(m, s, sample, axV=None, t_on=None, t_off=None, col=['k', 'k', 'k'], print_label=False,
               voltage_trace=None, time_len=None, fontscale=1.0, linescale=1.0, offset=0, scale_bar=True,
               test_idx=None, case=None, title=None, date_today=None, counter=0, legend=True,
               save_fig=False):
    """
    Function of Kaan, modified by Michael. Used for plotting fig 5b Prinz.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    if axV is None:
        _, ax = plt.subplots(1, len(sample), figsize=(6*len(sample),6))

    font_size = 15.0
    current_counter = 0

    dt = m.t[1] - m.t[0]
    scale_bar_breadth = 500
    scale_bar_voltage_breadth = 50

    offscale = 100
    offvolt = -50

    if scale_bar: scale_col = 'k'
    else: scale_col = 'w'

    for current_sample in sample:
        if axV is None: axV = ax[current_counter]
        if voltage_trace is None:
            data = m.gen_single(current_sample)
        else:
            data = voltage_trace

        Vx = data['data']
        params = data['params']

        current_col = 0
        for j in range(len(prinzdb.neutypes)):
            if time_len is not None:
                axV.plot(m.t[10000+offset:10000+offset+time_len:5], Vx[j, 10000+offset:10000+offset+time_len:5] + 140.0 * (2 - j),
                         label=prinzdb.neutypes[j], lw=0.3, c=col)
            else:
                axV.plot(m.t, Vx[j] + 120.0 * (2 - j), label=prinzdb.neutypes[j], lw=0.3, c=col[current_col])
            current_col += 1

        if print_label:
            axV.plot([1100.0 + (offset - 26500) * (m.t[1] - m.t[0])], [300], color=col, marker='o',
                     markeredgecolor='w', ms=8,
                     markeredgewidth=1.0, path_effects=[pe.Stroke(linewidth=1.3, foreground='k'), pe.Normal()])

        if scale_bar:

            # time bar
            axV.plot((offset+5500)*dt+offscale + np.arange(scale_bar_breadth)[::scale_bar_breadth - 1],
                     (-40+offvolt) * np.ones_like(np.arange(scale_bar_breadth))[::scale_bar_breadth - 1],
                     lw=1.0, color='w')

            # voltage bar
            axV.plot(
                (2850 + offset*dt + offscale) * np.ones_like(np.arange(scale_bar_voltage_breadth))[::scale_bar_voltage_breadth - 1],
                275 + np.arange(scale_bar_voltage_breadth)[::scale_bar_voltage_breadth - 1],
                lw=1.0, color=scale_col, zorder=10)


        box = axV.get_position()

        if t_on is not None:
            axV.axvline(t_on, c='r', ls='--')

        if t_on is not None:
            axV.axvline(t_off, c='r', ls='--')

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.axes.get_yaxis().set_ticks([])
        axV.axes.get_xaxis().set_ticks([])

        #if legend: axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
        #                      ncol=len(prinzdb.neutypes), fontsize=font_size*fontscale)
        #axV.xaxis.set_tick_params(labelsize=font_size*fontscale)
        #axV.yaxis.set_tick_params(labelsize=font_size*fontscale)
        #axV.spines['left'].set_linewidth(2.0 * linescale)
        #axV.spines['bottom'].set_linewidth(2.0 * linescale)

        axV.spines['right'].set_visible(False)
        axV.spines['top'].set_visible(False)
        axV.spines['bottom'].set_visible(False)
        axV.spines['left'].set_visible(False)
        #sns.set(style="ticks", font_scale=1)
        #sns.despine()

        if save_fig:
            plt.savefig(
                '../../thesis_results/pdf/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.pdf'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')
            plt.savefig(
                '../../thesis_results/png/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.png'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')
            plt.savefig(
                '../../thesis_results/svg/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.svg'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')

        current_counter += 1



def vis_sample_plain_31DSynthetic(m, s, sample, axV=None, t_on=None, t_off=None, col=['k', 'k', 'k'], print_label=False,
               voltage_trace=None, time_len=None, fontscale=1.0, linescale=1.0, offset=0, scale_bar=True,
               test_idx=None, case=None, title=None, date_today=None, counter=0, legend=True, draw_patch=False,
               save_fig=False):
    """
    Function of Kaan, modified by Michael. Used for plotting fig 5b Prinz.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    if axV is None:
        _, ax = plt.subplots(1, len(sample), figsize=(6*len(sample),6))

    font_size = 15.0
    current_counter = 0

    dt = m.t[1] - m.t[0]
    scale_bar_breadth = 500
    scale_bar_voltage_breadth = 50

    offscale = 100
    offvolt = -50

    if scale_bar: scale_col = 'k'
    else: scale_col = 'w'

    for current_sample in sample:
        if axV is None: axV = ax[current_counter]
        if voltage_trace is None:
            data = m.gen_single(current_sample)
        else:
            data = voltage_trace

        Vx = data['data']
        params = data['params']

        current_col = 0
        for j in range(len(prinzdb.neutypes)):
            if time_len is not None:
                axV.plot(m.t[10000+offset:10000+offset+time_len:5], Vx[j, 10000+offset:10000+offset+time_len:5] + 140.0 * (2 - j),
                         label=prinzdb.neutypes[j], lw=0.3, c=col)
            else:
                axV.plot(m.t, Vx[j] + 120.0 * (2 - j), label=prinzdb.neutypes[j], lw=0.3, c=col[current_col])
            current_col += 1

        if print_label:
            axV.plot([1100.0 + (offset - 26500) * (m.t[1] - m.t[0])], [300], color=col, marker='o',
                     markeredgecolor='w', ms=8,
                     markeredgewidth=1.0, path_effects=[pe.Stroke(linewidth=1.3, foreground='k'), pe.Normal()])

        if draw_patch:
            import matplotlib.patches as patches
            rect = patches.Rectangle((1650 + offscale, 266), 200, 65, linewidth=1, facecolor='w', zorder=3)
            axV.add_patch(rect)

        if scale_bar:

            # time bar
            axV.plot((offset+5500)*dt+offscale + np.arange(scale_bar_breadth)[::scale_bar_breadth - 1],
                     (-40+offvolt) * np.ones_like(np.arange(scale_bar_breadth))[::scale_bar_breadth - 1],
                     lw=1.0, color='w')

            # voltage bar
            axV.plot(
                (2850 + offset*dt + offscale) * np.ones_like(np.arange(scale_bar_voltage_breadth))[::scale_bar_voltage_breadth - 1],
                275 + np.arange(scale_bar_voltage_breadth)[::scale_bar_voltage_breadth - 1],
                lw=1.0, color=scale_col, zorder=10)


        box = axV.get_position()

        if t_on is not None:
            axV.axvline(t_on, c='r', ls='--')

        if t_on is not None:
            axV.axvline(t_off, c='r', ls='--')

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.axes.get_yaxis().set_ticks([])
        axV.axes.get_xaxis().set_ticks([])

        #if legend: axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
        #                      ncol=len(prinzdb.neutypes), fontsize=font_size*fontscale)
        #axV.xaxis.set_tick_params(labelsize=font_size*fontscale)
        #axV.yaxis.set_tick_params(labelsize=font_size*fontscale)
        #axV.spines['left'].set_linewidth(2.0 * linescale)
        #axV.spines['bottom'].set_linewidth(2.0 * linescale)

        axV.spines['right'].set_visible(False)
        axV.spines['top'].set_visible(False)
        axV.spines['bottom'].set_visible(False)
        axV.spines['left'].set_visible(False)
        #sns.set(style="ticks", font_scale=1)
        #sns.despine()

        if save_fig:
            plt.savefig(
                '../../thesis_results/pdf/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.pdf'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')
            plt.savefig(
                '../../thesis_results/png/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.png'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')
            plt.savefig(
                '../../thesis_results/svg/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.svg'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')

        current_counter += 1



def vis_sample_plain_bit_more(m, s, sample, axV=None, t_on=None, t_off=None, col=['k', 'k', 'k'],
               voltage_trace=None, time_len=None, fontscale=1.0, linescale=1.0, offset=0, scale_bar=False,
               test_idx=None, case=None, title=None, date_today=None, counter=0, legend=True,
               save_fig=False):
    """
    Function of Kaan, modified by Michael. Used for plotting fig 5b Prinz.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    if axV is None:
        _, ax = plt.subplots(1, len(sample), figsize=(6*len(sample),6))

    font_size = 8.0
    current_counter = 0

    dt = m.t[1] - m.t[0]

    for current_sample in sample:
        if axV is None: axV = ax[current_counter]
        if voltage_trace is None:
            data = m.gen_single(current_sample)
        else:
            data = voltage_trace

        Vx = data['data']
        params = data['params']

        current_col = 0
        for j in range(len(prinzdb.neutypes)):
            if time_len is not None:
                axV.plot(m.t[10000+offset:10000+offset+time_len], Vx[j, 10000+offset:10000+offset+time_len] + 140.0 * (2 - j),
                         label=prinzdb.neutypes[j], lw=0.3, c=col[current_col])
            else:
                axV.plot(m.t, Vx[j] + 120.0 * (2 - j), label=prinzdb.neutypes[j], lw=0.1, c=col[current_col], rasterized=True)
            current_col += 1

        label_col = 'w'

        axV.text(-0.035,  0.75, 'AB/PD', fontsize=font_size, c=label_col, transform=axV.transAxes)
        axV.text(-0.028, 0.45, 'LP', fontsize=font_size, c=label_col, transform=axV.transAxes)
        axV.text(-0.03,  0.15, 'PY', fontsize=font_size, c=label_col, transform=axV.transAxes)

        axV.plot([1000.0+(offset-26500)*(m.t[1]-m.t[0])], [314], color=col[0], marker='o', markeredgecolor='w', ms=8,
                          markeredgewidth=1.0, path_effects=[pe.Stroke(linewidth=1.3, foreground='k'), pe.Normal()])

        scale_bar_breadth = 500
        scale_bar_voltage_breadth = 50

        offscale=100
        offvolt =-50

        if scale_bar:

            # time bar
            axV.plot((offset+5500)*dt+offscale + np.arange(scale_bar_breadth)[::scale_bar_breadth - 1],
                     (-40+offvolt) * np.ones_like(np.arange(scale_bar_breadth))[::scale_bar_breadth - 1], lw=1.0, color='k')
            axV.text((offset+5500)*dt+offscale, -125, '500 ms', c=label_col, fontsize=font_size)

            import matplotlib.patches as patches
            rect = patches.Rectangle((4400+offscale, 296+offvolt), 500, 50, linewidth=1, facecolor='w', zorder=3)
            axV.add_patch(rect)

            # voltage bar
            axV.plot((4810+offscale) * np.ones_like(np.arange(scale_bar_voltage_breadth))[::scale_bar_voltage_breadth - 1],
                     -70 + np.arange(scale_bar_voltage_breadth)[
                                                       ::scale_bar_voltage_breadth - 1]
                     , lw=1.0, color='w', zorder=10)

            axV.plot(
                (4710 + offscale) * np.ones_like(np.arange(scale_bar_voltage_breadth))[::scale_bar_voltage_breadth - 1],
                275 + np.arange(scale_bar_voltage_breadth)[
                      ::scale_bar_voltage_breadth - 1]
                , lw=1.0, color='k', zorder=10)
            axV.text(5000, -70, '50 mV', c=label_col, fontsize=8.0)

        box = axV.get_position()

        if t_on is not None:
            axV.axvline(t_on, c='r', ls='--')

        if t_on is not None:
            axV.axvline(t_off, c='r', ls='--')

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.axes.get_yaxis().set_ticks([])
        axV.axes.get_xaxis().set_ticks([])

        if legend: axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                              ncol=len(prinzdb.neutypes), fontsize=font_size*fontscale)
        axV.xaxis.set_tick_params(labelsize=font_size*fontscale)
        axV.yaxis.set_tick_params(labelsize=font_size*fontscale)
        axV.spines['left'].set_linewidth(2.0 * linescale)
        axV.spines['bottom'].set_linewidth(2.0 * linescale)

        axV.spines['right'].set_visible(False)
        axV.spines['top'].set_visible(False)
        axV.spines['bottom'].set_visible(False)
        axV.spines['left'].set_visible(False)

        axV.set_title('')
        if save_fig:
            plt.savefig(
                '../../thesis_results/pdf/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.pdf'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')
            plt.savefig(
                '../../thesis_results/png/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.png'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')
            plt.savefig(
                '../../thesis_results/svg/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.svg'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')

        current_counter += 1



def vis_sample_plain_bit_more_31DSynthetic(m, s, sample, axV=None, t_on=None, t_off=None, col=['k', 'k', 'k'],
               voltage_trace=None, time_len=None, fontscale=1.0, linescale=1.0, offset=0, scale_bar=False,
               test_idx=None, case=None, title=None, date_today=None, counter=0, legend=True,
               save_fig=False):
    """
    Function of Kaan, modified by Michael. Used for plotting fig 5b Prinz.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    if axV is None:
        _, ax = plt.subplots(1, len(sample), figsize=(6*len(sample),6))

    font_size = 8.0
    current_counter = 0

    dt = m.t[1] - m.t[0]

    for current_sample in sample:
        if axV is None: axV = ax[current_counter]
        if voltage_trace is None:
            data = m.gen_single(current_sample)
        else:
            data = voltage_trace

        Vx = data['data']
        params = data['params']

        current_col = 0
        for j in range(len(prinzdb.neutypes)):
            if time_len is not None:
                axV.plot(m.t[10000+offset:10000+offset+time_len], Vx[j, 10000+offset:10000+offset+time_len] + 140.0 * (2 - j),
                         label=prinzdb.neutypes[j], lw=0.3, c=col[current_col])
            else:
                axV.plot(m.t, Vx[j] + 120.0 * (2 - j), label=prinzdb.neutypes[j], lw=0.1, c=col[current_col], rasterized=True)
            current_col += 1

        label_col = 'w'

        #axV.text(-0.035,  0.75, 'AB/PD', fontsize=font_size, c=label_col, transform=axV.transAxes)
        #axV.text(-0.028, 0.45, 'LP', fontsize=font_size, c=label_col, transform=axV.transAxes)
        #axV.text(-0.03,  0.15, 'PY', fontsize=font_size, c=label_col, transform=axV.transAxes)

        #axV.plot([1000.0+(offset-26500)*(m.t[1]-m.t[0])], [314], color=col[0], marker='o', markeredgecolor='w', ms=8,
        #                  markeredgewidth=1.0, path_effects=[pe.Stroke(linewidth=1.3, foreground='k'), pe.Normal()])

        scale_bar_breadth = 500
        scale_bar_voltage_breadth = 50

        offscale=100
        offvolt =-50

        if scale_bar:
            draw_col = 'k'
        else:
            draw_col = 'w'


        # time bar
        axV.plot((offset+5500)*dt+offscale + np.arange(scale_bar_breadth)[::scale_bar_breadth - 1],
                 (-40+offvolt) * np.ones_like(np.arange(scale_bar_breadth))[::scale_bar_breadth - 1], lw=1.0, color=draw_col)

        # voltage bar
        axV.plot((offset*dt)+(3100+offscale) * np.ones_like(np.arange(scale_bar_voltage_breadth))[::scale_bar_voltage_breadth - 1],
                 -70 + np.arange(scale_bar_voltage_breadth)[
                                                   ::scale_bar_voltage_breadth - 1]
                 , lw=1.0, color='w', zorder=10)

        axV.plot((offset*dt)+(3100 + offscale) * np.ones_like(np.arange(scale_bar_voltage_breadth))[::scale_bar_voltage_breadth - 1],
            275 + np.arange(scale_bar_voltage_breadth)[
                  ::scale_bar_voltage_breadth - 1]
            , lw=1.0, color=draw_col, zorder=10)

        box = axV.get_position()

        if t_on is not None:
            axV.axvline(t_on, c='r', ls='--')

        if t_on is not None:
            axV.axvline(t_off, c='r', ls='--')

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.axes.get_yaxis().set_ticks([])
        axV.axes.get_xaxis().set_ticks([])

        if legend: axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                              ncol=len(prinzdb.neutypes), fontsize=font_size*fontscale)
        axV.xaxis.set_tick_params(labelsize=font_size*fontscale)
        axV.yaxis.set_tick_params(labelsize=font_size*fontscale)
        axV.spines['left'].set_linewidth(2.0 * linescale)
        axV.spines['bottom'].set_linewidth(2.0 * linescale)

        axV.spines['right'].set_visible(False)
        axV.spines['top'].set_visible(False)
        axV.spines['bottom'].set_visible(False)
        axV.spines['left'].set_visible(False)

        axV.set_title('')
        if save_fig:
            plt.savefig(
                '../../thesis_results/pdf/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.pdf'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')
            plt.savefig(
                '../../thesis_results/png/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.png'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')
            plt.savefig(
                '../../thesis_results/svg/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.svg'.format(test_idx[0],
                                                                                                         counter),
                bbox_inches='tight')

        current_counter += 1





def vis_sample_subfig(m, s, sample, hyperparams, stats=None, t_on=None, t_off=None, with_ss=True, with_params=True, voltage_trace=None,
                      test_idx=None, case=None, title=None, date_today=None, counter=0, save_fig=False, legend_offset=0.0,
                      axV=None, axss=None, axmemparams=None, axsynparams=None, max_stats=None, min_stats=None,
                      mem_dimensions=None, mode='13D', mode_for_membrane_height=None,
                      stat_mean=None, stat_std=None, scale_bar=True, stat_scale=None, current_col='g',
                      max_conds=None, min_conds=None, legend=True, ss_names=True, param_names=True):
    """
    Based on vis_sample. Is called when the pdf should be shown next ot the sample.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    # Hyperparameters for plotting
    font_size=15.0 # fontsize of the labels
    col_bar = 'k' # color of the bars for summstats and conductances
    col_minmax = 'k' # color of the horizontal line indicating the max and min value of summstats and conds
    col_shade = 'k' # color of the shade between the max and min values
    values_each = 100 # not so important. How many values we evaluate for the max and min values
    indicator_fraction = 0.8 # breath of the horizontal bars for max and min, should be within [0,1]
    opacity = 0.5 # opacity of the shade
    width = 0.35  # the width of the bars
    neuron_labels = ['AB/PD', 'LP', 'PY'] # labels for the legends
    scale_bar_breadth = 1000
    scale_bar_voltage_breadth = 50

    if voltage_trace is None: data = m.gen_single(sample)
    else: data = voltage_trace

    Vx = data['data']
    params = data['params']

    #stats = s.calc([data])[0]
    stats_nan = deepcopy(stats)
    #stats[np.isnan(stats)]=0.0
    #stats = scale_to_experimental(stats)

    bar_scaling_factors = [1.0, 10, 100, 10, 100, 1, 10000, 10000]
    bar_scaling_factors = np.asarray([[1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000],
                                      [1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000],
                                      [1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000]])
    bar_vals = bar_scaling_factors[np.asarray(hyperparams.use_membrane)]

    if mem_dimensions is not None:
        params_trunc = params[mem_dimensions].tolist()
        params_trunc += params[-7:].tolist()
        bar_vals = bar_vals[mem_dimensions]
        params = np.asarray(params_trunc)

    step_Vtrace = 10
    if legend:
        for j in range(len(prinzdb.neutypes)):
            axV.plot(m.t[25500:25500+200000:step_Vtrace], Vx[j,25500:25500+200000:step_Vtrace]+140.0*(2-j), label=neuron_labels[j])
    else:
        for j in range(len(prinzdb.neutypes)):
            axV.plot(m.t[25500:25500+200000:step_Vtrace], Vx[j,25500:25500+200000:step_Vtrace]+140.0*(2-j), label=neuron_labels[j], c='k', lw=0.6)
    if scale_bar:
        axV.plot(4810+np.arange(scale_bar_breadth)[::scale_bar_breadth-1], 358*np.ones_like(np.arange(scale_bar_breadth))[::scale_bar_breadth-1], lw=1.0, color='k')
        axV.text(4845, 364, '1 sec', fontsize=font_size)

        import matplotlib.patches as patches
        rect = patches.Rectangle((5400, 296), 500, 50, linewidth=1,  facecolor='w', zorder=3)
        axV.add_patch(rect)

        axV.plot(5810*np.ones_like(np.arange(scale_bar_voltage_breadth))[::scale_bar_voltage_breadth-1],
                 358-scale_bar_voltage_breadth+np.arange(scale_bar_voltage_breadth)[::scale_bar_voltage_breadth-1]
                 , lw=1.0, color='k',zorder=10)
        axV.text(5430, 310, '50 mV', fontsize=font_size)
    axV.plot(m.t[22500]+20, 325, color=current_col, marker='o', markeredgecolor='w', ms=22,
             markeredgewidth=0.5)

    if not legend:
        axV.text(-0.08,  0.75, 'AB/PD', fontsize=font_size, transform=axV.transAxes)
        axV.text(-0.04, 0.45, 'LP', fontsize=font_size, transform=axV.transAxes)
        axV.text(-0.045,  0.15, 'PY', fontsize=font_size, transform=axV.transAxes)

    box = axV.get_position()

    if t_on is not None:
        axV.axvline(t_on, c='r', ls='--')

    if t_on is not None:
        axV.axvline(t_off, c='r', ls='--')

    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.axes.get_yaxis().set_ticks([])
    axV.axes.get_xaxis().set_ticks([])

    if legend:
        if scale_bar:
            axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=len(prinzdb.neutypes), fontsize=font_size)
        else:
            axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                       ncol=len(prinzdb.neutypes), fontsize=font_size)
    axV.xaxis.set_tick_params(labelsize=font_size)
    axV.yaxis.set_tick_params(labelsize=font_size)

    #axV.set_xlim((m.t[25500] - 400, 11500))
    axV.set_xlim((m.t[25500] - 200, m.t[25500+200000]+200))
    if scale_bar:
        axV.set_ylim((-95, 360))

    if with_params:
        lticks = np.arange(len(params[:-7]))

        end_of_time_axis = len(params[:-7]) - 1 + width

        full_time = np.linspace(width/2-0.5, end_of_time_axis+0.5-width/2, values_each * len(params[:-7]))
        full_min_conds = np.tile(bar_vals * min_conds[:-7] / 0.628e-3, (values_each, 1))
        full_min_conds = full_min_conds.flatten(order='F')
        full_max_conds = np.tile(bar_vals * max_conds[:-7] / 0.628e-3, (values_each, 1))
        full_max_conds = full_max_conds.flatten(order='F')

        axmemparams.bar(lticks + width / 2, bar_vals * params[:-7] / 0.628e-3, width,
                        bottom=min(1e-8, np.min(params[:-7])), color=col_bar)
        #min_conds_scaled = bar_vals * deepcopy(min_conds[:-7]) / 0.628e-3
        #max_conds_scaled = bar_vals * deepcopy(max_conds[:-7]) / 0.628e-3
        #axmemparams.plot(width / 2+np.arange(len(min_conds_scaled)), min_conds_scaled, col1)
        #axmemparams.plot(width / 2+np.arange(len(min_conds_scaled)), max_conds_scaled, col2)
        #axmemparams.fill_between(width / 2+np.arange(len(max_conds_scaled)), min_conds_scaled, max_conds_scaled,
        #                         facecolor=col3, alpha=0.5)
        for k in range(len(params[:-7])):
            start_t = int(values_each*k+(1-indicator_fraction)/2*values_each)
            end_t   = int(values_each*(k+1)-(1-indicator_fraction)/2*values_each)
            time_diff = end_t - start_t
            axmemparams.plot(full_time[start_t:end_t][::time_diff-1], full_min_conds[start_t:end_t][::time_diff-1], col_minmax)
            axmemparams.plot(full_time[start_t:end_t][::time_diff-1], full_max_conds[start_t:end_t][::time_diff-1], col_minmax)
            axmemparams.fill_between(full_time[start_t:end_t][::time_diff-1], full_min_conds[start_t:end_t][::time_diff-1],
                                     full_max_conds[start_t:end_t][::time_diff-1], facecolor=col_shade, alpha=opacity)

        names = viz.get_labels(hyperparams, mathmode=True, include_q10=False)[:-7]
        if mem_dimensions is not None:
            names = names[mem_dimensions]
        axmemparams.set_ylim((0, 700)) # 850
        axmemparams.set_xticks(lticks + width / 2)
        if param_names: axmemparams.set_xticklabels(names, rotation='vertical', fontsize=font_size)
        else:
            axmemparams.axes.get_xaxis().set_visible(False)
            axmemparams.axes.get_yaxis().set_visible(False)
        axmemparams.xaxis.set_tick_params(labelsize=font_size)
        axmemparams.yaxis.set_tick_params(labelsize=font_size)

        small_offset = [0.00, -0.0, -0.0, 0.0, -0.00, 0.0]
        font_decrease = 1.7
        if mode == '13D':
            for i in range(6):  # 520 or so
                axmemparams.text(-0.0 + i * 1.03, -390, 'x', fontsize=font_size / 2)
            axmemparams.set_ylim((0, 600))  # 850
            small_offset = [0.15, -0.04, -0.1, 0.0, -0.02, 0.0]
            for i in range(6):
                if i == 2 or i == 3 or i == 4 or i == 5:  # -620
                    axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -450,
                                     r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                            include_multiplier=False, negative_num=False),
                                     fontsize=font_size / font_decrease)
                else:
                    axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -450, r'$%s$' % str(int(bar_vals[i])),
                                     fontsize=font_size / font_decrease)
            axmemparams.text(0.11, -0.95, r'Membrane $\mathregular{\bar g}$', fontsize=font_size, transform=axmemparams.transAxes)
            axmemparams.text(0.22, -1.12, '[mS/cm'+chr(0x00b0 + 2)+']', fontsize=font_size, transform=axmemparams.transAxes)
        else:
            if mode_for_membrane_height == 'high':
                axmemparams.set_ylim((0, 1000))
                for i in range(6):  # 520 or so
                    axmemparams.text(-0.0 + i * 1.03, -650, 'x', fontsize=font_size / 2)
                for i in range(6):
                    if bar_vals[i] == 1:
                        axmemparams.text(-0.2 + i * 1.02 + 0.18, -750,
                                         r'$1$', fontsize=font_size / font_decrease)
                    elif bar_vals[i] == 10:
                        axmemparams.text(-0.2 + i * 1.0 + 0.05, -750,
                                         r'$10$', fontsize=font_size / font_decrease)
                    else:
                        axmemparams.text(-0.2 + i * 1.0, -750,
                                         r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                                include_multiplier=False, negative_num=False),
                                         fontsize=font_size / font_decrease)
            else:
                for i in range(6):  # 520 or so
                    axmemparams.text(-0.0 + i * 1.03, -470, 'x', fontsize=font_size / 2)
                for i in range(6):
                    if i==0 or i==1 or i == 2 or i == 3 or i == 4 or i == 5:         #-620
                        axmemparams.text(-0.2+i*1.0+small_offset[i], -530, r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),include_multiplier=False, negative_num=False), fontsize=font_size / font_decrease)
                    else:
                        axmemparams.text(-0.2+i*1.0+small_offset[i], -530, r'$%s$' % str(int(bar_vals[i])), fontsize=font_size/font_decrease)
            axmemparams.text(0.11, -0.95, r'Membrane $\mathregular{\bar g}$', fontsize=font_size, transform=axmemparams.transAxes)
            axmemparams.text(0.22, -1.12, '[mS/cm' + chr(0x00b0 + 2) + ']', fontsize=font_size, transform=axmemparams.transAxes)

        lticks = np.arange(len(params[-7:]))
        end_of_time_axis = len(params[-7:])-1+width
        full_time = np.linspace(width/2-0.5, end_of_time_axis+0.5-width/2, values_each * len(params[-7:]))
        full_min_conds = np.tile(min_conds[-7:] * 1e-3, (values_each,1))
        full_min_conds = full_min_conds.flatten(order='F')
        full_max_conds = np.tile(max_conds[-7:] * 1e-3, (values_each, 1))
        full_max_conds = full_max_conds.flatten(order='F')
        axsynparams.bar(lticks + width / 2, params[-7:]*1e-3, width, color=col_bar)
        #axsynparams.plot(width / 2+np.arange(len(min_conds[-7:])), min_conds[-7:]*0.628e-3, col1)
        #axsynparams.plot(width / 2+np.arange(len(min_conds[-7:])), max_conds[-7:]*0.628e-3, col2)
        #axsynparams.fill_between(width / 2 + np.arange(len(min_conds[-7:])), min_conds[-7:] * 0.628e-3,
        #                         max_conds[-7:] * 0.628e-3, facecolor=col3, alpha=0.5)
        for k in range(len(params[-7:])):
            start_t = int(values_each * k + (1 - indicator_fraction) / 2 * values_each)
            end_t = int(values_each * (k + 1) - (1 - indicator_fraction) / 2 * values_each)
            time_diff = end_t - start_t
            axsynparams.plot(full_time[start_t:end_t][::time_diff-1], full_min_conds[start_t:end_t][::time_diff-1], col_minmax)
            axsynparams.plot(full_time[start_t:end_t][::time_diff-1], full_max_conds[start_t:end_t][::time_diff-1], col_minmax)
            axsynparams.fill_between(full_time[start_t:end_t][::time_diff-1], full_min_conds[start_t:end_t][::time_diff-1],
                                     full_max_conds[start_t:end_t][::time_diff-1], facecolor=col_shade, alpha=opacity)

        axsynparams.text(0.22, -0.95, r'Synaptic $\mathregular{\bar g}$', fontsize=font_size, transform=axsynparams.transAxes)
        axsynparams.text(0.4, -1.12, '[nS]', fontsize=font_size, transform=axsynparams.transAxes)

        names = viz.get_labels(hyperparams, include_q10=False)[-7:]
        #axsynparams.set_ylabel('Synapses', fontsize=font_size)
        axsynparams.set_yscale('log')
        axsynparams.set_ylim((1e-8*1e-3, 1.3*1e-3*1e-3))
        axsynparams.set_xticks(lticks + width / 2)
        axsynparams.set_yticks([1e-11, 1e-9, 1e-7])
        #locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        #axsynparams.yaxis.set_minor_locator(locmin)
        #axsynparams.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if param_names: axsynparams.set_xticklabels(names, rotation='vertical', fontsize=font_size)
        else:
            axsynparams.axes.get_xaxis().set_visible(False)
            axsynparams.axes.get_yaxis().set_visible(False)
        axsynparams.xaxis.set_tick_params(labelsize=font_size)
        axsynparams.yaxis.set_tick_params(labelsize=font_size)

        axsynparams.spines['right'].set_visible(False)
        axsynparams.spines['top'].set_visible(False)
        axmemparams.spines['right'].set_visible(False)
        axmemparams.spines['top'].set_visible(False)

    if with_ss:
        lticks = np.arange(len(stats))

        if stat_scale is None:
            stats[8:] *= 2000
        min_stats_scaled = deepcopy(min_stats)
        max_stats_scaled = deepcopy(max_stats)
        if stat_scale is None:
            min_stats_scaled[8:] = min_stats_scaled[8:] * 2000
            max_stats_scaled[8:] = max_stats_scaled[8:] * 2000
        axss.bar(lticks + width / 2, stats, width, color=col_bar)
        #axss.plot(width / 2+np.arange(len(min_stats_scaled)), min_stats_scaled, col1)
        #axss.plot(width / 2+np.arange(len(min_stats_scaled)), max_stats_scaled, col2)
        #axss.fill_between(width / 2+np.arange(len(min_stats_scaled)), min_stats_scaled, max_stats_scaled,
        #                  facecolor=col3, alpha=0.5)

        end_of_time_axis = len(stats) - 1 + width
        full_time = np.linspace(width / 2 - 0.5, end_of_time_axis + 0.5 - width / 2, values_each * len(stats))
        full_min_ss = np.tile(min_stats_scaled, (values_each, 1))
        full_min_ss = full_min_ss.flatten(order='F')
        full_max_ss = np.tile(max_stats_scaled, (values_each, 1))
        full_max_ss = full_max_ss.flatten(order='F')
        for k in range(len(stats)):
            start_t = int(values_each * k + (1 - indicator_fraction) / 2 * values_each)
            end_t = int(values_each * (k + 1) - (1 - indicator_fraction) / 2 * values_each)
            time_diff = end_t - start_t
            axss.plot(full_time[start_t:end_t][::time_diff-1], full_min_ss[start_t:end_t][::time_diff-1], col_minmax)
            axss.plot(full_time[start_t:end_t][::time_diff-1], full_max_ss[start_t:end_t][::time_diff-1], col_minmax)
            axss.fill_between(full_time[start_t:end_t][::time_diff-1], full_min_ss[start_t:end_t][::time_diff-1],
                                     full_max_ss[start_t:end_t][::time_diff-1], facecolor=col_shade, alpha=opacity)

        axss.text(0.27,  -0.95, 'Summary statistics', fontsize=font_size, transform=axss.transAxes)
        axss.text(0.145, -1.12, '[st. dev. of experimental data]', fontsize=font_size, transform=axss.transAxes)

        nan_pos = np.where(np.isnan(stats_nan))[0]
        if stat_scale is not None:
            axss.scatter(nan_pos+width/2, 3.5*np.ones_like(nan_pos),
                         c='k', s=70.0, zorder=2, marker='x')
        else:
            axss.scatter(nan_pos + width / 2, 1900 * np.ones_like(nan_pos),
                         c='k', s=70.0, zorder=2, marker='x')

        # add some text for labels, title and axes ticks
        names = []
        for num in range(15):
            names.append(get_summ_stat_name(num))
        #axss.set_ylabel('Summary Statistics', fontsize=font_size)
        axss.set_yticks([-4, -2, 0, 2, 4])
        axss.set_yticklabels([r'$-4 \sigma$', '$-2 \sigma$', '0', '$2 \sigma$', '$4 \sigma$'])
        #axss.axes.get_yaxis().set_ticks([])
        axss.set_xticks(lticks + width / 2)
        if ss_names: axss.set_xticklabels(names, rotation='vertical', fontsize=font_size)
        else:
            axss.axes.get_xaxis().set_visible(False)
            axss.axes.get_yaxis().set_visible(False)
        axss.xaxis.set_tick_params(labelsize=font_size)
        axss.yaxis.set_tick_params(labelsize=font_size)
        if stat_scale is not None:
            axss.set_ylim([-4.0, 4.0])
        else:
            axss.set_ylim([-450, 2100])

        axss.spines['right'].set_visible(False)
        axss.spines['top'].set_visible(False)
    axV.spines['right'].set_visible(False)
    axV.spines['top'].set_visible(False)

    sns.set(style="ticks", font_scale=1)
    sns.despine()

    axV.set_title('')
    if save_fig:
        plt.savefig('../../thesis_results/pdf/'+date_today+'_sample_prinz_'+case+'_{}_{}.pdf'.format(test_idx[0], counter),
                    bbox_inches='tight')
        plt.savefig('../../thesis_results/png/'+date_today+'_sample_prinz_'+case+'_{}_{}.png'.format(test_idx[0], counter),
                    bbox_inches='tight')
        plt.savefig('../../thesis_results/svg/'+date_today+'_sample_prinz_'+case+'_{}_{}.svg'.format(test_idx[0], counter),
                    bbox_inches='tight')


    return axV, axss, axmemparams, axsynparams




def vis_sample_subfig_twitter(m, s, sample, hyperparams, stats=None, t_on=None, t_off=None, with_ss=False, with_params=False, voltage_trace=None,
                      test_idx=None, case=None, title=None, date_today=None, counter=0, save_fig=False, legend_offset=0.0,
                      axV=None, axss=None, axmemparams=None, axsynparams=None, max_stats=None, min_stats=None,
                      mem_dimensions=None, mode='13D', mode_for_membrane_height=None, offset=0,
                      stat_mean=None, stat_std=None, scale_bar=True, stat_scale=None, current_col='g',
                      max_conds=None, min_conds=None, legend=True, ss_names=True, param_names=True):
    """
    Based on vis_sample. Is called when the pdf should be shown next ot the sample.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    # Hyperparameters for plotting
    font_size=15.0 # fontsize of the labels
    col_bar = 'k' # color of the bars for summstats and conductances
    col_minmax = 'k' # color of the horizontal line indicating the max and min value of summstats and conds
    col_shade = 'k' # color of the shade between the max and min values
    values_each = 100 # not so important. How many values we evaluate for the max and min values
    indicator_fraction = 0.8 # breath of the horizontal bars for max and min, should be within [0,1]
    opacity = 0.5 # opacity of the shade
    width = 0.35  # the width of the bars
    neuron_labels = ['AB/PD', 'LP', 'PY'] # labels for the legends
    scale_bar_breadth = 1000
    scale_bar_voltage_breadth = 50

    if voltage_trace is None: data = m.gen_single(sample)
    else: data = voltage_trace

    Vx = data['data']
    params = data['params']

    #stats = s.calc([data])[0]
    stats_nan = deepcopy(stats)
    #stats[np.isnan(stats)]=0.0
    #stats = scale_to_experimental(stats)

    bar_scaling_factors = [1.0, 10, 100, 10, 100, 1, 10000, 10000]
    bar_scaling_factors = np.asarray([[1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000],
                                      [1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000],
                                      [1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000]])
    bar_vals = bar_scaling_factors[np.asarray(hyperparams.use_membrane)]

    if mem_dimensions is not None:
        params_trunc = params[mem_dimensions].tolist()
        params_trunc += params[-7:].tolist()
        bar_vals = bar_vals[mem_dimensions]
        params = np.asarray(params_trunc)

    step_Vtrace = 5
    for j in range(len(prinzdb.neutypes)):
        axV.plot(m.t[25500+offset:25500+115000+offset:step_Vtrace], Vx[j,25500+offset:25500+115000+offset:step_Vtrace]+140.0*(2-j), label=neuron_labels[j], c='k', lw=0.6)

    box = axV.get_position()

    if t_on is not None:
        axV.axvline(t_on, c='r', ls='--')

    if t_on is not None:
        axV.axvline(t_off, c='r', ls='--')

    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.axes.get_yaxis().set_ticks([])
    axV.axes.get_xaxis().set_ticks([])

    if legend:
        if scale_bar:
            axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=len(prinzdb.neutypes), fontsize=font_size)
        else:
            axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                       ncol=len(prinzdb.neutypes), fontsize=font_size)
    axV.xaxis.set_tick_params(labelsize=font_size)
    axV.yaxis.set_tick_params(labelsize=font_size)


    axV.spines['right'].set_visible(False)
    axV.spines['top'].set_visible(False)
    axV.spines['bottom'].set_visible(False)
    axV.spines['left'].set_visible(False)

    sns.set(style="ticks", font_scale=1)
    sns.despine()

    axV.set_title('')
    if save_fig:
        plt.savefig('../../thesis_results/pdf/'+date_today+'_sample_prinz_'+case+'_{}_{}.pdf'.format(test_idx[0], counter),
                    bbox_inches='tight')
        plt.savefig('../../thesis_results/png/'+date_today+'_sample_prinz_'+case+'_{}_{}.png'.format(test_idx[0], counter),
                    bbox_inches='tight')
        plt.savefig('../../thesis_results/svg/'+date_today+'_sample_prinz_'+case+'_{}_{}.svg'.format(test_idx[0], counter),
                    bbox_inches='tight')


    return axV




def vis_sample_subfig_no_voltage(m, s, sample, hyperparams, stats=None, t_on=None, t_off=None, with_ss=True, with_params=True, voltage_trace=None,
                      test_idx=None, case=None, title=None, date_today=None, counter=0, save_fig=False, legend_offset=0.0,
                      axss=None, axmemparams=None, axsynparams=None, max_stats=None, min_stats=None,
                      mem_dimensions=None, mode='13D', mode_for_membrane_height=None, labels_=True, color_input='k',
                      stat_mean=None, stat_std=None, scale_bar=True, stat_scale=None, current_col='g',
                      max_conds=None, min_conds=None, legend=True, ss_names=True, param_names=True):
    """
    Based on vis_sample. Is called when the pdf should be shown next ot the sample.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    # Hyperparameters for plotting
    font_size=8.0 # fontsize of the labels
    col_bar = color_input # color of the bars for summstats and conductances
    col_minmax = color_input # color of the horizontal line indicating the max and min value of summstats and conds
    col_shade = color_input # color of the shade between the max and min values
    values_each = 100 # not so important. How many values we evaluate for the max and min values
    indicator_fraction = 0.8 # breath of the horizontal bars for max and min, should be within [0,1]
    opacity = 0.5 # opacity of the shade
    width = 0.35  # the width of the bars
    neuron_labels = ['AB/PD', 'LP', 'PY'] # labels for the legends
    scale_bar_breadth = 1000
    scale_bar_voltage_breadth = 50

    plot_bars=False

    if voltage_trace is None: data = m.gen_single(sample)
    else: data = voltage_trace

    params = sample

    stats_nan = deepcopy(stats)

    bar_scaling_factors = np.asarray([[1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000],
                                      [1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000],
                                      [1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000]])
    bar_vals = bar_scaling_factors[np.asarray(hyperparams.use_membrane)]

    if mem_dimensions is not None:
        params_trunc = params[mem_dimensions].tolist()
        params_trunc += params[-7:].tolist()
        bar_vals = bar_vals[mem_dimensions]
        params = np.asarray(params_trunc)


    if with_params:
        lticks = np.arange(len(params[:-7]))

        end_of_time_axis = len(params[:-7]) - 1 + width

        full_time = np.linspace(width/2-0.5, end_of_time_axis+0.5-width/2, values_each * len(params[:-7]))
        full_min_conds = np.tile(bar_vals * min_conds[:-7] / 0.628e-3, (values_each, 1))
        full_min_conds = full_min_conds.flatten(order='F')
        full_max_conds = np.tile(bar_vals * max_conds[:-7] / 0.628e-3, (values_each, 1))
        full_max_conds = full_max_conds.flatten(order='F')

        if plot_bars:
            axmemparams.bar(lticks + width / 2, bar_vals * params[:-7] / 0.628e-3, width,
                            bottom=min(1e-8, np.min(params[:-7])), color=col_bar)

        for k in range(len(params[:-7])):
            start_t = int(values_each*k+(1-indicator_fraction)/2*values_each)
            end_t   = int(values_each*(k+1)-(1-indicator_fraction)/2*values_each)
            time_diff = end_t - start_t
            axmemparams.plot(full_time[start_t:end_t][::time_diff-1], full_min_conds[start_t:end_t][::time_diff-1], c=col_minmax)
            axmemparams.plot(full_time[start_t:end_t][::time_diff-1], full_max_conds[start_t:end_t][::time_diff-1], c=col_minmax)
            axmemparams.fill_between(full_time[start_t:end_t][::time_diff-1], full_min_conds[start_t:end_t][::time_diff-1],
                                     full_max_conds[start_t:end_t][::time_diff-1], facecolor=col_shade, alpha=opacity)

        names = viz.get_labels_8pt(hyperparams, include_q10=False)[:-7]
        if mem_dimensions is not None:
            names = names[mem_dimensions]
        axmemparams.set_ylim((0, 1000)) # 850
        axmemparams.set_xticks(lticks + width / 2)
        new_names = []
        count = 0
        for n in names:
            #if int(bar_vals[count]):
            #    new_names.append(str(int(bar_vals[count])) + ' ' + 'x ' + n)
            #else:
            new_names.append(str(int(bar_vals[count]))+' '+'x '+n)
            count += 1
        if param_names: axmemparams.set_xticklabels(new_names, rotation='vertical', fontsize=font_size)
        else:
            axmemparams.axes.get_xaxis().set_visible(False)
            #axmemparams.axes.get_yaxis().set_visible(False)
        axmemparams.xaxis.set_tick_params(labelsize=font_size)
        axmemparams.yaxis.set_tick_params(labelsize=font_size)

        small_offset = [0.00, -0.0, -0.0, 0.0, -0.00, 0.0]
        font_decrease = 1.7
        if labels_:
            if mode == '13D':
                axmemparams.set_ylim((0, 1000))  # 850
                axmemparams.text(0.36, -1.04, 'Membrane conductances', fontsize=font_size,
                                 transform=axmemparams.transAxes)
                axmemparams.text(0.43, -1.20, r'$\mathdefault{[mS/cm}^2\mathdefault{]}$', fontsize=font_size, transform=axmemparams.transAxes)
            else:
                if mode_for_membrane_height == 'high':
                    axmemparams.set_ylim((0, 1000))
                    for i in range(6):  # 520 or so
                        axmemparams.text(-0.0 + i * 1.03, -650, 'x', fontsize=font_size / 2)
                    for i in range(6):
                        if bar_vals[i] == 1:
                            axmemparams.text(-0.2 + i * 1.02 + 0.18, -750,
                                             r'$1$', fontsize=font_size / font_decrease)
                        elif bar_vals[i] == 10:
                            axmemparams.text(-0.2 + i * 1.0 + 0.05, -750,
                                             r'$10$', fontsize=font_size / font_decrease)
                        else:
                            axmemparams.text(-0.2 + i * 1.0, -750,
                                             r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                                    include_multiplier=False, negative_num=False),
                                             fontsize=font_size / font_decrease)
                else:
                    for i in range(6):  # 520 or so
                        axmemparams.text(-0.0 + i * 1.03, -470, 'x', fontsize=font_size / 2)
                    for i in range(6):
                        if i==0 or i==1 or i == 2 or i == 3 or i == 4 or i == 5:         #-620
                            axmemparams.text(-0.2+i*1.0+small_offset[i], -530, r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),include_multiplier=False, negative_num=False), fontsize=font_size / font_decrease)
                        else:
                            axmemparams.text(-0.2+i*1.0+small_offset[i], -530, r'$%s$' % str(int(bar_vals[i])), fontsize=font_size/font_decrease)
                #axmemparams.text(0.11, -1.50, r'Membrane $\mathregular{\bar g}$', fontsize=font_size, transform=axmemparams.transAxes)
                axmemparams.text(0.11, -1.60, r'Membrane $\mathregular{\bar g}$', fontsize=font_size,
                                 transform=axmemparams.transAxes)
                axmemparams.text(0.22, -1.77, '[mS/cm' + chr(0x00b0 + 2) + ']', fontsize=font_size, transform=axmemparams.transAxes)

        lticks = np.arange(len(params[-7:]))
        end_of_time_axis = len(params[-7:])-1+width
        full_time = np.linspace(width/2-0.5, end_of_time_axis+0.5-width/2, values_each * len(params[-7:]))
        full_min_conds = np.tile(min_conds[-7:] * 1e-3, (values_each,1))
        full_min_conds = full_min_conds.flatten(order='F')
        full_max_conds = np.tile(max_conds[-7:] * 1e-3, (values_each, 1))
        full_max_conds = full_max_conds.flatten(order='F')

        full_min_conds *= 1e9
        full_max_conds *= 1e9
        if plot_bars:
            axsynparams.bar(lticks + width / 2, params[-7:]*1e-3, width, color=col_bar)

        for k in range(len(params[-7:])):
            start_t = int(values_each * k + (1 - indicator_fraction) / 2 * values_each)
            end_t = int(values_each * (k + 1) - (1 - indicator_fraction) / 2 * values_each)
            time_diff = end_t - start_t
            axsynparams.plot(full_time[start_t:end_t][::time_diff-1], full_min_conds[start_t:end_t][::time_diff-1], c=col_minmax)
            axsynparams.plot(full_time[start_t:end_t][::time_diff-1], full_max_conds[start_t:end_t][::time_diff-1], c=col_minmax)
            axsynparams.fill_between(full_time[start_t:end_t][::time_diff-1], full_min_conds[start_t:end_t][::time_diff-1],
                                     full_max_conds[start_t:end_t][::time_diff-1], facecolor=col_shade, alpha=opacity)

        if labels_:
            #axsynparams.text(0.27, -0.85, r'Synaptic $\mathregular{\bar g}$', fontsize=font_size, transform=axsynparams.transAxes)
            axsynparams.text(0.09, -1.04, 'Synaptic conductances', fontsize=font_size,
                             transform=axsynparams.transAxes)
            axsynparams.text(0.37, -1.19, '[nS]', fontsize=font_size, transform=axsynparams.transAxes)

        names = viz.get_labels_8pt(hyperparams, mathmode=True, include_q10=False)[-7:]
        axsynparams.set_yscale('log')
       # axsynparams.set_ylim((1e-8*1e-3*1e9, 1.3*1e-3*1e-3*1e9))
        axsynparams.set_xticks(lticks + width / 2)
        #axsynparams.set_yticks([0.01, 1.0, 100])
        axsynparams.set_ylim([0.01, 1000])
        if param_names: axsynparams.set_xticklabels(names, rotation='vertical', fontsize=font_size)
        else:
            axsynparams.axes.get_xaxis().set_visible(False)
        axsynparams.xaxis.set_tick_params(labelsize=font_size)
        axsynparams.yaxis.set_tick_params(labelsize=font_size)

        axsynparams.spines['right'].set_visible(False)
        axsynparams.spines['top'].set_visible(False)
        axmemparams.spines['right'].set_visible(False)
        axmemparams.spines['top'].set_visible(False)

        axmemparams.tick_params(width=2.0 * 0.666, length=5.0 * 0.666)
        axsynparams.tick_params(width=2.0 * 0.666, length=5.0 * 0.666)

        axmemparams.tick_params(width=2.0 * 0.4, length=5.0 * 0.4, which='minor')
        axsynparams.tick_params(width=2.0 * 0.4, length=5.0 * 0.4, which='minor')

    if with_ss:
        lticks = np.arange(len(stats))

        if stat_scale is None:
            stats[8:] *= 2000
        min_stats_scaled = deepcopy(min_stats)
        max_stats_scaled = deepcopy(max_stats)
        if stat_scale is None:
            min_stats_scaled[8:] = min_stats_scaled[8:] * 2000
            max_stats_scaled[8:] = max_stats_scaled[8:] * 2000
        if plot_bars:
            axss.bar(lticks + width / 2, stats, width, color=col_bar)

        end_of_time_axis = len(stats) - 1 + width
        full_time = np.linspace(width / 2 - 0.5, end_of_time_axis + 0.5 - width / 2, values_each * len(stats))
        full_min_ss = np.tile(min_stats_scaled, (values_each, 1))
        full_min_ss = full_min_ss.flatten(order='F')
        full_max_ss = np.tile(max_stats_scaled, (values_each, 1))
        full_max_ss = full_max_ss.flatten(order='F')

        for k in range(len(stats)):
            start_t = int(values_each * k + (1 - indicator_fraction) / 2 * values_each)
            end_t = int(values_each * (k + 1) - (1 - indicator_fraction) / 2 * values_each)
            time_diff = end_t - start_t
            axss.plot(full_time[start_t:end_t][::time_diff-1], full_min_ss[start_t:end_t][::time_diff-1], c=col_minmax)
            axss.plot(full_time[start_t:end_t][::time_diff-1], full_max_ss[start_t:end_t][::time_diff-1], c=col_minmax)
            axss.fill_between(full_time[start_t:end_t][::time_diff-1], full_min_ss[start_t:end_t][::time_diff-1],
                                     full_max_ss[start_t:end_t][::time_diff-1], facecolor=col_shade, alpha=opacity)

        if labels_:
            axss.text(0.33,  -0.68, 'Summary statistics', fontsize=font_size, transform=axss.transAxes)
            axss.text(0.322, -0.80, '[st. dev. of samples]', fontsize=font_size, transform=axss.transAxes)

            nan_pos = np.where(np.isnan(stats_nan))[0]
            if stat_scale is not None:
                axss.scatter(nan_pos+width/2, 1.7*np.ones_like(nan_pos),
                             c=col_minmax, s=25.0, zorder=2, marker='x')
            else:
                axss.scatter(nan_pos + width / 2, 1900 * np.ones_like(nan_pos),
                             c=col_minmax, s=25.0, zorder=2, marker='x')

        # add some text for labels, title and axes ticks
        names = []
        for num in range(15):
            names.append(get_summ_stat_name_text(num))
        #axss.set_yticks([-4, -2, 0, 2, 4])
        axss.set_yticks([-2, -1, 0, 1, 2])
        #axss.set_yticklabels([r'$-4 \sigma$', '$-2 \sigma$', '0', '$2 \sigma$', '$4 \sigma$'])
        axss.set_yticklabels(['$\mathdefault{-2} \sigma$', '$\mathdefault{-}\sigma$', '0', '$\sigma$', '$\mathdefault{2} \sigma$'])
        axss.set_xticks(lticks + width / 2)
        if ss_names: axss.set_xticklabels(names, rotation='vertical', fontsize=font_size)
        else:
            axss.axes.get_xaxis().set_visible(False)
            #axss.axes.get_yaxis().set_visible(False)
        axss.xaxis.set_tick_params(labelsize=font_size)
        axss.yaxis.set_tick_params(labelsize=font_size)
        if stat_scale is not None:
            axss.set_ylim([-2.0, 2.0])
        else:
            axss.set_ylim([-450, 2100])

        axss.spines['right'].set_visible(False)
        axss.spines['top'].set_visible(False)

        axss.tick_params(width=2.0 * 0.666, length=5.0 * 0.666)

        #axss.get_xaxis().set_tick_params(
        #    which='both', direction='out', labelsize=font_size*3)
    sns.set(style="ticks", font_scale=1)
    sns.despine()

    if save_fig:
        plt.savefig('../../thesis_results/pdf/'+date_today+'_sample_prinz_'+case+'_{}_{}.pdf'.format(test_idx[0], counter),
                    bbox_inches='tight')
        plt.savefig('../../thesis_results/png/'+date_today+'_sample_prinz_'+case+'_{}_{}.png'.format(test_idx[0], counter),
                    bbox_inches='tight')
        plt.savefig('../../thesis_results/svg/'+date_today+'_sample_prinz_'+case+'_{}_{}.svg'.format(test_idx[0], counter),
                    bbox_inches='tight')


    if axmemparams is not None and axss is not None:
        return  axss, axmemparams, axsynparams
    elif axss is not None:
        return axss
    elif axmemparams is not None:
        return axmemparams, axsynparams




def vis_ss_barplot(m, s, sample, hyperparams, stats=None, t_on=None, t_off=None, with_ss=True, with_params=True, voltage_trace=None,
                      test_idx=None, case=None, title=None, date_today=None, counter=0, save_fig=False, legend_offset=0.0,
                      axss=None, axmemparams=None, axsynparams=None, max_stats=None, min_stats=None,
                      mem_dimensions=None, mode='13D', mode_for_membrane_height=None, labels_=True, color_input='k',
                      stat_mean=None, stat_std=None, scale_bar=True, stat_scale=None, current_col='g',
                      max_conds=None, min_conds=None, legend=True, ss_names=True, param_names=True):
    """
    Based on vis_sample. Is called when the pdf should be shown next ot the sample.

    :param m: generator object, from m = netio.create_simulators(params)[0]
    :param s: summstat object, from s = netio.create_summstats(params)
    :param sample: membrane/synaptic conductances
    :param t_on:
    :param t_off:
    :param with_ss: bool, True if bars for summary stats are wanted
    :param with_params: bool, True if bars for parameters are wanted
    :return: figure object
    """

    # Hyperparameters for plotting
    font_size=8.0 # fontsize of the labels
    col_bar = color_input # color of the bars for summstats and conductances
    col_minmax = color_input # color of the horizontal line indicating the max and min value of summstats and conds
    col_shade = color_input # color of the shade between the max and min values
    values_each = 100 # not so important. How many values we evaluate for the max and min values
    indicator_fraction = 0.8 # breath of the horizontal bars for max and min, should be within [0,1]
    opacity = 0.5 # opacity of the shade
    width = 0.35  # the width of the bars
    neuron_labels = ['AB/PD', 'LP', 'PY'] # labels for the legends
    scale_bar_breadth = 1000
    scale_bar_voltage_breadth = 50

    plot_bars=False

    params = sample

    stats_nan = deepcopy(stats)

    bar_scaling_factors = np.asarray([[1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000],
                                      [1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000],
                                      [1.0, 100.0, 100.0, 10.0, 100.0, 1.0, 10000, 10000]])
    bar_vals = bar_scaling_factors[np.asarray(hyperparams.use_membrane)]

    if mem_dimensions is not None:
        params_trunc = params[mem_dimensions].tolist()
        params_trunc += params[-7:].tolist()
        bar_vals = bar_vals[mem_dimensions]
        params = np.asarray(params_trunc)


    if with_ss:
        lticks = np.arange(len(stats))

        if stat_scale is None:
            stats[8:] *= 2000
        min_stats_scaled = deepcopy(min_stats)
        max_stats_scaled = deepcopy(max_stats)
        if stat_scale is None:
            min_stats_scaled[8:] = min_stats_scaled[8:] * 2000
            max_stats_scaled[8:] = max_stats_scaled[8:] * 2000
        if plot_bars:
            axss.bar(lticks + width / 2, stats, width, color=col_bar)

        end_of_time_axis = len(stats) - 1 + width
        full_time = np.linspace(width / 2 - 0.5, end_of_time_axis + 0.5 - width / 2, values_each * len(stats))
        full_min_ss = np.tile(min_stats_scaled, (values_each, 1))
        full_min_ss = full_min_ss.flatten(order='F')
        full_max_ss = np.tile(max_stats_scaled, (values_each, 1))
        full_max_ss = full_max_ss.flatten(order='F')

        for k in range(len(stats)):
            start_t = int(values_each * k + (1 - indicator_fraction) / 2 * values_each)
            end_t = int(values_each * (k + 1) - (1 - indicator_fraction) / 2 * values_each)
            time_diff = end_t - start_t
            axss.plot(full_time[start_t:end_t][::time_diff-1], full_min_ss[start_t:end_t][::time_diff-1], c=col_minmax)
            axss.plot(full_time[start_t:end_t][::time_diff-1], full_max_ss[start_t:end_t][::time_diff-1], c=col_minmax)
            axss.fill_between(full_time[start_t:end_t][::time_diff-1], full_min_ss[start_t:end_t][::time_diff-1],
                                     full_max_ss[start_t:end_t][::time_diff-1], facecolor=col_shade, alpha=opacity)

        if labels_:
            axss.text(0.33,  -0.68, 'Summary statistics', fontsize=font_size, transform=axss.transAxes)
            axss.text(0.322, -0.80, '[st. dev. of samples]', fontsize=font_size, transform=axss.transAxes)

            nan_pos = np.where(np.isnan(stats_nan))[0]
            if stat_scale is not None:
                axss.scatter(nan_pos+width/2, 1.7*np.ones_like(nan_pos),
                             c=col_minmax, s=25.0, zorder=2, marker='x')
            else:
                axss.scatter(nan_pos + width / 2, 1900 * np.ones_like(nan_pos),
                             c=col_minmax, s=25.0, zorder=2, marker='x')

        # add some text for labels, title and axes ticks
        names = []
        for num in range(15):
            names.append(get_summ_stat_name_text(num))
        #axss.set_yticks([-4, -2, 0, 2, 4])
        axss.set_yticks([-2, -1, 0, 1, 2])
        #axss.set_yticklabels([r'$-4 \sigma$', '$-2 \sigma$', '0', '$2 \sigma$', '$4 \sigma$'])
        axss.set_yticklabels(['$\mathdefault{-2} \sigma$', '$\mathdefault{-}\sigma$', '0', '$\sigma$', '$\mathdefault{2} \sigma$'])
        axss.set_xticks(lticks + width / 2)
        if ss_names: axss.set_xticklabels(names, rotation='vertical', fontsize=font_size)
        else:
            axss.axes.get_xaxis().set_visible(False)
            #axss.axes.get_yaxis().set_visible(False)
        axss.xaxis.set_tick_params(labelsize=font_size)
        axss.yaxis.set_tick_params(labelsize=font_size)
        if stat_scale is not None:
            axss.set_ylim([-2.0, 2.0])
        else:
            axss.set_ylim([-450, 2100])

        axss.spines['right'].set_visible(False)
        axss.spines['top'].set_visible(False)

        axss.tick_params(width=2.0 * 0.666, length=5.0 * 0.666)

        #axss.get_xaxis().set_tick_params(
        #    which='both', direction='out', labelsize=font_size*3)
    sns.set(style="ticks", font_scale=1)
    sns.despine()

    if save_fig:
        plt.savefig('../../thesis_results/pdf/'+date_today+'_sample_prinz_'+case+'_{}_{}.pdf'.format(test_idx[0], counter),
                    bbox_inches='tight')
        plt.savefig('../../thesis_results/png/'+date_today+'_sample_prinz_'+case+'_{}_{}.png'.format(test_idx[0], counter),
                    bbox_inches='tight')
        plt.savefig('../../thesis_results/svg/'+date_today+'_sample_prinz_'+case+'_{}_{}.svg'.format(test_idx[0], counter),
                    bbox_inches='tight')


    if axmemparams is not None and axss is not None:
        return  axss, axmemparams, axsynparams
    elif axss is not None:
        return axss
    elif axmemparams is not None:
        return axmemparams, axsynparams
