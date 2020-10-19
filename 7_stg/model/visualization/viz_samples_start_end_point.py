import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from print_helper import conductance_to_value_exp, build_string, build_string_gen
import seaborn as sns
from matplotlib import lines
import matplotlib.gridspec as gridspec
import prinzdb
from print_helper import get_summ_stat_name, get_synapse_name, get_summ_stat_name_asterisk, scale_to_experimental
import sys
sys.path.append("../visualization")
import viz
from copy import deepcopy
import matplotlib.ticker


def vis_sample_plain(m, s, sample, axV=None, t_on=None, t_off=None, col=['k', 'k', 'k'], figsize=None,
               voltage_trace=None, time_len=None, fontscale=1.0, linescale=1.0, offset=0, cols=None,
               order='horizontal', scale_bar=[False],
               test_idx=None, case=None, show_xaxis=False, title=None, date_today=None, counter=0, legend=True,
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

    if figsize is None:
        figsize = (3*len(sample),3)
    if axV is None:
        if order == 'horizontal': _, ax = plt.subplots(1, len(sample), figsize=figsize)
        elif order == 'vertical': _, ax = plt.subplots(len(sample), 1, figsize=figsize)

    current_counter = 0

    for ii, current_sample in enumerate(sample):
        if len(sample) > 1: axV = ax[ii]
        else: axV = ax
        if voltage_trace is None:
            data = m.gen_single(current_sample, seed_sim=True, to_seed=418010)
        else:
            data = voltage_trace

        #summs = s.calc([data])[0]
        #print('summs', summs)

        Vx = data['data']

        current_col = 0
        for j in range(len(prinzdb.neutypes)):
            if time_len is not None:
                axV.plot(m.t[:time_len:5]/1000, Vx[j, 10000+offset[ii]:10000+offset[ii]+time_len:5] + 130.0 * (2 - j),
                         label=prinzdb.neutypes[j], lw=0.6, c=cols[ii])
            else:
                axV.plot(m.t/1000, Vx[j] + 120.0 * (2 - j), label=prinzdb.neutypes[j], lw=0.6, c=cols[ii])
            current_col += 1

        box = axV.get_position()

        if t_on is not None:
            axV.axvline(t_on, c='r', ls='--')

        if t_on is not None:
            axV.axvline(t_off, c='r', ls='--')

        axV.set_position([box.x0, box.y0, box.width, box.height])
        axV.axes.get_yaxis().set_ticks([])
        if not show_xaxis:
            axV.axes.get_xaxis().set_ticks([])
        else:
            axV.set_xlabel('time [seconds]')

        if legend: axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                              ncol=len(prinzdb.neutypes))

        axV.spines['right'].set_visible(False)
        axV.spines['top'].set_visible(False)
        axV.spines['bottom'].set_visible(False)
        axV.spines['left'].set_visible(False)

        plt.subplots_adjust(wspace=0.05)

        axV.set_title('')

        current_counter += 1

        if scale_bar[ii]:
            scale_bar_breadth = 0.5
            scale_bar_voltage_breadth = 50
            axV.plot(0 + np.arange(30*scale_bar_breadth)/30, -95 * np.ones_like(np.arange(30*scale_bar_breadth)), lw=1.0,
                     color='k')
            axV.plot(2.75 * np.ones_like(np.arange(scale_bar_voltage_breadth)),
                     -85 + np.arange(scale_bar_voltage_breadth), lw=1.0, color='k')
        else:
            scale_bar_breadth = 0.5
            scale_bar_voltage_breadth = 50
            axV.plot(0 + np.arange(30 * scale_bar_breadth) / 30, -95 * np.ones_like(np.arange(30 * scale_bar_breadth)), lw=1.0,
                     color='w')
            axV.plot(2.75 * np.ones_like(np.arange(scale_bar_voltage_breadth)),
                     -85 + np.arange(scale_bar_voltage_breadth), lw=1.0, color='w')

        print('hi')
        axV.set_ylim([-108, 320])


def vis_sample(m, s, sample, hyperparams, t_on=None, t_off=None, with_ss=True, with_params=True,
               mem_dimensions=None,mode2=None, mode='13D',
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
        print(np.shape(params_trunc))
        params = np.asarray(params_trunc)

    if with_params and with_ss:
        fig = plt.figure(figsize=(2,2.3))
        gs = gridspec.GridSpec(2, 3, width_ratios=[len(stats), len(params[:-7]), len(params[-7:])],
                               wspace=0.25, height_ratios=[0.7, 0.3])
        axV = plt.subplot(gs[0, :])
        axss = plt.subplot(gs[1, 0])
        axmemparams = plt.subplot(gs[1, 1])
        axsynparams = plt.subplot(gs[1, 2])
    elif with_params:
        fig = plt.figure(figsize=(2,2.3))
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
        scale_bar_breadth = scale_bar_breadth  * len(m.t) / time_len

    for j in range(len(prinzdb.neutypes)):
        if time_len is not None:
            axV.plot(m.t[10000+offset:10000+offset+time_len], Vx[j, 10000+offset:10000+offset+time_len] + 120.0 * (2 - j),
                     label=prinzdb.neutypes[j], lw=0.6, c='k')
        else:
            axV.plot(m.t, Vx[j] + 120.0 * (2 - j), label=prinzdb.neutypes[j], lw=0.6, c='k')
        current_col += 1
    if scale_bar:
        if mode2 == 'small':
            axV.plot(10860 + np.arange(scale_bar_breadth), 318 * np.ones_like(np.arange(scale_bar_breadth)), lw=1.0,
                     color='k', zorder=5)
            axV.text(10905, 324, '1 sec')

            import matplotlib.patches as patches
            rect = patches.Rectangle((11890, 234), 2000, 100, linewidth=1, facecolor='w', zorder=3)
            axV.add_patch(rect)

            axV.plot(13490 * np.ones_like(np.arange(scale_bar_voltage_breadth)),
                     318 - scale_bar_voltage_breadth + np.arange(scale_bar_voltage_breadth), lw=1.0, color='k', zorder=6)
            axV.text(11770, 270, '50 mV')
        else:
            axV.plot(10860 + np.arange(scale_bar_breadth), 318 * np.ones_like(np.arange(scale_bar_breadth)), lw=1.0,
                     color='k')
            axV.text(10905, 324, '1 sec')

            import matplotlib.patches as patches
            rect = patches.Rectangle((10900, 264), 700, 50, linewidth=1, facecolor='w', zorder=3)
            axV.add_patch(rect)

            axV.plot(11860 * np.ones_like(np.arange(scale_bar_voltage_breadth)),
                     318 - scale_bar_voltage_breadth + np.arange(scale_bar_voltage_breadth), lw=1.0, color='k')
            axV.text(10930, 270, '50 mV')



    if not legend and vis_legend:
        if mode2=='small':
            axV.text(-0.15, 0.75, 'AB/PD', transform=axV.transAxes)
            axV.text(-0.1, 0.45, 'LP', transform=axV.transAxes)
            axV.text(-0.1, 0.15, 'PY', transform=axV.transAxes)
        else:
            axV.text(-1540+offset_labels, 220, 'AB/PD')
            axV.text(-1050+offset_labels,  95, 'LP')
            axV.text(-1080+offset_labels, -30, 'PY')

    box = axV.get_position()

    if t_on is not None:
        axV.axvline(t_on, c='r', ls='--')

    if t_on is not None:
        axV.axvline(t_off, c='r', ls='--')

    axV.set_position([box.x0, box.y0, box.width, box.height])
    axV.axes.get_yaxis().set_ticks([])
    axV.axes.get_xaxis().set_ticks([])

    if legend: axV.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                          ncol=len(prinzdb.neutypes))
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
        axmemparams.set_ylim((0, 600))
        # axmemparams.set_ylabel('Membrane', fontsize=font_size)
        axmemparams.set_xticks(lticks + width / 2)
        if param_names:
            axmemparams.set_xticklabels(names, rotation='vertical')
        else:
            axmemparams.axes.get_xaxis().set_visible(False)
            axmemparams.axes.get_yaxis().set_visible(False)

        small_offset = [0.00, -0.0, -0.0, 0.0, -0.00, 0.0]
        font_decrease = 1.7
        if mode == '13D':
            if mode2 == 'small':
                for i in range(6):  # 520 or so
                    axmemparams.text(-0.0 + i * 1.03, -310, 'x')
                axmemparams.set_ylim((0, 600))  # 850
                small_offset = [0.15, -0.04, -0.1, 0.0, -0.02, 0.0]
                for i in range(6):
                    if i == 2 or i == 3 or i == 4 or i == 5:  # -620
                        axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -350,
                                         r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                                include_multiplier=False, negative_num=False))
                    else:
                        axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -350, r'$%s$' % str(int(bar_vals[i])))
                axmemparams.text(0.11, -0.73, r'Membrane $\mathregular{\bar g}$',
                                 transform=axmemparams.transAxes)
                axmemparams.text(0.22, -0.85, '[mS/cm' + chr(0x00b0 + 2) + ']',
                                 transform=axmemparams.transAxes)
            else:

                for i in range(6):  # 520 or so
                    axmemparams.text(-0.0 + i * 1.03, -390, 'x')
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
                axmemparams.text(0.11, -0.95, r'Membrane $\mathregular{\bar g}$',
                                 transform=axmemparams.transAxes)
                axmemparams.text(0.22, -1.12, '[mS/cm' + chr(0x00b0 + 2) + ']',
                                 transform=axmemparams.transAxes)
        else:
            if mode2 == 'small':
                axmemparams.set_ylim((0, 600))  # 850
                small_offset = [0.0, .0, -0.0, 0.0, -0.00, 0.0]
                for i in range(6):
                    axmemparams.text(-0.0 + i * 1.03, -310, 'x')
                    if bar_vals[i] == 1.0:
                        mystr = '1'
                        axmemparams.text(-0.2 + i * 1.03 + 0.1, -350,
                                         r'$%s$' % mystr)
                    elif bar_vals[i] == 10.0:
                            mystr = '10'
                            axmemparams.text(-0.2 + i * 1.005 + 0.02, -350,
                                             r'$%s$' % mystr)
                    else:
                            axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -350,
                                             r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                                    include_multiplier=False, negative_num=False))
                axmemparams.text(0.11, -0.73, r'Membrane $\mathregular{\bar g}$',
                                 transform=axmemparams.transAxes)
                axmemparams.text(0.22, -0.85, '[mS/cm' + chr(0x00b0 + 2) + ']',
                                 transform=axmemparams.transAxes)
            else:

                for i in range(6):  # 520 or so
                    axmemparams.text(-0.0 + i * 1.03, -390, 'x', fontsize=font_size / 2)
                axmemparams.set_ylim((0, 600))  # 850
                small_offset = [0.0, .0, -0.0, 0.0, -0.00, 0.0]
                for i in range(6):
                    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5:  # -620
                        axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -450,
                                         r'$%s$' % build_string(conductance_to_value_exp([bar_vals[i]]),
                                                                include_multiplier=False, negative_num=False))
                    else:
                        axmemparams.text(-0.2 + i * 1.0 + small_offset[i], -450, r'$%s$' % str(int(bar_vals[i])),
                                         fontsize=font_size / font_decrease)
                axmemparams.text(0.11, -0.95, r'Membrane $\mathregular{\bar g}$',
                                 transform=axmemparams.transAxes)
                axmemparams.text(0.22, -1.12, '[mS/cm' + chr(0x00b0 + 2) + ']',
                                 transform=axmemparams.transAxes)

        lticks = np.arange(len(params[-7:]))
        axsynparams.bar(lticks + width / 2, params[-7:] * 0.628e-3, width,
                        bottom=min(1e-8 * 0.628e-3, np.min(params[-7:] * 0.628e-3)), color='k')

        if mode2 == 'small':
            axsynparams.text(0.22, -0.73, r'Synaptic $\mathregular{\bar g}$',
                             transform=axsynparams.transAxes)
            axsynparams.text(0.4, -0.85, '[nS]',  transform=axsynparams.transAxes)
        else:
            axsynparams.text(0.22, -0.95, r'Synaptic $\mathregular{\bar g}$',
                             transform=axsynparams.transAxes)
            axsynparams.text(0.4, -1.12, '[nS]',  transform=axsynparams.transAxes)

        names = viz.get_labels(hyperparams, include_q10=False)[-7:]
        # axsynparams.set_ylabel('Synapses', fontsize=font_size)
        axsynparams.set_yscale('log')
        axsynparams.set_ylim((1e-8 * 1e-3, 1e-3 * 1e-3))
        axsynparams.set_xticks(lticks + width / 2)
        axsynparams.set_yticks([1e-11, 1e-9, 1e-7])
        if param_names:
            axsynparams.set_xticklabels(names, rotation='vertical',)
        else:
            axsynparams.axes.get_xaxis().set_visible(False)
            axsynparams.axes.get_yaxis().set_visible(False)

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
        axss.text(0.145, -1.12, '[st. dev. of experimental data]', fontsize=font_size, transform=axss.transAxes)

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
            '../../thesis_results/pdf/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.pdf'.format(test_idx[0],
                                                                                                     counter),
            bbox_inches='tight')
        plt.savefig(
            '../../thesis_results/png/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.png'.format(test_idx[0],
                                                                                                     counter),
            bbox_inches='tight', dpi=500)
        plt.savefig(
            '../../thesis_results/svg/' + date_today + '_sample_prinz_plain_' + case + '_{}_{}.svg'.format(test_idx[0],
                                                                                                     counter),
            bbox_inches='tight')

    return fig
