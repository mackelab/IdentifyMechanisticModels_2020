#########################################################################################################
#
# Author: Michael Deistler
# Date: 27th of March 2019
# Usage: val1 = 1e-8
#        val2 = 1e-3
#        str1 = build_string(conductance_to_value_exp([val1]))
#        str2 = build_string(conductance_to_value_exp([val2]))
#        xticks = [3, 8]
#        ax.set_xticks([np.log(0.1 ** i) for i in xticks])
#        ax.set_xticklabels([r'$%s$'%str1, r'$%s$'%str2], fontsize=14.0)
#
#########################################################################################################
import numpy as np


def build_string(q_exp, include_multiplier=True, negative_num=True):
    # see unicode: https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts#Superscripts_and_subscripts_block
    s = ''
    for current_q in q_exp:
        if include_multiplier:
            s += current_q[0]
            s += '\cdot 10'
        else: s += '10'
        list_of_exponentials = []
        if len(current_q[1]) == 1:
            list_of_exponentials.append(int(current_q[1]))
        else:
            list_of_exponentials.append(int(current_q[1][0]))
            list_of_exponentials.append(int(current_q[1][1]))

        counter = 0
        for i in list_of_exponentials:
            if i == 1:
                # https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts#Superscripts_and_subscripts_block
                if counter == 0 and negative_num: s += chr(0x2070 + 11)
                s += chr(0x00b9)
            elif 2 <= i <= 3:
                if counter == 0 and negative_num: s += chr(0x2070 + 11)
                s += chr(0x00b0 + i)
            else:
                if counter == 0 and negative_num: s += chr(0x2070 + 11) # unicode sign for superscript minus
                s += chr(0x2070 + i)
            counter += 1
            #if current_q != q_exp[-1]:
            #    s += '\:|\:'
    return s


def conductance_to_value_exp(conductances):
    output = []
    for g in conductances:
        str_number = '%06.2e' % (g)
        value = str_number[0:3]
        if str_number[6] != '0':
            exponent = str_number[6:8]
        else:
            exponent = str_number[7]
        results = [value, exponent]
        output.append(results)
    return output


def build_string_gen(q_exp, positive_exp, add_value=True):
    # see unicode: https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts#Superscripts_and_subscripts_block
    all_strings = []
    run_var = 0
    for current_q in q_exp:
        s = ''
        if add_value:
            s += current_q[0]
            s += '\cdot 10'
        else:
            s += '10'

        i = int(current_q[1]) # current exponent to be a superscript

        if i == 1:
            # https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts#Superscripts_and_subscripts_block
            if not positive_exp[run_var]:
                s += chr(0x2070 + 11)
            s += chr(0x00b9)
        elif 2 <= i <= 3:
            if not positive_exp[run_var]:
                s += chr(0x2070 + 11)
            s += chr(0x00b0 + i)
        else:
            if not positive_exp[run_var]:
                s += chr(0x2070 + 11)
            s += chr(0x2070 + i)
        all_strings.append(s)
        run_var += 1
    return all_strings


def scale_to_experimental(stats):
    experimental_means = np.asarray([1509, 582, 399, 530, 221, -61, 803, 1141, 0.385, 0.264, 0.348, 0.148, -0.040, 0.533, 0.758])
    experimental_stds  = np.asarray([279,  133, 113, 150, 109, 60,  169, 216,  0.040, 0.059, 0.054, 0.065, 0.034,  0.054, 0.060])
    stats = (stats - experimental_means) / experimental_stds
    return stats


def get_summ_stat_name(num):
    if num == 0: return r'$T$'  # 'Cycle period'
    if num == 1: return r'$d^b_{PD}$'  # 'Burst length PD'
    if num == 2: return r'$d^b_{LP}$'  # 'Burst length LP'
    if num == 3: return r'$d^b_{PY}$'  # 'Burst length PY'
    if num == 4: return r'$\Delta t^{es}_{PD\mathrm{-}LP}$'  # 'End to start PD-LP'
    if num == 5: return r'$\Delta t^{es}_{LP\mathrm{-}PY}$'  # 'End to start LP-PY'
    if num == 6: return r'$\Delta t^{ss}_{PD\mathrm{-}LP}$'  # 'Start to start PD-LP'
    if num == 7: return r'$\Delta t^{ss}_{LP\mathrm{-}PY}$'  # 'Start to start LP-PY'
    if num == 8: return r'$d_{PD}$'  # 'Duty cycle PD'
    if num == 9: return r'$d_{LP}$'  # 'Duty cycle LP'
    if num == 10: return r'$d_{PY}$'  # 'Duty cycle PY'
    if num == 11: return r'$\Delta\theta_{PD\mathrm{-}LP}$'  # 'Phase gap PD-LP'
    if num == 12: return r'$\Delta\theta_{LP\mathrm{-}PY}$'  # 'Phase gap LP-PY'
    if num == 13: return r'$\theta_{LP}$'  # 'Phase LP'
    if num == 14: return r'$\theta_{PY}$'  # 'Phase PY'

def get_summ_stat_name_text(num):
    if num == 0: return r'$T$'  # 'Cycle period'
    if num == 1: return r'$d^{\mathdefault{b}}_{\mathdefault{AB}}$'  # 'Burst length PD'
    if num == 2: return r'$d^{\mathdefault{b}}_{\mathdefault{LP}}$'  # 'Burst length LP'
    if num == 3: return r'$d^{\mathdefault{b}}_{\mathdefault{PY}}$'  # 'Burst length PY'
    if num == 4: return r'$\Delta t^{\mathdefault{es}}_{\mathdefault{AB-LP}}$'  # 'End to start PD-LP'
    if num == 5: return r'$\Delta t^{\mathdefault{es}}_{\mathdefault{LP-PY}}$'  # 'End to start LP-PY'
    if num == 6: return r'$\Delta t^{\mathdefault{ss}}_{\mathdefault{AB-LP}}$'  # 'Start to start PD-LP'
    if num == 7: return r'$\Delta t^{\mathdefault{ss}}_{\mathdefault{LP-PY}}$'  # 'Start to start LP-PY'
    if num == 8: return r'$d_{\mathdefault{AB}}$'  # 'Duty cycle PD'
    if num == 9: return r'$d_{\mathdefault{LP}}$'  # 'Duty cycle LP'
    if num == 10: return r'$d_{\mathdefault{PY}}$'  # 'Duty cycle PY'
    if num == 11: return r'$\Delta\phi_{\mathdefault{AB-LP}}$'  # 'Phase gap PD-LP'
    if num == 12: return r'$\Delta\phi_{\mathdefault{LP-PY}}$'  # 'Phase gap LP-PY'
    if num == 13: return r'$\phi_{\mathdefault{LP}}$'  # 'Phase LP'
    if num == 14: return r'$\phi_{\mathdefault{PY}}$'  # 'Phase PY'


def get_summ_stat_name_asterisk(num):
    if num == 0: return r'$T$'  # 'Cycle period'
    if num == 1: return r'$d^b_{PD}$'  # 'Burst length PD'
    if num == 2: return r'$d^b_{LP}$'  # 'Burst length LP'
    if num == 3: return r'$d^b_{PY}$'  # 'Burst length PY'
    if num == 4: return r'$\Delta t^{es}_{PD\mathrm{-}LP}$'  # 'End to start PD-LP'
    if num == 5: return r'$\Delta t^{es}_{LP\mathrm{-}PY}$'  # 'End to start LP-PY'
    if num == 6: return r'$\Delta t^{ss}_{PD\mathrm{-}LP}$'  # 'Start to start PD-LP'
    if num == 7: return r'$\Delta t^{ss}_{LP\mathrm{-}PY}$'  # 'Start to start LP-PY'
    if num == 8: return r'$d_{PD}^{\#}$'  # 'Duty cycle PD'
    if num == 9: return r'$d_{LP}^{\#}$'  # 'Duty cycle LP'
    if num == 10: return r'$d_{PY}^{\#}$'  # 'Duty cycle PY'
    if num == 11: return r'$\Delta\theta_{PD\mathrm{-}LP}^{\#}$'  # 'Phase gap PD-LP'
    if num == 12: return r'$\Delta\theta_{LP\mathrm{-}PY}^{\#}$'  # 'Phase gap LP-PY'
    if num == 13: return r'$\theta_{LP}^{\#}$'  # 'Phase LP'
    if num == 14: return r'$\theta_{PY}^{\#}$'  # 'Phase PY'


# get the title of the synapses
def pick_synapse(num, mathmode=False):
    if mathmode:
        if num == 0: return r'$\mathdefault{AB-LP}$'
        if num == 1: return r'$\mathdefault{PD-LP}$'
        if num == 2: return r'$\mathdefault{AB-PY}$'
        if num == 3: return r'$\mathdefault{PD-PY}$'
        if num == 4: return r'$\mathdefault{LP-PD}$'
        if num == 5: return r'$\mathdefault{LP-PY}$'
        if num == 6: return r'$\mathdefault{PY-LP}$'
    else:
        if num == 0: return 'AB-LP'
        if num == 1: return 'PD-LP'
        if num == 2: return 'AB-PY'
        if num == 3: return 'PD-PY'
        if num == 4: return 'LP-PD'
        if num == 5: return 'LP-PY'
        if num == 6: return 'PY-LP'


def get_synapse_name(num):
    return r'$g_{%s}$' %(pick_synapse(num))


def get_membrane_name(num):
    return r'$g_{%s}$' %(pick_synapse(num))



def scale_to_experimental(stats):
    experimental_means = np.asarray([1509, 582, 399, 530, 221, -61, 803, 1141, 0.385, 0.264, 0.348, 0.148, -0.040, 0.533, 0.758])
    experimental_stds  = np.asarray([279,  133, 113, 150, 109, 60,  169, 216,  0.040, 0.059, 0.054, 0.065, 0.034,  0.054, 0.060])
    stats = (stats - experimental_means) / experimental_stds
    return stats
