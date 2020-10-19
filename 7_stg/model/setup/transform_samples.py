import viz
import numpy as np
from copy import deepcopy


def transform_params(conductance_dataset, params):
    """
    Transform parameters to all be normalized

    :param conductance_dataset: dataset of conductance parameter values
    :param params: ParameterSet object
    :return: conductance parameter values in same range
    """
    lims = viz.calc_lims(params)
    new_lims = []
    log_sample_params = np.log(conductance_dataset)
    log_sample_params_scaled = deepcopy(log_sample_params)
    target_margin = np.abs(np.log(1e-8) - np.log(1e-3))
    dim_params = np.shape(conductance_dataset)[1]
    scaling_factors = []
    for i in range(dim_params):
        parameter_range = np.abs(lims[i, 0] - lims[i, 1])
        new_lims.append(lims[i] * target_margin / parameter_range)
        log_sample_params_scaled[:, i] = log_sample_params[:, i] * target_margin / parameter_range
        scaling_factors.append(target_margin / parameter_range)
    conductance_dataset = np.exp(log_sample_params_scaled)

    return conductance_dataset, new_lims, scaling_factors

def transform_params_norm(conductance_dataset, params, prior):
    """
    Transform parameters to all be normalized

    :param conductance_dataset: dataset of conductance parameter values
    :param params: ParameterSet object
    :return: conductance parameter values in same range
    """
    lims = viz.calc_lims(params)
    new_lims = []
    log_sample_params = np.log(conductance_dataset)
    params_mean = prior.mean
    params_std = prior.std
    conductance_dataset = np.exp((log_sample_params - params_mean) / params_std)
    log_sample_params_scaled = deepcopy(log_sample_params)
    target_margin = np.abs(np.log(1e-8) - np.log(1e-3))
    dim_params = np.shape(conductance_dataset)[1]
    scaling_factors = []
    for i in range(dim_params):
        parameter_range = np.abs(lims[i, 0] - lims[i, 1])
        new_lims.append(lims[i] * target_margin / parameter_range)
        log_sample_params_scaled[:, i] = log_sample_params[:, i] * target_margin / parameter_range
        scaling_factors.append(target_margin / parameter_range)
    conductance_dataset = np.exp(log_sample_params_scaled)

    return conductance_dataset, new_lims, scaling_factors


def inv_transform_params(conductance_dataset, params):
    """
    Bring parameters to original range

    :param conductance_dataset: dataset of conductance parameter values
    :param params: ParameterSet object
    :return: conductance parameter values in original spacing
    """
    lims = viz.calc_lims(params)
    log_sample_params = np.log(conductance_dataset)
    log_sample_params_scaled = deepcopy(log_sample_params)
    target_margin = np.abs(np.log(1e-8) - np.log(1e-3))
    dim_params = np.shape(conductance_dataset)[1]
    for i in range(dim_params):
        parameter_range = np.abs(lims[i, 0] - lims[i, 1])
        log_sample_params_scaled[:, i] = log_sample_params[:, i] / target_margin * parameter_range
    conductance_dataset = np.exp(log_sample_params_scaled)

    return conductance_dataset











































import sys
import numpy as np
sys.path.append('../setup/')
sys.path.append('../simulator/')
sys.path.append('../inference/')
import netio

def calc_deviation(summstats_gt, summstats_rep, precisions):
    summstats_gt  = np.asarray(summstats_gt)
    summstats_rep = np.asarray(summstats_rep)
    precisions = np.asarray(precisions)
    return np.nanmean(np.linalg.norm(precisions*(summstats_gt - summstats_rep), axis=1))


def extract_dim_of_highest_variation(dim_per_neuron):
    coeffs_ABPD = extract_CV_ABPD()
    coeffs_LP   = extract_CV_LP()
    coeffs_PY   = extract_CV_PY()

    indizes_ABPD = find_largest_CV(coeffs_ABPD, dim_per_neuron)
    indizes_LP   = find_largest_CV(coeffs_LP, dim_per_neuron)
    indizes_PY   = find_largest_CV(coeffs_PY, dim_per_neuron)

    list_ABPD = np.zeros(8)
    list_LP = np.zeros(8)
    list_PY = np.zeros(8)

    if dim_per_neuron > 0:
        list_ABPD[indizes_ABPD] = 1
        list_LP[indizes_LP] = 1
        list_PY[indizes_PY] = 1

    return np.asarray([list_ABPD, list_LP, list_PY])


def find_largest_CV(coeffs, dim_per_neuron):
    indizes = np.argsort(coeffs)[-dim_per_neuron:]
    indizes = indizes[::-1]
    return indizes

def extract_CV_ABPD():
    neuron_PM = []
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_PM.append(netio.create_neurons(neurons)[0].tolist())
    neurons = [['PM', 'PM_1', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_PM.append(netio.create_neurons(neurons)[0].tolist())
    neurons = [['PM', 'PM_2', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_PM.append(netio.create_neurons(neurons)[0].tolist())
    neurons = [['PM', 'PM_3', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_PM.append(netio.create_neurons(neurons)[0].tolist())
    neurons = [['PM', 'PM_4', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_PM.append(netio.create_neurons(neurons)[0].tolist())
    neuron_PM = np.transpose(neuron_PM)

    coeff_PM = []
    for cond in neuron_PM:
        coeff_PM.append(coefficient_of_variation(cond))
    return coeff_PM

def extract_CV_LP():
    neuron_LP = []
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_0', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_LP.append(netio.create_neurons(neurons)[1].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_1', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_LP.append(netio.create_neurons(neurons)[1].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_LP.append(netio.create_neurons(neurons)[1].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_3', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_LP.append(netio.create_neurons(neurons)[1].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_4', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_LP.append(netio.create_neurons(neurons)[1].tolist())

    neuron_LP = np.transpose(neuron_LP)

    coeff_LP = []
    for cond in neuron_LP:
        coeff_LP.append(coefficient_of_variation(cond))
    return coeff_LP

def extract_CV_PY():
    neuron_PY = []
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_0', 0.000628]]
    neuron_PY.append(netio.create_neurons(neurons)[2].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_1', 0.000628]]
    neuron_PY.append(netio.create_neurons(neurons)[2].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_2', 0.000628]]
    neuron_PY.append(netio.create_neurons(neurons)[2].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_3', 0.000628]]
    neuron_PY.append(netio.create_neurons(neurons)[2].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_4', 0.000628]]
    neuron_PY.append(netio.create_neurons(neurons)[2].tolist())
    neurons = [['PM', 'PM_0', 0.000628], ['LP', 'LP_2', 0.000628], ['PY', 'PY_5', 0.000628]]
    neuron_PY.append(netio.create_neurons(neurons)[2].tolist())
    neuron_PY = np.transpose(neuron_PY)

    coeff_PY = []
    for cond in neuron_PY:
        coeff_PY.append(coefficient_of_variation(cond))
    return coeff_PY



def coefficient_of_variation(conductances):
    cond_std = np.std(conductances)
    cond_mean = np.mean(conductances)
    if cond_mean == 0.0:
        if cond_std == 0.0:
            return 0.0
        else:
            return 'error'
    else:
        return cond_std / cond_mean * 100

def print_neuron_names(mylist):
    for num in mylist:
        if num == 0: print('g_Na')
        if num == 1: print('g_CaT')
        if num == 2: print('g_CaS')
        if num == 3: print('g_A')
        if num == 4: print('g_KCa')
        if num == 5: print('g_Kd')
        if num == 6: print('g_H')
        if num == 7: print('g_leak')























