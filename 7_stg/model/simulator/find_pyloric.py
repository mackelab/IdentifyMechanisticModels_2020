##################################################################################
#                                                                                #
# Michael: I think that this is the file to evaluate whether a rhythm is pyloric #
# It does not exactly follow Prinz et al.'s definition of a pyloric rhythm. It   #
# simply checks whether there are bursts in the the data --> not NaN.            #
#                                                                                #
##################################################################################

import sys
import numpy as np
#import dill
#from sortedcontainers import SortedList
import sys
import numpy as np
#import dill
import os
from parameters import ParameterSet
from copy import deepcopy
import time


def merge_datasets(filedir, dataset=None, outfile_name=None, adhere_prior=False,
                   prior=None, exclude_NaN=False, enforce_pyloric=False, verbose=False):
    """
    This function takes the dataset provided in the variable dataset and extends it by the files in filedir.
    It adds all variables where the phasegap is well defined and prints the number of pyloric samples.

    :param filedir: string to folder, e.g. '../results/samples/samples_13D_new'
    :param dataset: string to dataset, e.g. '../results/samples/pyloricsamples_13D.npz'
    :param outfile_name: string, name of file to write to, e.g. 'pyloricsamples_13D.npz'
    :param adhere_prior: bool, if True, samples are deleted if they come from a region outside of the prior
    :param verbose: bool
    :return: saves the extended dataset (summ_stats, seed, and params) as .npz file
    """

    assert adhere_prior==False or prior is not None, 'You have to give a prior if you want to adhere its boundaries'
    # getting all files from directory filedir
    files = os.listdir(filedir)

    # checking for hidden files. If the file starts with '.', we discard it. Also discard readme.txt
    filenames = []
    for file in files:
        if file[0] != '.' and file != 'readme.txt':
            filenames.append(file)

    picked_params = []
    picked_stats = []
    picked_seeds = []
    last_len = 0

    # we will create multiple files (as Jakob suggested). Each contains a fraction of all samples. This reduces the risk of
    # loosing the entire dataset due to some small mistake. Here, we loop over those multiple files.
    for fname in filenames:
        data = np.load("{}/{}".format(filedir, fname))  # samples_dir = results/samples/
        summstats = data['data']
        counter_local = 0
        counter_pyloric = 0
        param_data_exp = np.exp(data['params'])
        param_data = data['params']
        seed_data = data['seeds']
        for current_summ_stat in summstats:
            current_params_exp = param_data_exp[counter_local]
            current_params = param_data[counter_local]
            current_seeds = seed_data[counter_local]
            if current_summ_stat[-1] > 0.0:
                counter_pyloric += 1
            if exclude_NaN:
                if (not enforce_pyloric and not np.isnan(current_summ_stat[4]))\
                        or (enforce_pyloric and current_summ_stat[-1] == 1):
                    if (adhere_prior and params_are_bounded(current_params_exp, prior)) or not adhere_prior:
                        picked_params.append(current_params)
                        picked_stats.append(current_summ_stat[:-1])  # drop last one which simply indicates pyloric-ness.
                        picked_seeds.append(current_seeds)
            else:
                if (adhere_prior and params_are_bounded(current_params_exp, prior)) or not adhere_prior:
                    picked_params.append(current_params)
                    picked_stats.append(current_summ_stat[:-1])  # drop last one which simply indicates pyloric-ness.
                    picked_seeds.append(current_seeds)
            counter_local += 1

        if verbose:
            print("File "+fname+" contained {} samples".format(counter_local))
            print("of which {} go into the dataset".format(len(picked_params) - last_len))
            print("of which {} are pyloric".format(counter_pyloric))
        last_len = len(picked_params)

    if dataset is not None:
        data = np.load(dataset)  # samples_dir = results/samples/pyloricsamples_13D
        picked_params_gt = data['params']
        picked_stats_gt = data['stats']
        picked_seeds_gt = data['seeds']
        picked_seeds = np.concatenate((picked_seeds_gt, np.asarray(picked_seeds)))
        picked_stats = np.concatenate((picked_stats_gt, np.asarray(picked_stats)))
        picked_params = np.concatenate((picked_params_gt, np.exp(np.asarray(picked_params))))
        print('Successfully merged datasets')

    np.savez_compressed(outfile_name, seeds=picked_seeds, params=picked_params,
                        stats=picked_stats)


def find_pyloric_like(filename, outfile_name, num_stds=2.0):
    data = np.load(filename)  # samples_dir = results/samples/
    summstats = data['stats']

    picked_params = []
    picked_stats = []
    picked_seeds = []
    counter = 0

    data_seeds = data['seeds']
    data_stats = data['stats']
    data_params = data['params']

    for ss in summstats:
        if check_ss(ss, num_stds=num_stds):
            picked_seeds.append(data_seeds[counter])
            picked_stats.append(data_stats[counter])
            picked_params.append(data_params[counter])
        counter += 1

    np.savez_compressed(outfile_name, seeds=picked_seeds, params=picked_params,
                        stats=picked_stats)

"""
def check_ss(summstats):
    if summstats[0] < 952.0   or summstats[0] > 2067.0: return False
    if summstats[1] < 317.0   or summstats[1] > 847.0:  return False
    if summstats[2] < 172.0   or summstats[2] > 625.0:  return False
    if summstats[3] < 230.0   or summstats[3] > 830.0:  return False
    if summstats[4] < 4.0     or summstats[4] > 439.0:  return False
    if summstats[5] < -181.0  or summstats[5] > 59.0:   return False
    if summstats[6] < 464.0   or summstats[6] > 1142.0: return False
    if summstats[7] < 709.0   or summstats[7] > 1572.0: return False
    if summstats[8] < 0.305   or summstats[8] > 0.464:  return False
    if summstats[9] < 0.146   or summstats[9] > 0.383:  return False
    if summstats[10] < 0.240  or summstats[10] > 0.456: return False
    if summstats[11] < 0.018  or summstats[11] > 0.278: return False
    if summstats[12] < -0.108 or summstats[12] > 0.029: return False
    if summstats[13] < 0.426  or summstats[13] > 0.640: return False
    if summstats[14] < 0.638  or summstats[14] > 0.877: return False
    return True
"""

def check_ss(summstats, num_stds=2.0):
    experimental_means = np.asarray([1509, 582, 399, 530, 221, -61, 803, 1141, 0.385, 0.264, 0.348, 0.148, -0.040, 0.533, 0.758])
    experimental_stds = np.asarray( [279, 133, 113, 150, 109, 60, 169, 216, 0.040, 0.059, 0.054, 0.065, 0.034, 0.054, 0.060])

    for ss, em, es in zip(summstats, experimental_means, experimental_stds):
        if ss < em - num_stds*es or ss > em + num_stds*es:
            return False
    return True


def merge_samples(filedir, name='params'):
    """
    Since sampling from MAFs requires rejection sampling, we do this externally.
    This function then merges the externally created files into a single list.

    :param filedir: string to folder, e.g. '../results/samples/samples_13D_new'
    :return: all_conds: list of samples
    """
    files = os.listdir(filedir)
    filenames = []
    for file in files:
        if file[0] != '.' and file != 'readme.txt':
            filenames.append(file)
    all_conds = []
    for fname in filenames:
        data = np.load("{}/{}".format(filedir, fname))  # samples_dir = results/samples/
        conductances = data[name]
        for cond in conductances:
            all_conds.append(cond)
    return all_conds


def params_are_bounded(conductances, prior, normalized=False):
    if conductances.ndim == 1:
        conductances = [conductances]
    vals = []
    for cond in conductances:
        conds = deepcopy(cond)
        if not normalized:
            conds[:-7] = np.log(conds[:-7])
        if np.all(prior.lower < conds) and np.all(prior.upper > conds):
            vals.append(True)
        else: vals.append(False)
    return vals

def single_params_are_bounded(conductances, prior, normalized=False):
    if conductances.ndim == 1:
        conductances = [conductances]
    vals = []
    for cond in conductances:
        conds = deepcopy(cond)
        if not normalized:
            conds[:-7] = np.log(conds[:-7])
        if np.all(prior.lower < conds) and np.all(prior.upper > conds):
            vals.append(True)
        else: vals.append(False)
    return vals

def Kaans_function():
    """
    Can be discarded. Was used as a base for the function merge_datasets.
    """

    strict = False
    if sys.argv[1] == "strict":
        strict = True
        filenames = sys.argv[2:]
        print("Strict run")
    else:
        filenames = sys.argv[1:]

    picked_params = []
    picked_stats = []

    class EntryWrapper:
        def __init__(self, params, summstats, target):
            self.params = params
            self.summstats = summstats[:4]
            self.dist = np.sum((self.summstats - target) ** 2)

        def __lt__(self, other):
            return not np.isnan(self.dist) and self.dist < other.dist

    counter = 0

    last_len = 0

    # we will create multiple files (as Jakob suggested). Each contains a fraction of all samples. This reduces the risk of
    # loosing the entire dataset due to some small mistake. Here, we loop over those multiple files.
    for fname in filenames:
        f = open(fname, "rb")

        counter_local = counter + 0
        print("Opened file {}".format(fname))
        while True:
            try:
                entry = dill.load(f)
            except EOFError:
                break

            counter += 1

            params = entry[0]
            summstats = entry[1]
            assert len(summstats) == 1
            summstats = summstats[0]

            if (not strict and not np.isnan(summstats[4])) or (strict and not np.isnan(summstats[-1])):
                picked_params.append(params)
                picked_stats.append(summstats)

        print("File contained {} samples".format(counter - counter_local))
        print("of which {} pyloric".format(len(picked_params) - last_len))
        last_len = len(picked_params)

    print("went through {} elements".format(counter))

    ofname = "pyloricsamples"
    if strict:
        ofname = "pyloricsamples_strict"

    picked_params, picked_stats = np.array(picked_params), np.array(picked_stats)

    np.savez_compressed(ofname, params=picked_params, stats=picked_stats)

    print("Collected {}".format(len(picks)))

