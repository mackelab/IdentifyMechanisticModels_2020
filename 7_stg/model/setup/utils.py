import numpy as np
import delfi.distribution as dd
import netio
from copy import deepcopy


def logistic_fct(array):
    return 1 / (1+np.exp(-array))


def inv_logistic_fct(array):
    return -np.log(1/array - 1)


import numpy as np


def check_equality(sample1, sample2, margin=0.1, mode='percentage'):
    if mode == 'percentage':
        percentage_diff = np.abs(sample1 - sample2) / sample2
        if np.all(percentage_diff < margin):
            return True
        else:
            return False
    elif mode == 'experimental':
        exp_stds = np.asarray([279, 133, 113, 150, 109, 60, 169, 216, 0.040, 0.059, 0.054, 0.065, 0.034, 0.054, 0.060])
        percentage_diff = np.abs(sample1 - sample2)[:15] / exp_stds
        if np.all(percentage_diff < margin):
            return True
        else:
            return False


def check_num_different_conds(sample_params_scaled, index1, index2, all_index1, all_index2):
    s1 = sample_params_scaled[index1]
    s2 = sample_params_scaled[index2]
    ratio = s1 / s2
    run_true = (ratio > np.ones_like(ratio) * 3.0) | (ratio < np.ones_like(ratio) / 3.0)
    run_true_2_0 = (ratio > np.ones_like(ratio) * 2.0) | (ratio < np.ones_like(ratio) / 2.0)
    ratio_diff = np.abs(s1 - s2)
    run_syn = (ratio_diff > np.ones_like(ratio_diff) * 3.0)
    run_syn_2_0 = (ratio_diff > np.ones_like(ratio_diff) * 2.0)
    if np.sum(run_true_2_0[:-7]) > 2.5 and np.sum(run_syn_2_0[-7:]) > 4.5 and \
            ratio_diff[-7] > 3.2 and ratio_diff[-6] > 2.7:
        print('Index1:', index1, 'Index2:', index2, '. Num of diff samples is ', np.sum(run_true),
              '. Num or diff memb conds:', np.sum(run_true[:-7]), 'and', np.sum(run_true_2_0[:-7]),
              '. Num of diff syn conds:', np.sum(run_syn[-7:]))
        all_index1.append(index1)
        all_index2.append(index2)
    return all_index1, all_index2


def set_trn_data(inf_SNPE_MAF, trn_data_old, n_train_round_R1):
    # CODE THAT IS COPIED FROM APT DELFI REPO. STORES THE OLD TRAINING DATA IN THE NETWORK

    from copy import deepcopy
    n_samples = deepcopy(n_train_round_R1)
    n_pilot = np.minimum(n_samples, len(inf_SNPE_MAF.unused_pilot_samples[0]))

    if n_pilot > 0 and inf_SNPE_MAF.generator.proposal is inf_SNPE_MAF.generator.prior:  # reuse pilot samples
        params = inf_SNPE_MAF.unused_pilot_samples[0][:n_pilot, :]
        stats = inf_SNPE_MAF.unused_pilot_samples[1][:n_pilot, :]
        inf_SNPE_MAF.unused_pilot_samples = \
            (inf_SNPE_MAF.unused_pilot_samples[0][n_pilot:, :],
             inf_SNPE_MAF.unused_pilot_samples[1][n_pilot:, :])

        n_samples -= n_pilot
        if n_samples > 0:
            params_rem = trn_data_old[0]
            stats_rem = trn_data_old[1]
            params = np.concatenate((params, params_rem), axis=0)
            stats = np.concatenate((stats, stats_rem), axis=0)
            # z-transform params and stats
    else:
        params = trn_data_old[0]
        stats = trn_data_old[1]

    params = (params - inf_SNPE_MAF.params_mean) / inf_SNPE_MAF.params_std
    stats = (stats - inf_SNPE_MAF.stats_mean) / inf_SNPE_MAF.stats_std
    trn_data = (params, stats)

    inf_SNPE_MAF.trn_datasets.append(trn_data)

    return inf_SNPE_MAF