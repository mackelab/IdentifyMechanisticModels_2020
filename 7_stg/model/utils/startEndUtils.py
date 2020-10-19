import numpy as np


def check_equality(sample1, sample2, margin=0.1, stats_std=None, mode='percentage'):
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
    elif mode == 'dataset':
        exp_stds = stats_std
        percentage_diff = np.abs(sample1 - sample2)[:15] / exp_stds[:15]
        if np.all(percentage_diff < margin):
            return True
        else:
            return False


def check_num_different_conds(params1, params2, all_index1, all_index2):
    s1 = params1
    s2 = params2
    ratio = s1 / s2
    run_true = (ratio > np.ones_like(ratio) * 2.0) | (ratio < np.ones_like(ratio) / 2.0)
    ratio_diff = np.abs(s1 - s2)
    run_syn = (ratio_diff > np.ones_like(ratio_diff) * 2.0)
    if np.sum(run_true[:-7]) > 7.5 and \
            np.sum(run_true[:8]) > 1.5 and np.sum(run_true[8:16]) > 1.5 and \
            np.sum(run_true[16:24]) > 1.5 and \
            np.sum(run_syn[-7:]) > 4.5:
        print('Num of diff samples is ', np.sum(run_true),
              '. Num or diff memb conds:', np.sum(run_true[:-7]),
              '. Num of diff syn conds:', np.sum(run_syn[-7:]))
        all_index1.append(params1)
        all_index2.append(params2)
    return all_index1, all_index2