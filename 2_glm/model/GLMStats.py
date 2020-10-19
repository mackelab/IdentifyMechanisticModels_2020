import numpy as np

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats


class GLMStats(BaseSummaryStats):
    """SummaryStats class for the GLM

    Calculates sufficient statistics
    """
    def __init__(self, n_summary=10, seed=None):
        super(GLMStats, self).__init__(seed=seed)
        self.n_summary = n_summary

    def calc(self, repetition_list):
        """Calculate sufficient statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.arrray, 2d with n_reps x n_summary
        """
        stats = []
        for r in range(len(repetition_list)):
            x = repetition_list[r]

            N = x['data'].shape[0]
            N_xcorr = self.n_summary-1

            sta = np.correlate(x['data'], x['I'], 'full')[N-1:N+N_xcorr-1]
            sum_stats_vec = np.concatenate((np.array([np.sum(x['data'])]), sta))

            stats.append(sum_stats_vec)

        return np.asarray(stats)
