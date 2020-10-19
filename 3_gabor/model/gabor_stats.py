import numpy as np

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats


class maprfStats(BaseSummaryStats):
    """SummaryStats class for the GLM

    Calculates sufficient statistics
    """
    def __init__(self, n_summary=442, seed=None):
        super(maprfStats, self).__init__(seed=seed)
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

            sta = np.dot(x['data'], x['I']) 
            
            n_spikes = x['data'].sum()
            if n_spikes > 0. :
                sta /= n_spikes

            sta -= sta.mean()

            stats.append(np.hstack((sta.reshape(1,-1), np.atleast_2d(x['data'].sum()))))

        return np.asarray(stats).reshape(-1,stats[-1].size)
