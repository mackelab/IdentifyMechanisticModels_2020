import numpy as np
import os
import pickle

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy.signal import resample


class ChannelOmniStats(BaseSummaryStats):
    """SummaryStats class for Channel model

    Calculates summary statistics based on PC reconstruction coefficients
    """
    def __init__(self, seed=None):
        super().__init__(seed=seed)

        self.channel_type = 'k'

        path1 = os.path.dirname(__file__)

        self.pcs = pickle.load(open(path1+'/pca/pow1_sumstats_lfs.pkl', 'rb'))

    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        stats = []

        protocols = ['v_act', 'v_inact', 'v_deact', 'v_ap', 'v_ramp']

        for r in range(len(repetition_list)):
            trace = repetition_list[r]

            for protocol in protocols:
                I = trace[protocol]['data']

                a = self.pcs[protocol[2:]].pcs
                a = np.hstack((a, np.ones((a.shape[0], 1))))
                b = I.reshape(-1)
                coef, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

                stats.append(coef)

        return np.asarray(stats).reshape(1, -1)
