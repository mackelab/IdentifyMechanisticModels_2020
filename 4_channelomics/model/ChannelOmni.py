import numpy as np
import os
import pandas as pd

from delfi.simulator.BaseSimulator import BaseSimulator


class ChannelOmni(BaseSimulator):
    def __init__(self, third_exp_model=False, seed=None):
        """Channel simulator

        Parameters
        ----------
        third_exp_model : bool
            If True, model set to 3rd order expansion
        seed : int or None
            If set, randomness across runs is disabled
        """
        channel_type = 'k'
        self.cython = True

        self.third_exp_model = third_exp_model
        if self.third_exp_model:
            n_params = 2+8
        else:
            n_params = 2+6

        super().__init__(dim_param=n_params, seed=seed)
        self.channel_type = channel_type

        path1 = os.path.dirname(__file__)

        # protocols
        self.protocols = {}
        for p in ['v_act', 'v_inact', 'v_deact', 'v_ap', 'v_ramp']:
            self.protocols[p] = pd.read_csv(
                path1 + '/protocols/' + self.channel_type+'_channel_'+p+'.dat', sep='\t')
            self.protocols[p] = self.protocols[p].drop(self.protocols[p].columns[-1], axis=1)

        # reversal potential for channel of interest
        E_channel = {'k': -86.7, 'na': 50, 'ca': 135, 'ih': -45}

        # inward current pre-multiplied by -1 (see page 15 of Podlasky et al. 2017)
        fact_inward = {'k': 1, 'na': -1, 'ca': -1, 'ih': 1}

        self.E_channel = E_channel[channel_type]
        self.fact_inward = fact_inward[channel_type]

        from . import ChannelOmniCython as bm

        self.bm = bm

    def gen_single(self, params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = np.asarray(params)
        assert params.ndim == 1, 'params.ndim must be 1'

        channel_seed = self.gen_newseed()

        channel = self.bm.ChannelSim(params.reshape(1, -1), self.third_exp_model, seed=channel_seed)
        states = channel.sim_time(self.protocols, self.E_channel, self.fact_inward)

        return states
