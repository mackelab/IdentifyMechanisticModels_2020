import numpy as np
import scipy.signal as ss

from delfi.simulator.BaseSimulator import BaseSimulator


class GLM(BaseSimulator):
    def __init__(self, duration=100, len_filter=9, seed=None, seed_input=None):
        """GLM simulator

        Parameters
        ----------
        duration : int
            Duration of traces in ms
        len_filter : int
            Length of filter
        seed : int or None
            If set, randomness across runs is disabled
        seed_input : int or None
            If set, randomness in input is controlled by seed_input rather than
            by seed
        """
        super(GLM, self).__init__(dim_param=len_filter+1, seed=seed)
        self.duration = duration
        self.len_filter = len_filter
        self.seed_input = seed_input
        self.n_params = self.len_filter + 1

        # parameters that globally govern the simulations
        self.dt = 1
        self.t = np.arange(0, self.duration, self.dt)

        # input: gaussian white noise N(0, 1)
        if self.seed_input is None:
            new_seed = self.gen_newseed()
        else:
            new_seed = self.seed_input
        self.rng_input = np.random.RandomState(seed=new_seed)
        self.I = self.rng_input.randn(len(self.t)).reshape(-1,1)

    def gen_single(self, params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of the forward run. Additional entries can be present.
        """
        params = np.asarray(params)

        assert params.ndim == 1, 'params.ndim must be 1'
        assert params.shape[0] == self.n_params, 'params.shape[0] must be dim params long'

        b0 = params[0]
        b0.astype(float)
        h = params[1:]
        h.astype(float)

        # simulation
        psi = b0 + ss.lfilter(h, 1, self.I, axis=0)

        # psi goes through a sigmoid non-linearity: firing probability
        z = 1 /(1 + np.exp(-psi))

        # sample the spikes
        N = 1   # number of trials
        y = self.rng.uniform(size=(len(self.t), N)) < z
        y = np.sum(y, axis=1)

        return {'data': y.reshape(-1), 'I': self.I.reshape(-1)}
