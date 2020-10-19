import numpy as np
import time

from . import ChannelOmniCython_comp

# def solver(protocols, act_var,inact_var)
# protocols: array of V values
# act_var: array of act_var values (OUTPUT)
# inact_var: array of inact_var values (OUTPUT)
# I_channel: array of I_channel values (OUTPUT)

class ChannelSim:
    def __init__(self, param_values, third_exp_model=True, seed=None):
        self.param_values = np.asarray(param_values)
        self.param_names = ['a_act','b_act','c_act','d_act','e_act','f_act','g_act','h_act']

        self.third_exp_model = third_exp_model
        if self.third_exp_model:
            self.param_names += ['k_act', 'l_act']
            self.solver = ChannelOmniCython_comp.expeuler3
        else:
            self.solver = ChannelOmniCython_comp.expeuler2

        self.params = {name : self.param_values[0, i].astype(float)
                       for i, name in enumerate(self.param_names)}

        self.seed = seed
        if seed is not None:
            ChannelOmniCython_comp.seed(seed)
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, protocols, E_channel, fact_inward):
        """Simulates the model for a specified time duration."""

        noise_std = 0.5

        traces = {}
        for k, protocol in protocols.items():
            # extract voltages and time from protocol
            V = protocol.values[:,1:].T
            V = V + noise_std*self.rng.randn(*V.shape)
            t = protocol.values[:,0]    # attention: time step varies, but we will assume that it is constant
            tstep = float(np.mean(np.diff(t))) # ms

            # explictly cast everything to double precision
            t = t.astype(np.float64)
            V = V.astype(np.float64)
            act_var = np.zeros_like(V).astype(np.float64)
            I_channel = np.zeros_like(V).astype(np.float64)

            if self.third_exp_model:
                I_channel = self.solver(t, V, act_var, I_channel, tstep,
                                    1,
                                    self.params['a_act'],self.params['b_act'],
                                    self.params['c_act'],self.params['d_act'],
                                    self.params['e_act'],self.params['f_act'],
                                    self.params['g_act'],self.params['h_act'],
                                    self.params['k_act'],self.params['l_act'],
                                    E_channel, fact_inward)
            else:
                I_channel = self.solver(t, V, act_var, I_channel, tstep,
                                    1,
                                    self.params['a_act'],self.params['b_act'],
                                    self.params['c_act'],self.params['d_act'],
                                    self.params['e_act'],self.params['f_act'],
                                    self.params['g_act'],self.params['h_act'],
                                    E_channel,fact_inward)

            traces[k] = {'data': I_channel, 'time': t}

        return traces
