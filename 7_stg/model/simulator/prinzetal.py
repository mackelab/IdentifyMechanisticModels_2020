import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator
import HH
import summstats
import prinzdb
import multiprocessing as mp
import sys
sys.path.append('../setup')
import netio
from copy import deepcopy
import time

# Remove and import from utils.py
def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x

def param_invtransform(prior_log, x):
    if prior_log:
        x[-7:] = np.exp(x[-7:])
        return x
    else:
        return x

class Prinzetal(BaseSimulator):
    def __init__(self, 
                 tmax, 
                 dt, 
                 models,
                 proctolin,
                 params,
                 noise_fact = 0.0, 
                 cython=False, 
                 prior_log=False, 
                 seed=None,
                 transform_=False,
                 init_val=0.0,
                 prior=None,
                 **kwargs):
        """Hodgkin-Huxley neuron simulator

        Parameters
        ----------
        tmax : float
            Maximum simulation time (mS)
        dt : float
            Simulation timestep (mS)
        models : 1d array
            List of neurons in the system (for the format, see HH.py)
        cython : bool
            If True, will use cython version of simulator
        prior_log : bool
            Set to true if prior is logarithmix (recommended)
        seed : int or None
            If set, randomness across runs is disabled
        """
        super().__init__(dim_param=len(prinzdb.syntypes), seed=seed)

        self.cython = cython
        self.tmax = tmax
        self.dt = dt
        self.t = np.arange(0, self.tmax, self.dt)
        self.prior_log = prior_log
        self.noise_fact = noise_fact
        self.transform_=transform_
        self.prior=prior
        self.kwargs = kwargs
        self.init_val = init_val

        if self.noise_fact == 0:
            self.I = np.zeros((len(models), len(self.t)))

        if cython:
            self.solver = HH.cHH()
        else:
            self.solver = HH.HH()

        # parameters that globally govern the simulations
        self.models = models
        self.proctolin = proctolin

        if params.comp_neurons is not None:
            neuron_1 = np.asarray(netio.create_neurons(params.comp_neurons[0]))
            neuron_2 = np.asarray(netio.create_neurons(params.comp_neurons[1]))
            membrane_cond_mins = np.minimum(neuron_1, neuron_2)
            membrane_cond_maxs = np.maximum(neuron_1, neuron_2)
            params.use_membrane = membrane_cond_mins != membrane_cond_maxs
        self.params = params


    def gen(self, params_list, n_reps=1, pbar=None):
        data_list = []
        iter_ = 0
        start_seed = deepcopy(self.seed)
        for param in params_list:
            rep_list = []
            for r in range(n_reps):
                rep_list.append(self.gen_single(param, seed_sim=True, to_seed=int(start_seed + iter_)))
            data_list.append(rep_list)
            if pbar:
                pbar.update(1)
            iter_ += 1

        return data_list


    def gen_single(self, all_params, seed_sim=False, to_seed=0):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        conductance_params : list or np.array, 1d of length dim_param
            Parameter vector
        seed: bool, whether to seed or not

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """

        number_syn_q10 = np.sum(self.params.Q10_gbar_syn) + np.sum(self.params.Q10_tau_syn) + np.sum(self.params.Q10_gbar_mem)
        if np.any(self.params.Q10_gbar_mem):
            new_all_params = all_params[:-np.sum(self.params.Q10_gbar_mem)]
            gbar_q10_membrane = all_params[-np.sum(self.params.Q10_gbar_mem):]
        else:
            new_all_params = all_params
            gbar_q10_membrane = self.params.Q10_gbar_mem_default

        if np.any(self.params.Q10_tau_syn):
            if np.any(self.params.Q10_gbar_syn):
                gbar_q10 = new_all_params[-4:-2]
                tau_q10 = new_all_params[-2:]
            else:
                gbar_q10 = self.params.Q10_gbar_syn_default
                tau_q10 = new_all_params[-2:]
        else:
            if np.any(self.params.Q10_gbar_syn):
                gbar_q10 = new_all_params[-2:]
                tau_q10 = self.params.Q10_tau_syn_default
            else:
                gbar_q10 = self.params.Q10_gbar_syn_default
                tau_q10 = self.params.Q10_tau_syn_default

        if number_syn_q10 > 0:
            conductance_params = all_params[:-number_syn_q10] # membrane and synapse gbar
        else:
            conductance_params = all_params  # membrane and synapse gbar

        # gbar_q10[0] = q10 for glutamate synapse
        # gbar_q10[1] = q10 for cholinergic synapse
        gbar_q10_params_full = [gbar_q10[0], gbar_q10[1], gbar_q10[0], gbar_q10[1], gbar_q10[0], gbar_q10[0], gbar_q10[0]]
        tau_q10_params_full  = [ tau_q10[0],  tau_q10[1],  tau_q10[0],  tau_q10[1],  tau_q10[0],  tau_q10[0],  tau_q10[0]]

        if self.transform_: conductance_params = conductance_params * self.prior.std + self.prior.mean

        if seed_sim:
            if to_seed:
                self.reseed(to_seed)
            else:
                seed1 = time.time()
                seed = int((seed1 % 1) * 1e7)
                self.reseed(seed)
        conductance_params = param_invtransform(self.prior_log, np.asarray(conductance_params))

        assert conductance_params.ndim == 1, 'params.ndim must be 1'

        membrane_params = conductance_params[0:-7]
        synaptic_params = conductance_params[-7:]
        conns = prinzdb.build_conns(-synaptic_params)

        # build the used membrance conductances as parameters. Rest as fixed values.
        current_num = 0
        membrane_conds = []
        for neuron_num in range(3): # three neurons
            membrane_cond = []
            for cond_num in range(8): # 8 membrand conductances per neuron
                if self.params.use_membrane[neuron_num][cond_num]:
                    membrane_cond.append(membrane_params[current_num])
                    current_num += 1
                else:
                    membrane_cond.append(self.models[neuron_num][cond_num])
            if self.params.use_proctolin:
                membrane_cond.append(membrane_params[current_num])
                current_num += 1
            else:
                membrane_cond.append(self.proctolin[neuron_num]) # proctolin is made part of the membrane conds here.
            membrane_conds.append(np.asarray(membrane_cond))

        if self.noise_fact != 0:
            self.I = self.rng.normal(scale = self.noise_fact, size=(len(self.models), len(self.t)))

        # calling the solver --> HH.HH()
        data = self.solver.sim_time(self.dt, 
                                    self.t, 
                                    self.I, 
                                    membrane_conds, # membrane conductances
                                    conns, # synaptic conductances (always variable)
                                    gbar_q10_params_full,
                                    tau_q10_params_full,
                                    gbar_q10_membrane,
                                    verbose=False,
                                    start_val_input=self.init_val,
                                    **self.kwargs)

        return {'data': data['Vs'],
                'params' : conductance_params,
                'tmax': self.tmax,
                'dt': self.dt,
                'I': self.I}

