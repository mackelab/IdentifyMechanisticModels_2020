import numpy as np

import delfi.distribution as dd
import delfi.generator as dg

import prinzetal
import summstats
import time

from parameters import ParameterSet

import os
dirname = os.path.dirname(__file__)
filedir_setups = os.path.join(dirname, '../setup/')

neumodels = ParameterSet(filedir_setups+'models.prm') # bit stupid to jump out of dir and back in. But there was a stupid error.

setups_filename = filedir_setups+'setups.prm'
setups_dict = ParameterSet(setups_filename)

def load_setup(name):
    return setups_dict[name]

def load(filename):
    params = ParameterSet('{}.prm'.format(filename))
    with np.load('{}.npz'.format(filename), allow_pickle=True) as zdata:
        data = { k: zdata[k] for k in zdata.files }
        
    return params, data

def save(dest, params, **kwargs):
    filename = time.strftime('{}/%d_%b_%H_%M_%S'.format(dest))
    params.save(url='{}.prm'.format(filename))
    np.savez('{}.npz'.format(filename), **kwargs, allow_pickle=True)
    return filename


# get conductances of all neurons (given in models.prm) and return it in the list ret
def create_neurons(neuron_list):
    ret = []
    for n in neuron_list:
        neuron = np.asarray(neumodels[n[0]][n[1]]) * n[2]
        ret.append(neuron)
    return ret


# creates the actual simulator object (multiple objects for multiple cores). This object will later call the simulation.
def create_simulators(params):
    neurons = create_neurons(params.neurons)
    proctolin = np.asarray(params.proctolin)
    #neurons = np.asarray(neurons)
    #neurons = neurons[np.asarray(params.use_membrane)==True].tolist() # only the membrane conductances that are fixed

    if params.seed is not None:
        seeds = [ int(params.seed + i) for i in range(params.n_cores) ]
    else:
        seeds = [ None for _ in range(params.n_cores) ]

    tmax = params.t_burnin + params.t_window

    mlist = [ prinzetal.Prinzetal(tmax, 
                                  params.dt, 
                                  neurons,
                                  proctolin,
                                  params,
                                  noise_fact=params.noise_fact, 
                                  cython=params.cython,
                                  prior_log=True,
                                  seed=seed,
                                  transform_=params.transform,
                                  init_val=params.init_val,
                                  prior=create_prior(params, log=True),
                                  **params.model_params)
              for seed in seeds ]
    return mlist


# creates a prior for inference
def create_prior(params, log=False):

    if params.novel_prior:
        assert params.comp_neurons is None, "Is you are using a novel prior, you can not use comp_neurons"

        # rows:    LP, PY, PD
        # columns: the eight membrane conductances
        # contains the minimal values that were used by Prinz et al.
        low_val = 0.0
        membrane_cond_mins = np.asarray([[100, 2.5, 2, 10, 5, 50, 0.01, low_val],  # PM
                                         [100, low_val, 4, 20, low_val, 25, low_val, 0.02],  # LP
                                         [100, 2.5, low_val, 40, low_val, 75, low_val, low_val]]) * 0.628e-3  # PY

        # contains the maximal values that were used by Prinz et al.
        membrane_cond_maxs = np.asarray([[400, 5.0, 6, 50, 10, 125, 0.01, low_val],  # PM
                                         [100, low_val, 10, 50, 5, 100, 0.05, 0.03],  # LP
                                         [500, 10, 2, 50, low_val, 125, 0.05, 0.03]]) * 0.628e-3  # PY

        ranges = np.asarray([100, 2.5, 2, 10, 5, 25, 0.01, 0.01]) * 0.628e-3
        membrane_cond_mins = membrane_cond_mins - ranges
        membrane_cond_maxs = membrane_cond_maxs + ranges

        membrane_cond_mins[membrane_cond_mins<0.0] = 0.0

        use_membrane = np.asarray(params.use_membrane)
        membrane_used_mins = membrane_cond_mins[use_membrane == True].flatten()
        membrane_used_maxs = membrane_cond_maxs[use_membrane == True].flatten()

        # proctolin
        proctolin_gbar_mins = [0.0, 0.0, 0.0]
        proctolin_gbar_maxs = np.asarray([6.0, 8.0, 0.0]) * 1e-6
        use_proctolin = params.use_proctolin

        syn_dim_mins = np.ones_like(params.true_params) * params.syn_min  # syn_min is the start of uniform interval
        syn_dim_maxs = np.ones_like(params.true_params) * params.syn_max  # syn_max is the end of uniform interval
        syn_dim_maxs[0] *= 10.0

        if log:
            syn_dim_mins = np.log(syn_dim_mins)
            syn_dim_maxs = np.log(syn_dim_maxs)

        # q10 values for synapses
        gbar_q10_syn_mins = np.asarray([1.0, 1.0])
        gbar_q10_syn_maxs = np.asarray([2.0, 2.0])
        tau_q10_syn_mins  = np.asarray([1.0, 1.0])
        tau_q10_syn_maxs  = np.asarray([4.0, 4.0])
        use_gbar_syn = np.asarray(params.Q10_gbar_syn)
        use_tau_syn  = np.asarray(params.Q10_tau_syn)
        gbar_q10_syn_used_mins = gbar_q10_syn_mins[use_gbar_syn].flatten()
        gbar_q10_syn_used_maxs = gbar_q10_syn_maxs[use_gbar_syn].flatten()
        tau_q10_syn_used_mins  = tau_q10_syn_mins[use_tau_syn].flatten()
        tau_q10_syn_used_maxs  = tau_q10_syn_maxs[use_tau_syn].flatten()

        # q10 values for membrane
        gbar_q10_mem_mins = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        gbar_q10_mem_maxs = np.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        use_gbar_mem = np.asarray(params.Q10_gbar_mem)
        gbar_q10_mem_used_mins = gbar_q10_mem_mins[use_gbar_mem].flatten()
        gbar_q10_mem_used_maxs = gbar_q10_mem_maxs[use_gbar_mem].flatten()

        if use_proctolin:
            membrane_and_sny_mins = np.concatenate((membrane_used_mins, proctolin_gbar_mins, syn_dim_mins, gbar_q10_syn_used_mins, tau_q10_syn_used_mins, gbar_q10_mem_used_mins))
            membrane_and_sny_maxs = np.concatenate((membrane_used_maxs, proctolin_gbar_maxs, syn_dim_maxs, gbar_q10_syn_used_maxs, tau_q10_syn_used_maxs, gbar_q10_mem_used_maxs))
        else:
            membrane_and_sny_mins = np.concatenate((membrane_used_mins, syn_dim_mins, gbar_q10_syn_used_mins, tau_q10_syn_used_mins, gbar_q10_mem_used_mins))
            membrane_and_sny_maxs = np.concatenate((membrane_used_maxs, syn_dim_maxs, gbar_q10_syn_used_maxs, tau_q10_syn_used_maxs, gbar_q10_mem_used_maxs))
    else:
        if params.comp_neurons is not None:
            neuron_1 = np.asarray(create_neurons(params.comp_neurons[0]))
            neuron_2 = np.asarray(create_neurons(params.comp_neurons[1]))
            membrane_cond_mins = np.minimum(neuron_1, neuron_2)
            membrane_cond_maxs = np.maximum(neuron_1, neuron_2)
            membrane_cond_mins[membrane_cond_mins==0.0]   += 1e-20
            membrane_cond_maxs[membrane_cond_maxs == 0.0] += 1e-20
            params.use_membrane = membrane_cond_mins!=membrane_cond_maxs
        else:
            # rows:    LP, PY, PD
            # columns: the eight membrane conductances
            # contains the minimal values that were used by Prinz et al.
            low_val = 0.0
            membrane_cond_mins = np.asarray([[100, 2.5,     2,       10, 5,       50,  0.01,    low_val ], # PM
                                             [100, low_val, 4,       20, low_val, 25,  low_val, 0.02    ],   # LP
                                             [100, 2.5,     low_val, 40, low_val, 75,  low_val, low_val ]]) * 0.628e-3 # PY

            # contains the maximal values that were used by Prinz et al.
            membrane_cond_maxs = np.asarray([[400, 5.0,     6,       50, 10,      125, 0.01,    low_val ], # PM
                                             [100, low_val, 10,      50, 5,       100, 0.05,    0.03    ],   # LP
                                             [500, 10,      2,       50, low_val, 125, 0.05,    0.03    ]]) * 0.628e-3 # PY

        security_factor = 1.25 # factor that we increase the margin used by Prinz et al. with
        membrane_cond_mins *= (2.0-security_factor)
        membrane_cond_maxs *= security_factor

        membrane_cond_maxs[membrane_cond_maxs == 0.0] = np.asarray([1.0, 1.0, 0.01])*0.628e-3

        use_membrane = np.asarray(params.use_membrane)
        membrane_used_mins = membrane_cond_mins[use_membrane == True].flatten()
        membrane_used_maxs = membrane_cond_maxs[use_membrane == True].flatten()

        syn_dim_mins = np.ones_like(params.true_params) * params.syn_min # syn_min is the start of uniform interval
        syn_dim_maxs = np.ones_like(params.true_params) * params.syn_max # syn_max is the end of uniform interval

        if log:
            syn_dim_mins = np.log(syn_dim_mins)
            syn_dim_maxs = np.log(syn_dim_maxs)

        membrane_and_sny_mins = np.concatenate((membrane_used_mins, syn_dim_mins))
        membrane_and_sny_maxs = np.concatenate((membrane_used_maxs, syn_dim_maxs))

    return dd.Uniform(membrane_and_sny_mins, membrane_and_sny_maxs, seed=int(params.seed))


# function to create the summary statistics
def create_summstats(params):
    return summstats.classes[params.summstats_class](t_on=params.t_burnin, 
                                                     t_off=params.t_burnin+params.t_window, 
                                                     include_pyloric_ness=params.include_pyloric_ness,
                                                     include_plateaus=params.include_plateau,
                                                     seed=int(params.seed))


# creates the prior, summstats, and a generator for creating samples
def create_delfi(params):
    mlist = create_simulators(params)
    p = create_prior(params)
    s = create_summstats(params)

    g = dg.MPGenerator(models=mlist, prior=p, summary=s) # MPGenerator could also just be 'Default'

    return mlist, p, s, g
