import numpy as np
import os
import sys

sys.path.append(os.getcwd())
sys.path.append('model/setup')
sys.path.append('model/inference')
sys.path.append('model/simulator')
import time
import netio
from find_pyloric import params_are_bounded
from copy import deepcopy
import delfi.distribution as dd

################################################################################
# seeding by time stamp
seed1 = time.time()
seed = int((seed1 % 1) * 1e7)
rng = np.random.RandomState(seed=seed)

################################################################################
hyper_params = netio.load_setup("train_31D_R1_BigPaper")

# define prior
seed_p = rng.randint(1e7)
hyper_params.seed = seed_p
num_dim = np.sum(hyper_params.use_membrane) + 7
p = dd.Uniform(-np.sqrt(3) * np.ones(num_dim), np.sqrt(3) * np.ones(num_dim), seed=hyper_params.seed)

################################################################################
# sample from prior and run simulations
pool_size = 28  # 28 worked

import dill as pickle

with open('results/31D_nets/191001_seed1_Exper11deg.pkl', 'rb') as file:
    inf_SNPE_MAF, log, params = pickle.load(file)

summstats_experimental = np.load('results/31D_experimental/190807_summstats_prep845_082_0044.npz')['summ_stats']
posterior_prev_round = inf_SNPE_MAF.predict([summstats_experimental])


# sample from prior and run simulations
num_sim_per_cpu_inside = 2520  # 90 worked

hyper_params_inside = netio.load_setup("train_31D_R1_BigPaper")
num_dim_inside= np.sum(hyper_params.use_membrane) + 7
p_inside = dd.Uniform(-np.sqrt(3) * np.ones(num_dim_inside),
                      np.sqrt(3) * np.ones(num_dim_inside),
                      seed=hyper_params_inside.seed)

def sim_f():
    start_time = time.time()

    conductance_params = np.empty((0, 31))
    num_success = 0
    num_iter = 0
    while num_success < num_sim_per_cpu_inside:
        sample_param = posterior_prev_round.gen(n_samples=5000)
        sample_param = np.asarray(sample_param)
        bounded_params = np.asarray(params_are_bounded(deepcopy(sample_param), p_inside, normalized=True))
        num_success += np.sum(bounded_params)
        conductance_params = np.append(conductance_params, sample_param[bounded_params], axis=0)
        num_iter += 1
    print('num_iter', num_iter)

    # sample from posterior
    conductance_params = np.asarray(conductance_params)
    print('np.shape(conductance_params)', np.shape(conductance_params))
    conductance_params = conductance_params[:num_sim_per_cpu_inside]
    print('Overall time:  ', time.time() - start_time)
    return conductance_params

for k in range(1000):
    print('Iteration:  ', k)

    data = []
    params_seed = np.ones(pool_size)
    data.append(sim_f())

    outfile = 'results/31D_samples/02_cond_vals/params_' + str(k) + '.npz'
    np.savez_compressed(outfile, conductance_params=data)
