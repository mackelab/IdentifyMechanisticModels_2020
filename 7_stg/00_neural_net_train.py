import numpy as np
import time
import sys
sys.path.append("model/setup")
sys.path.append("model/simulator")
sys.path.append("model/inference")
sys.path.append("model/visualization")
sys.path.append("model/utils")

import netio
import train_utils as tu
import dill as pickle
import delfi.distribution as dd
from delfi.generator import Default

params_filename = "train_net"
params = netio.load_setup(params_filename)
mlist = netio.create_simulators(params)
dimensions = np.sum(params.use_membrane) + 7


################################################################################
#                                   Load data                                  #
################################################################################

filedir = "results/31D_samples/pyloricsamples_31D_noNaN_3.npz"
pilot_data, _, params_mean, params_std = tu.load_trn_data_normalize(filedir, params)
print('We use', len(pilot_data[0]), 'training samples.')


################################################################################
#                                  Define prior                                #
################################################################################

# create standard prior for the actual conductances
prior = netio.create_prior(params, log=True)
prior_norm = dd.Uniform(lower=-np.sqrt(3)*np.ones(dimensions), upper=np.sqrt(3)*np.ones(dimensions))

################################################################################
#                                  Load pairs                                  #
################################################################################

xo_stats = np.load('results/31D_experimental/190807_summstats_prep845_082_0044.npz')['summ_stats']
obs = xo_stats


################################################################################
#                               Define ss and g                                #
################################################################################

# Using the 'PrinzStats' here, defined in summstats.py. Inferring the summstats
# only between t_burnin and t_burnin+t_window. There are 15 stats.
s = netio.create_summstats(params)

# create the generator object
g = Default(model=mlist[0], prior=prior_norm, summary=s)


################################################################################
#                                Run inference                                 #
################################################################################

from delfi.inference import APT

# create inference object
inf_SNPE = APT(generator=g,
               prior_norm=False,
               n_hiddens=params.n_hiddens,
               pilot_samples=pilot_data,
               obs=obs,
               verbose=False,
               density=params.density_type,
               seed=1)
print('---- Successfully created a SNPE-C object ----')

start_time = time.time()
log, train_data, posteriors = inf_SNPE.run(proposal='prior',
                                           n_train=params.n_train,
                                           n_rounds=params.n_rounds,
                                           epochs=params.n_epochs,
                                           silent_fail=False,
                                           train_on_all=True,
                                           reuse_prior_samples=True,
                                           val_frac=0.1,
                                           verbose=True)
print("Estimated time", time.time()-start_time, "seconds")


################################################################################
#                                  Save data                                   #
################################################################################

inf_SNPE.trn_datasets = []
# with open('results/31D_nets/191001_seed1_Exper11deg.pkl', 'wb') as file:
#     pickle.dump([inf_SNPE, log, params], file)
print('Success')
