import numpy as np
import sys
sys.path.append("model/setup")
import netio
#from delfi.utils.utils_prinzetal import inv_logistic_fct, logistic_fct

def load_trn_data(filedir, params):

    # loading data. Files in the archive are 'params' and 'stats'
    data = np.load(filedir)  # samples_dir = results/samples/

    # 7 parameters: Na+ current, CaT current (T-type Calcium, low-threshold), CaS current, A current (transient potassium current), KCa current, Kd current, H current (hyperpolarization current)
    sample_params = data["params"]  # there are 7 parameters in the network
    sample_stats = data["stats"]  # there are 15 summary_stats in 'PrinzStats' (see the params variable above).
    # These 15 stats can be seen in summstats.py. They are: cycle_period, burst_length*3, end_to_start*2, start_to_end*2, duty_cycle*3, phase_gap*2, phase*2

    prior = netio.create_prior(params, log=True)
    sample_params = (sample_params - prior.lower) / (prior.upper - prior.lower)

    sample_params = inv_logistic_fct(np.asarray(sample_params))

    # normalize data
    params_mean = np.mean(sample_params, axis=0)
    params_std  = np.std(sample_params, axis=0)
    sample_params = (sample_params - params_mean) / params_std

    # extract number of training samples
    sample_params_pilot = sample_params[:params.pilot_samples]
    sample_stats_pilot = sample_stats[:params.pilot_samples]
    sample_params_train = sample_params[params.pilot_samples:params.pilot_samples + params.n_train]
    sample_stats_train = sample_stats[params.pilot_samples:params.pilot_samples + params.n_train]

    pilot_data = (sample_params_pilot, sample_stats_pilot)
    trn_data = [sample_params_train, sample_stats_train]  # taking log of conductances to get the training data

    return pilot_data, trn_data, params_mean, params_std



from scipy.special import expit, logit
import delfi.distribution as dd

def load_trn_data_newTF(filedir, params):

    # loading data. Files in the archive are 'params' and 'stats'
    data = np.load(filedir)  # samples_dir = results/samples/

    # 7 parameters: Na+ current, CaT current (T-type Calcium, low-threshold), CaS current, A current (transient potassium current), KCa current, Kd current, H current (hyperpolarization current)
    sample_params = data["params"]  # there are 7 parameters in the network
    sample_stats = data["stats"]  # there are 15 summary_stats in 'PrinzStats' (see the params variable above).
    # These 15 stats can be seen in summstats.py. They are: cycle_period, burst_length*3, end_to_start*2, start_to_end*2, duty_cycle*3, phase_gap*2, phase*2

    prior = netio.create_prior(params, log=True)

    lower = np.asarray(prior.lower)
    upper = np.asarray(prior.upper)
    inputscale = lambda x: (x - lower) / (upper - lower)
    bijection = lambda x: logit(inputscale(x))  # logit function with scaled input

    sample_params = bijection(sample_params)

    # normalize data
    params_mean = np.mean(sample_params, axis=0)
    params_std  = np.std(sample_params, axis=0)
    sample_params = (sample_params - params_mean) / params_std

    # extract number of training samples
    sample_params_pilot = sample_params[:params.pilot_samples]
    sample_stats_pilot = sample_stats[:params.pilot_samples]
    sample_params_train = sample_params[params.pilot_samples:params.pilot_samples + params.n_train]
    sample_stats_train = sample_stats[params.pilot_samples:params.pilot_samples + params.n_train]

    pilot_data = (sample_params_pilot, sample_stats_pilot)
    trn_data = [sample_params_train, sample_stats_train]  # taking log of conductances to get the training data

    return pilot_data, trn_data, params_mean, params_std


def load_trn_data_normalize(filedir, params):

    # loading data. Files in the archive are 'params' and 'stats'
    data = np.load(filedir)  # samples_dir = results/samples/

    # 7 parameters: Na+ current, CaT current (T-type Calcium, low-threshold), CaS current, A current (transient potassium current), KCa current, Kd current, H current (hyperpolarization current)
    sample_params = data["params"]  # there are 7 parameters in the network
    sample_stats = data["stats"]  # there are 15 summary_stats in 'PrinzStats' (see the params variable above).
    # These 15 stats can be seen in summstats.py. They are: cycle_period, burst_length*3, end_to_start*2, start_to_end*2, duty_cycle*3, phase_gap*2, phase*2

    prior = netio.create_prior(params, log=True)

    # normalize data
    params_mean = prior.mean
    params_std = prior.std
    sample_params = (sample_params - params_mean) / params_std

    # extract number of training samples
    sample_params_pilot = sample_params[:params.pilot_samples]
    sample_stats_pilot = sample_stats[:params.pilot_samples]
    sample_params_train = sample_params[params.pilot_samples:params.pilot_samples + params.n_train]
    sample_stats_train = sample_stats[params.pilot_samples:params.pilot_samples + params.n_train]

    pilot_data = (sample_params_pilot, sample_stats_pilot)
    trn_data = [sample_params_train, sample_stats_train]  # taking log of conductances to get the training data

    return pilot_data, trn_data, params_mean, params_std


def forward_tf(cond_params, prior=None, params_mean=None, params_std=None, steps='111'):
    if steps[0] == '1': cond_params = (cond_params - prior.lower) / (prior.upper - prior.lower)
    if steps[1] == '1': cond_params = inv_logistic_fct(np.asarray(cond_params))
    if steps[2] == '1': cond_params = (cond_params - params_mean) / params_std
    return cond_params

def forward_tf_newTF(cond_params, prior=None):
    lower = np.asarray(prior.lower)
    upper = np.asarray(prior.upper)
    inputscale = lambda x: (x - lower) / (upper - lower)
    bijection = lambda x: logit(inputscale(x))  # logit function with scaled input
    cond_params = bijection(cond_params)

    return cond_params




def load_pair(filedir, **kwargs):
    data_xo = np.load(filedir)
    xo_params1 = data_xo["params1"]
    xo_stats1 = data_xo["summstats1"]
    xo_params1 = forward_tf(xo_params1, **kwargs)

    xo_params2 = data_xo["params2"]
    xo_stats2 = data_xo["summstats2"]
    xo_params2 = forward_tf(xo_params2, **kwargs)

    return xo_params1, xo_stats1, xo_params2, xo_stats2

def load_pair_newTF(filedir, **kwargs):
    data_xo = np.load(filedir)
    xo_params1 = data_xo["params1"]
    xo_stats1 = data_xo["summstats1"]
    xo_params1 = forward_tf_newTF(xo_params1, **kwargs)

    xo_params2 = data_xo["params2"]
    xo_stats2 = data_xo["summstats2"]
    xo_params2 = forward_tf_newTF(xo_params2, **kwargs)

    return xo_params1, xo_stats1, xo_params2, xo_stats2

def load_pair_normalize(filedir, prior):
    params_mean = prior.mean
    params_std = prior.std

    data_xo = np.load(filedir)
    xo_params1 = data_xo["params1"]
    xo_stats1 = data_xo["summstats1"]
    xo_params1 = (xo_params1 - params_mean) / params_std

    xo_params2 = data_xo["params2"]
    xo_stats2 = data_xo["summstats2"]
    xo_params2 = (xo_params2 - params_mean) / params_std

    return xo_params1, xo_stats1, xo_params2, xo_stats2


def load_single_sample_normalize(filedir, prior, log_synapses=False):
    params_mean = prior.mean
    params_std = prior.std
    data_xo = np.load(filedir)
    xo_params1 = data_xo["params1"]
    xo_stats1 = data_xo["summstats1"]
    if log_synapses:
        xo_params1[-17:-10] = np.log(xo_params1[-17:-10])
    xo_params1 = (xo_params1 - params_mean) / params_std

    return xo_params1, xo_stats1




def load_single_sample(filedir, **kwargs):
    data_xo = np.load(filedir)
    xo_params1 = data_xo["params1"]
    xo_stats1 = data_xo["summstats1"]
    xo_params1 = forward_tf(xo_params1, **kwargs)

    return xo_params1, xo_stats1


def save_samples(sample_params, sample_stats, sample_seed, index_fast, save_samples, case='fast'):
    counter = 0
    for inspect_num in range(len(save_samples)):
        if save_samples[inspect_num]:
            outfile_pair = '../../results/observations/31D/' + case + '_sample_{}.npz'.format(counter)
            np.savez_compressed(outfile_pair, index1=int(index_fast[inspect_num]),
                                params1=sample_params[int(index_fast[inspect_num])],
                                summstats1=sample_stats[int(index_fast[inspect_num])],
                                seed1=sample_seed[int(index_fast[inspect_num])])
            counter += 1