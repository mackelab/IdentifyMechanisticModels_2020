import delfi.distribution as dd
import inspect
import numpy as np
import os
import pickle

from . import HodgkinHuxley as hh
from . import HodgkinHuxleyStatsMoments

from delfi.summarystats import Identity


def obs_params(reduced_model=False):
    """Parameters for x_o

    Parameters
    ----------
    reduced_model : bool
        If True, outputs two parameters
    Returns
    -------
    true_params : array
    labels_params : list of str
    """

    if reduced_model:
        true_params = np.array([50., 5.])
    else:
        true_params = np.array([50., 5., 0.1, 0.07, 6e2, 60., 0.1, 70.])

    labels_params = ['g_Na', 'g_K', 'g_leak', 'g_M',
                          't_max', '-V_T', 'noise','-E_leak']
    labels_params = labels_params[0:len(true_params)]

    return true_params, labels_params

def syn_current(duration=120, dt=0.01, t_on = 10, step_current=True,
                curr_level = 5e-4, seed=None):
    t_offset = 0.
    duration = duration
    t_off = duration - t_on
    t = np.arange(0, duration+dt, dt)

    # external current
    A_soma = np.pi*((70.*1e-4)**2)  # cm2
    I = np.zeros_like(t)
    I[int(np.round(t_on/dt)):int(np.round(t_off/dt))] = curr_level/A_soma # muA/cm2
    if step_current is False:
        rng_input = np.random.RandomState(seed=seed)

        times = np.linspace(0.0, duration, int(duration / dt) + 1)
        I_new = I*1.
        tau_n = 3.
        nois_mn = 0.2*I
        nois_fact = 2*I*np.sqrt(tau_n)
        for i in range(1, times.shape[0]):
            I_new[i] = I_new[i-1] + dt*(-I_new[i-1] + nois_mn[i-1] +
                        nois_fact[i-1]*rng_input.normal(0)/(dt**0.5))/tau_n
        I = I_new*1.

    return I, t_on, t_off, dt

def syn_obs_data(I, dt, params, V0=-70, seed=None, cython=False):
    """Data for x_o
    """
    m = hh.HodgkinHuxley(I=I, dt=dt, V0=V0, seed=seed, cython=cython)
    return m.gen_single(params)

def syn_obs_stats(I, params, dt, t_on, t_off, data=None, V0=-70, summary_stats=1, n_xcorr=5,
                  n_mom=5, n_summary=10, seed=None, cython=False):
    """Summary stats for x_o
    """

    if data is None:
        m = hh.HodgkinHuxley(I=I, dt=dt, V0=V0, seed=seed, cython=cython)
        data = m.gen_single(params)

    if summary_stats == 0:
        s = Identity()
    elif summary_stats == 1:
        s = HodgkinHuxleyStatsMoments(t_on, t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary)
    return s.calc([data])

def allen_obs_data(ephys_cell=464212183,sweep_number=33,A_soma=np.pi*(70.*1e-4)**2):
    """Data for x_o. Cell from AllenDB

    Parameters
    ----------
    ephys_cell : int
        Cell identity from AllenDB
    sweep_number : int
        Stimulus identity for cell ephys_cell from AllenDB
    """
    t_offset = 815.
    duration = 1450.
    dir_cache = os.path.dirname(inspect.getfile(hh.HodgkinHuxley))
    real_data_path = dir_cache + '/ephys_cell_{}_sweep_number_{}.pkl'.format(ephys_cell,sweep_number)
    if not os.path.isfile(real_data_path):
        from allensdk.core.cell_types_cache import CellTypesCache
        from allensdk.api.queries.cell_types_api import CellTypesApi

        manifest_file = 'cell_types/manifest.json'

        cta = CellTypesApi()
        ctc = CellTypesCache(manifest_file=manifest_file)
        data_set = ctc.get_ephys_data(ephys_cell)
        sweep_data = data_set.get_sweep(sweep_number)  # works with python2 and fails with python3
        sweeps = cta.get_ephys_sweeps(ephys_cell)

        sweep = sweeps[sweep_number]

        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][0:index_range[1]+1] # in A
        v = sweep_data["response"][0:index_range[1]+1] # in V
        sampling_rate = sweep_data["sampling_rate"] # in Hz
        dt = 1e3/sampling_rate # in ms
        i *= 1e6 # to muA
        v *= 1e3 # to mV
        v = v[int(t_offset/dt):int((t_offset+duration)/dt)]
        i = i[int(t_offset/dt):int((t_offset+duration)/dt)]


        real_data_obs = np.array(v).reshape(1, -1, 1)
        I_real_data = np.array(i).reshape(-1)
        t_on = int(sweep['stimulus_start_time']*sampling_rate)*dt-t_offset
        t_off = int( (sweep['stimulus_start_time']+sweep['stimulus_duration'])*sampling_rate )*dt-t_offset

        io.save((real_data_obs,I_real_data,dt,t_on,t_off), real_data_path)
    else:
        def pickle_load(file):
            """Loads data from file."""
            f = open(file, 'rb')
            data = pickle.load(f, encoding='latin1')
            f.close()
            return data
        real_data_obs,I_real_data,dt,t_on,t_off = pickle_load(real_data_path)

    t = np.arange(0, duration, dt)

    # external current
    I = I_real_data/A_soma # muA/cm2

    # return real_data_obs, I_obs
    return {'data': real_data_obs.reshape(-1),
            'time': t,
            'dt': dt,
            'I': I.reshape(-1),
            't_on': t_on,
            't_off': t_off}

def allen_obs_stats(data=None,ephys_cell=464212183, sweep_number=33, summary_stats=1,
                    n_xcorr=5, n_mom=5, n_summary=13):
    """Summary stats for x_o. Cell from AllenDB

    Parameters
    ----------
    ephys_cell : int
        Cell identity from AllenDB
    sweep_number : int
        Stimulus identity for cell ephys_cell from AllenDB
    """

    if data is None:
        data = allen_obs_data(ephys_cell=ephys_cell,sweep_number=sweep_number)

    t_on = data['t_on']
    t_off = data['t_off']

    if summary_stats == 0:
        s = Identity()
    elif summary_stats == 1:
        s = HodgkinHuxleyStatsMoments(t_on, t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary)
    return s.calc([data])

def resting_potential(data, dt, t_on):
    """Resting potential estimated from x_o
    """
    return np.mean(data[0:int(t_on/dt)-5])

def prior(true_params,prior_uniform=True,prior_extent=False,prior_log=False,seed=None):
    """Prior"""
    if not prior_extent:
        range_lower = param_transform(prior_log,0.5*true_params)
        range_upper = param_transform(prior_log,1.5*true_params)
    else:
        range_lower = param_transform(prior_log,np.array([.5,1e-4,1e-4,1e-4,50.,40.,1e-4,35.]))
        range_upper = param_transform(prior_log,np.array([80.,15.,.6,.6,3000.,90.,.15,100.]))

        range_lower = range_lower[0:len(true_params)]
        range_upper = range_upper[0:len(true_params)]

    if prior_uniform:
        prior_min = range_lower
        prior_max = range_upper
        return dd.Uniform(lower=prior_min, upper=prior_max,
                           seed=seed)
    else:
        prior_mn = param_transform(prior_log,true_params)
        prior_cov = np.diag((range_upper - range_lower)**2)/12
        return dd.Gaussian(m=prior_mn, S=prior_cov, seed=seed)

def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x

def param_invtransform(prior_log, x):
    if prior_log:
        return np.exp(x)
    else:
        return x
