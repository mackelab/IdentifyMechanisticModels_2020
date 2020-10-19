import delfi.distribution as dd
import numpy as np
import collections
import matplotlib.pyplot as plt

import delfi.generator as dg
import delfi.distribution as dd

from model.gabor_rf import maprf as model
from model.gabor_stats import maprfStats

import theano
import theano.tensor as tt

from maprf.utils import empty
from maprf.inference import Inference, GaborSampler, CenterSampler
from maprf.rfs.v1 import SimpleLinear_full_kt
from maprf.glm import Poisson

def setup_sim(seed, path, no_transform=False):

    sim_info = np.load(path +'results/sim_info.npy', allow_pickle=True)[()]
    d, params_ls = sim_info['d'], sim_info['params_ls']

    m = model(filter_shape= np.array((d,d,2)),
              parametrization=sim_info['parametrization'],
              params_ls=params_ls,
              seed=seed,
              dt=sim_info['dt'],
              duration=sim_info['duration'],
              no_transform=no_transform)

    p, prior = get_maprf_prior_01(params_ls, seed, no_transform=no_transform)

    s = maprfStats(n_summary=d*d+1) # summary stats (d x d RF + spike_count)


    def rej(x):
        # rejects summary statistic if number of spikes == 0
        return x[:,-1] > 0

    # generator object that auto-rejects some data-pairs (theta_i, x_i) right at sampling
    g = dg.RejKernel(model=m, prior=p, summary=s, rej=rej, seed=seed)

    return g, prior, d

def get_maprf_prior_01(params_ls, seed=None, no_transform=False):
    ## prior over simulation parameters
    prior = collections.OrderedDict()

    if no_transform:
        lims = np.array([[-1.5, -1.1, .001,         0.01,          .001, 0.01, 0.01, -.999, -.999],
                         [ 1.5,  1.1, .999*np.pi, 2.49,   1.999*np.pi, 1.99, 3.99, .999,   .999]]).T
        p = dd.Uniform(lower=lims[:,0], upper=lims[:,1])
        return p, prior

    ## prior over simulation parameters
    if 'bias' in params_ls['glm']:
        prior['bias'] = {'mu' : np.array([-0.57]), 'sigma' : np.array([np.sqrt(1.63)]) }
    if 'gain' in params_ls['kernel']['s']:
#        prior['A'] = {'mu' : np.zeros(1), 'sigma' : 2 * np.ones(1) }
        prior['log_A'] = {'mu' : np.zeros(1), 'sigma' : np.ones(1) / 2 }  
    if 'phase' in params_ls['kernel']['s']:
        prior['logit_φ']  = {'mu' : np.array([0]), 'sigma' : np.array([1.9]) }
    if 'freq' in params_ls['kernel']['s']:
        prior['log_f']  = {'mu' : np.zeros(1), 'sigma' : np.ones(1) / 2 }
    if 'angle' in params_ls['kernel']['s']:
        prior['logit_θ']  = {'mu' : np.zeros(1), 'sigma' :   np.array([1.78]) }
    if 'ratio' in params_ls['kernel']['s']:
        prior['log_γ']  = {'mu' : np.zeros(1), 'sigma' : np.ones(1) / 2}
    if 'width' in params_ls['kernel']['s']:
        prior['log_b']  = {'mu' : np.zeros(1), 'sigma' : np.ones(1) / 2}
    if 'xo' in params_ls['kernel']['l']:
        prior['logit_xo'] = {'mu' : np.array([0.]), 'sigma' : np.array([1.78]) }
        #prior['xo'] = {'mu' : np.array([0.]), 'sigma' : np.array([1./np.sqrt(4)]) }
    if 'yo' in params_ls['kernel']['l']:
        prior['logit_yo'] = {'mu' : np.array([0.]), 'sigma' : np.array([1.78]) }
        #prior['yo'] = {'mu' : np.array([0.]), 'sigma' : np.array([1./np.sqrt(4)]) }
    L = np.diag(np.concatenate([prior[i]['sigma'] for i in list(prior.keys())]))
    if 'value' in params_ls['kernel']['t']:
        ax_t = m.dt * np.arange(1,len_kt+1)
        Λ =  np.diag(ax_t / 0.075 * np.exp(1 - ax_t / 0.075))
        D = np.eye(ax_t.shape[0]) - np.eye(ax_t.shape[0], k=-1)
        F = np.dot(D, D.T)
        Σ = np.dot(Λ, np.linalg.inv(F).dot(Λ))
        prior['kt'] = {'mu': np.zeros_like(ax_t), 'sigma': np.linalg.inv(D).dot(Λ)}
        L = np.block([[L, np.zeros((L.shape[0], ax_t.size))],
                      [np.zeros((ax_t.size, L.shape[1])), prior['kt']['sigma']]])
    mu  = np.concatenate([prior[i][ 'mu'  ] for i in prior.keys()])
    p = dd.Gaussian(m=mu, S=L.T.dot(L), seed=seed)

    return p, prior


def setup_sampler(prior, obs, d, g, params_dict,
                  fix_position=True, parametrization='logit_φ'):

    # generative model
    rf = SimpleLinear_full_kt()
    emt = Poisson()

    # inputs and outputs
    data = [theano.shared(empty(3), 'frames'),
            theano.shared(empty(1, dtype='int64'))]
    frames, spikes = data

    # fill the grids
    grid_x, grid_y = np.meshgrid(g.model.axis_x, g.model.axis_y)

    rf.grids['s'][0].set_value(grid_x)
    rf.grids['s'][1].set_value(grid_y)
    rf.grids['t'][0].set_value(g.model._gen.axis_t)

    # inference model
    inference = Inference(rf, emt, bias=params_dict['glm']['bias'])
    inference.priors = {
        'glm': {         'bias':  {'name':  'gamma',
                                   'varname': 'λo',
                                   'alpha': 1.0,    #prior['λo']['alpha'][0],
                                   'beta':  1.0}},  #prior['λo']['beta'][0]}},
        'kernel': {'s': {'ratio':  {'name': 'normal',
                                    'varname': 'log_γ',
                                    'sigma': prior['log_γ']['sigma'][0],
                                    'mu':    prior['log_γ']['mu'][0]},
                         'width':  {'name': 'normal',
                                    'varname': 'log_b',
                                    'sigma': prior['log_b']['sigma'][0],
                                    'mu':    prior['log_b']['mu'][0]},
#                         'gain':   {'name': 'normal',
#                                    'varname': 'A',
#                                    'mu':    prior['A']['mu'][0],
#                                    'sigma': prior['A']['sigma'][0]},
                         'gain':   {'name': 'lognormal',
                                    'varname': 'log_A',
                                    'mu':    prior['log_A']['mu'][0],
                                    'sigma': prior['log_A']['sigma'][0]},                         
                         'phase':  {'name': 'logitnormal',
                                    'varname': 'logit_φ',
                                    'mu':    prior['logit_φ']['mu'][0],
                                    'sigma': prior['logit_φ']['sigma'][0]},
                         'freq':   {'name': 'lognormal',
                                    'varname': 'log_f',
                                    'mu':    prior['log_f']['mu'][0],
                                    'sigma': prior['log_f']['sigma'][0]},
                         'angle':  {'name': 'logitnormal',
                                    'varname': 'logit_θ',
                                    'mu':    prior['logit_θ']['mu'][0],
                                    'sigma': prior['logit_θ']['sigma'][0]}},
                   'l': { 'xo' : {'name' : 'logitnormal',
                                  'varname' : 'logit_xo',
                                  'sigma': prior['logit_xo']['sigma'][0],
                                  'mu':    prior['logit_xo']['mu'][0]},
                          'yo' : {'name' : 'logitnormal',
                                  'varname' : 'logit_yo',
                                  'sigma': prior['logit_yo']['sigma'][0],
                                  'mu':    prior['logit_yo']['mu'][0]}}
                         } }

    if 'kt' in prior.keys():
        inference.priors['kernel']['t'] = prior['kt']

    inference.add_sampler(
        GaborSampler(fix_position=fix_position,
                     parametrization=parametrization)
    )
    #inference.add_sampler(
    #    CenterSampler(model=inference,
    #                 center=inference.samplers[0].center)
    #)

    print(inference.samplers[0].params)

    # temporal kernel (here [1,0])
    kt = tt.scalar('kt') #tt.vector('kt')
    # ensure normalization (to fix firing rate) :
    inference.rf.filter.kernel['t'] = 1. * kt #/ tt.sqrt(tt.dot(kt, kt))
    inference.add_inputs(kt)

    print('inputs: ', inference.inputs)
    print('priors: ', inference.priors)

    inference.build(data)
    inference.compile()

    loglik = inference.loglik
    # set MCMC chain initializer
    if fix_position:
        loglik['logit_xo'] = 0.
        loglik['logit_yo'] = 0.
    else:
        kl =  params_dict['kernel']['l']
        loglik['logit_xo'] = np.log( (1+kl['xo']) / (  1. - kl['xo']))
        loglik['logit_yo'] = np.log( (1+kl['yo']) / (  1. - kl['yo']))

    loglik['kt'] = params_dict['kernel']['t']['value']#.copy()

    ks = params_dict['kernel']['s']
    loglik['log_γ'] = np.log(ks['ratio'])
    loglik['log_b'] = np.log(ks['width'])
    #loglik['A']   = 1. * ks['gain']
    loglik['log_A']   = np.log(ks['gain'])    
    loglik['logit_φ'] =  np.log(ks['phase'] / (  np.pi - ks['phase']))
    loglik['log_f'] = np.log(ks['freq'])
    loglik['logit_θ'] = np.log(ks['angle'] / (2*np.pi - ks['angle']))

    # hand over data
    frames.set_value(obs['I'][:,:].reshape(-1,d,d))
    spikes.set_value(obs['data'][:])

    # use this instead for sampling from the prior:
    #frames.set_value(0*obs['I'][:1,:].reshape(-1,d,d))
    #spikes.set_value(0*obs['data'][:1])

    return inference, data


def symmetrize_sample_modes(samples):

    assert samples.ndim==2 and samples.shape[1] == 9

    # assumes phase in [0, pi]
    assert np.min(samples[:,2]) >= 0. and np.max(samples[:,2] <= np.pi)
    # assumes angle in [0, 2*pi]
    assert np.min(samples[:,4]) >= 0. and np.max(samples[:,4] <= 2*np.pi)
    # assumes freq, ratio and width > 0
    assert np.all(np.min(samples[:,np.array([3,5,6])], axis=0) >= 0.)

    samples1 = samples.copy()
    idx = np.where( samples[:,4] > np.pi )[0]
    samples1[idx,4] = samples1[idx,4] - np.pi
    idx = np.where( samples[:,4] < np.pi )[0]
    samples1[idx,4] = samples1[idx,4] + np.pi
    samples1[:,2] = np.pi - samples1[:,2]
    samples_all = np.vstack((samples, samples1))[::2, :]

    samples1 = samples_all.copy()
    samples1[:,1] = - samples1[:,1]
    samples1[:,2] = np.pi - samples1[:,2]
    samples_all = np.vstack((samples_all, samples1))[::2, :]

    return samples_all



## convenience functions



def get_data_o(filename, g, seed):

    m, s = g.model, g.summary

    params_dict_true = np.load(filename, allow_pickle=True)[()]
    params_dict_true['kernel']['t']['value'] = np.cast[theano.config.floatX](1.)
    m.rng = np.random.RandomState(seed=seed)
    m.params_dict = params_dict_true.copy()
    pars_true = m.read_params_buffer()

    return s.calc([m.gen_single()]), pars_true




## visiualization helpers



def quick_plot(g, obs_stats, d, pars_true, posterior, log=None):

    m, p = g.model, g.prior

    # bunch of example prior draws
    plt.figure(figsize=(16,6))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(m.params_to_rf(p.gen().reshape(-1))[0], interpolation='None')
    plt.subplot(2,5,3)
    plt.title('RF prior draws')
    plt.show()

    plt.figure(figsize=(16,5))
    plt.subplot(1,5,1)
    plt.imshow(m.params_to_rf(p.mean)[0], interpolation='None')
    plt.title('prior mean RF')
    plt.subplot(1,5,2)
    plt.imshow(obs_stats[0,:-1].reshape(d,d), interpolation='None')
    plt.title('data STA')
    plt.subplot(1,5,3)
    plt.imshow(m.params_to_rf(pars_true)[0], interpolation='None')
    plt.title('ground-truth RF')
    plt.subplot(1,5,4)
    plt.imshow(m.params_to_rf(posterior.calc_mean_and_cov()[0])[0], interpolation='None')
    plt.title('posterior mean RF')
    plt.subplot(1,5,5)
    a_max = np.argmax(posterior.a)
    plt.imshow(m.params_to_rf(posterior.xs[a_max].m)[0], interpolation='None')
    plt.title('posterior mode RF')
    plt.show()

    # bunch of example posterior draws
    plt.figure(figsize=(16,6))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(m.params_to_rf(posterior.gen().reshape(-1))[0], interpolation='None')
    plt.subplot(2,5,3)
    plt.title('RF posterior draws')
    plt.show()

    if isinstance(log,list) and len(log) >0:
        plt.subplot(1,2,1)
        plt.semilogx(log[-1]['loss'][100:])
        plt.subplot(1,2,2)
        plt.plot(log[-1]['loss'][100:])
        plt.show()

def contour_draws(p, g, obs_stats, d, n_draws=10, lvls=[0.5, 0.5]):

    plt.figure(figsize=(6,6))
    plt.imshow(obs_stats[0,:-1].reshape(d,d), interpolation='None', cmap='gray')
    for i in range(n_draws):
        rfm = g.model.params_to_rf(p.gen().reshape(-1))[0]
        plt.contour(rfm, levels=[lvls[0]*rfm.min(), lvls[1]*rfm.max()])
        #print(rfm.min(), rfm.max())
        #plt.hold(True)
    plt.title('RF posterior draws')
    plt.show()
