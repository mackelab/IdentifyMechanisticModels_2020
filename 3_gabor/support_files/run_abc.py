import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
import delfi.utils.io as io
import delfi.summarystats as ds
#from delfi.kernel.Kernel_learning import kernel_opt
import numpy as np
import os
import scipy.misc
import sys

def run_smc(model, prior, summary, obs_stats, 
        seed=None,n_particles=1e3,eps_init=2,maxsim=5e7, 
        fn=None, AW=False, OW=False):
    """Runs Sequential Monte Carlo ABC algorithm.
    Adapted from epsilonfree code https://raw.githubusercontent.com/gpapamak/epsilon_free_inference/8c237acdb2749f3a340919bf40014e0922821b86/demos/mg1_queue_demo/mg1_abc.py

    Parameters
    ----------
    model : 
         Model
    prior :
         Prior
    summary :
         Function to compute summary statistics
    obs_stats: 
         Observed summary statistics
    n_params : 
         Number of parameters
    seed : int or None
        If set, randomness in sampling is disabled
    n_particles : int
        Number of particles for SMC-ABC
    eps_init : Float
        Initial tolerance for SMC-ABC
    maxsim : int
        Maximum number of simulations for SMC-ABC
    """    
    set_folders()

    n_params, n_stats = get_in_out_dims(model, prior, summary)
    prefix = str(n_params)+'params'
    n_particles = int(n_particles)
    maxsim = int(maxsim)
    
    #####################
    np.random.seed(seed)

    # set parameters
    ess_min = 0.0
    eps_lvls, eps_last = set_eps_lvls(eps_init)
    all_ps, all_xs, all_logweights, all_eps, all_nsims = [], [], [], [], []

    # sample initial population
    eps = eps_lvls[0]
    ps, xs, logweights, nsims = sample_lvl_init(model, prior, summary, 
                                    calc_dist, obs_stats, n_particles, 
                                    n_params, n_stats, eps)
    remsims = maxsim - nsims
    weights = np.exp(logweights)
    all_ps.append(ps)
    all_xs.append(xs)
    all_logweights.append(logweights)
    all_eps.append(eps)
    all_nsims.append(nsims)

    iter = 0
    print('iteration = {0}, eps = {1:.2}, ess = {2:.2%}, nsims = {3}'.format(iter, float(eps), 1.0, nsims))

    while eps > eps_last:

        # calculate kernel bandwidth
        bw_x  = set_kernel_bandwidths(data=xs, weights=weights, 
                                      obs_stats=obs_stats, rule='of_thumb') if AW else None

        iter += 1
        eps = eps_lvls[iter]

        ps, xs, logweights, nsims, break_flag, bw_th = sample_lvl(model, prior, summary, 
                                        calc_dist, obs_stats, ps, xs, logweights, 
                                        eps, remsims, bw_x, OW)
   
        weights = np.exp(logweights)
        remsims -= nsims

        if break_flag:
            break

        # calculate effective sample size
        ess = 1.0 / (np.sum(weights ** 2) * n_particles)
        print('iteration = {0}, eps = {1:.2}, ess = {2:.2%}, nsims = {3}'.format(iter, float(eps), ess, nsims))
        if ess < ess_min:
            ps, xs, logweights = resample(ps, xs, weights)
            weights = np.exp(logweights)

        all_ps.append(ps)
        all_xs.append(xs)
        all_logweights.append(logweights)
        all_eps.append(eps_lvls[iter])
        all_nsims.append(nsims)

        if not fn is None:
            np.save(fn, 
            {'seed' : seed, 
             'all_ps' : all_ps,
             'all_logweights' : all_logweights, 
             'all_eps' : all_eps, 
             'all_nsims' : all_nsims,
             'model' : model,
             'prior' : prior,
             'summary' : summary,
             'obs_stats' : obs_stats,
             'n_particles' : n_particles,
             'maxsim' : maxsim,
             'eps_init' : eps_init})

    return all_ps, all_xs, all_logweights, all_eps, all_nsims

        
def calc_dist(stats_1, stats_2):
    """Euclidian distance between summary statistics"""
    return np.sqrt(np.sum((stats_1 - stats_2) ** 2))

def set_folders():
    # check for subfolders, create if they don't exist
    dirs = {}
    dirs['dir_abc'] = './results/abc/'
    for k, v in dirs.items():
        if not os.path.exists(v):
            os.makedirs(v)

def get_in_out_dims(model, prior, summary):
    p = prior.gen(n_samples=1)[0]
    x = summary.calc([ model.gen_single(p) ])
    return p.size, x.size

def set_eps_lvls(eps_init, eps_last=0.01, eps_decay=0.9):
    if isinstance(eps_init, list):
        eps_last = eps_init[-1]
    else: 
        eps, eps_init = eps_init, []
        while eps > eps_last:
            eps *= eps_decay
            eps_init.append(eps)
    return eps_init, eps_last

def sample_lvl_init(model, prior, summary, calc_dist, obs_stats,
                    n_particles, n_params, n_stats, eps):
    
    ps, xs = np.empty([n_particles, n_params]), np.empty([n_particles, n_stats])
    logweights = np.zeros(n_particles)- np.log(n_particles)
    nsims = 0
    for i in range(n_particles):

        dist = float('inf')
        while dist > eps:

            ps[i] = prior.gen(n_samples=1)[0]
            states = model.gen_single(ps[i])
            xs[i] = summary.calc([states])
            dist = calc_dist(xs[i], obs_stats)
            nsims += 1

    return ps, xs, logweights, nsims

def sample_lvl(model, prior, summary, calc_dist, obs_stats, 
               ps, xs, logweights, eps, remsims, bw_x, OW):

    n_particles, n_params = ps.shape

    # perturb particles
    new_ps, new_xs = np.empty_like(ps), np.empty_like(xs)
    new_logweights = np.empty_like(logweights)

    if not bw_x is None: 
        # adapt log-weights with kernels ( w -> v in Bonassi & West !)
        print('using AW')
        logweights = logweights - 0.5 * np.sum(np.linalg.solve(bw_x, (obs_stats - xs).T) ** 2, axis=0)
        logweights = logweights - scipy.misc.logsumexp(logweights)
    weights = np.exp(logweights)

    bw_th = set_kernel_bandwidths(data=ps, weights=weights, 
                                  obs_stats=obs_stats, rule='of_thumb')

    nsims, break_flag = 0, False
    for i in range(n_particles):

        dist = float('inf')

        while dist > eps:
            idx = discrete_sample(weights)[0]

            new_ps[i] = ps[idx] + np.dot(bw_th, np.random.randn(n_params))
            if isinstance(prior, dd.Uniform):
                while np.any(new_ps[i] < prior.lower) or \
                   np.any(new_ps[i] > prior.upper):                      
                    new_ps[i] = ps[idx] + np.dot(bw_th, np.random.randn(n_params))

            states = model.gen_single(new_ps[i])
            new_xs[i] = summary.calc([states])
            dist = calc_dist(new_xs[i], obs_stats)
            nsims += 1

            if nsims>=remsims:
                print('Maximum number of simulations reached.')
                break_flag = True
                break

        # k_{th,t}(theta^(t)_j | theta^{t-1}_j)
        logkernel = -0.5 * np.sum(np.linalg.solve(bw_th, (new_ps[i] - ps).T) ** 2, axis=0)
        new_logweights[i] = prior.eval(new_ps[i, np.newaxis], log=True)[0] - scipy.misc.logsumexp(logweights + logkernel)

        if break_flag:
            break

    new_logweights = new_logweights - scipy.misc.logsumexp(new_logweights)

    return new_ps, new_xs, new_logweights, nsims, break_flag, bw_th

def resample(ps, xs, weights):
    # resample particles
    new_ps, new_xs = np.empty_like(ps), np.empty_like(xs)
    for i in range(new_ps.shape[0]):
        idx = discrete_sample(weights)[0]
        new_ps[i] = ps[idx]
        new_xs[i] = xs[idx]
    logweights = np.zeros_like(weights) - np.log(weights.size)

    return new_ps, new_xs, logweights

def set_kernel_bandwidths(data, weights, obs_stats, rule='of_thumb',
                          data2=None, dist=None, eps=None):

    n_particles = weights.size
    assert n_particles == data.shape[0]
    d = 2 * data.shape[1] # assumes #parameters = #summary stats !

    if rule == 'of_thumb':
        # see West (1993) and Scott & Sain (2005)
        cov = np.diag(np.diag(np.atleast_2d(np.cov(data.T, aweights=weights))))
        std = np.linalg.cholesky(cov)
        return std * n_particles**( - 1./(d+4.))

    if rule == 'norm_comp_opt':
        # optimal component-wise Gaussian kernels
        # see Filippi et al. (2012)

        idx0 = np.where(dist(data2, obs_stats) < eps)[0]
        w = weights[idx0] / weights[idx0].sum()

        wd = np.sqrt(w).dot(data[idx0,:].reshape(idx0.size,1,-1) - data.reshape(1,n_particles,-1))
        sig2s = w.dot(weights.dot(()**2))

        return np.diag(np.sqrt(sig2s))

    if rule == 'norm_full_opt':
        # optimal full Gaussian kernels
        # see Filippi et al. (2012)

        idx0 = np.where(dist(data2, obs_stats) < eps)[0]
        w = weights[idx0] / weights[idx0].sum()

        sii = np.einsum('ij,ik,i->jk', data[idx0], data[idx0], w)
        sjj = np.einsum('ij,ik,i->jk', data, data, weights) 
        sij = np.einsum('ij,kl,i,k->jl', data, data[idx0], weights, w)

        return np.linalg.cholesky(sii + sjj - sij - sij.T)

    if rule == 'norm_OLCM':
        # optimal local covariance matrix kernels
        # see Filippi et al. (2012)

        #idx0 = np.where(dist(data2, obs_stats) < eps)[0]
        #w = weights[idx0] / weights[idx0].sum()
        #bw = np.empty((n_particles, d, d))
        #for i in range(n_particles):
        #    tmp = data[idx0] - data[i]
        #    bw[i] = np.einsum('ij,ik,i->jk', tmp, tmp, w) 
        # return bw # bw.shape = (n_particles, d, d) !


        raise NotImplementedError


    elif rule == 'None' or rule is None:
        return None

    elif rule == 'x_kl':

        raise NotImplementedError

        """
        print('fitting kernel')
        
        cbkrnl, cbl = kernel_opt(
            iws=weights.astype(np.float32), 
            stats=data.astype(np.float32),
            obs=obs_stats.astype(np.float32), 
            kernel_loss='x_kl', 
            epochs=1000,  
            minibatch=n_particles//10,
            stop_on_nan=True,
            seed=99, 
            monitor=None)

        cbkrnl.B += 1e-10 * np.ones_like(cbkrnl.B)
        #print('cbkrnl.B', cbkrnl.B)

        return np.linalg.inv(np.diag(cbkrnl.B))
        """
        
    else: 
        raise NotImplementedError

def discrete_sample(p, n_samples=1):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples
    :return: vector of samples
    """

    # check distribution
    #assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    r = np.random.rand(n_samples, 1)
    return np.sum((r > c).astype(int), axis=1)
