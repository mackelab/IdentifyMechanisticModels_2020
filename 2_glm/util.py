import delfi.distribution as dd
import numpy as np

from model.GLM import GLM
from model.GLMStats import GLMStats
from math import factorial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pypolyagamma import PyPolyaGamma
from tqdm import tqdm


def obs_params(len_filter=9, a=1.):
    """Parameters for x_o

    Parameters
    ----------
    len_filter : float
        length of GLM temporal filter 
    a : float        
        inverse time constant of the filter

    Returns
    -------
    true_params : array
        (b0, h) = offset, temporal filters
    labels_params : list of str
    """
    b0 = -2.
    tau = np.linspace(1, len_filter, len_filter)  # support for the filter
    h = (a * tau)**3 * np.exp(-a * tau)  # temporal filter

    true_params = np.concatenate((np.array([b0]),h))
    labels_params = ['b0']
    for i in range(len_filter):
        labels_params.append('h'+str(i+1))

    return true_params, labels_params

def obs_data(params, seed=None, seed_input=None, duration = 100):
    """Data for x_o
    """
    m = GLM(len_filter=len(params)-1, seed=seed, seed_input=seed_input,
        duration=duration)
    return m.gen_single(params)

def obs_stats(params, seed=None, seed_input=None, duration = 100):
    """Summary stats for x_o
    """
    m = GLM(len_filter=len(params)-1, seed=seed, seed_input=seed_input,
        duration=duration)
    data = m.gen_single(params)
    s = GLMStats(n_summary=m.n_params)
    return s.calc([data])


def smoothing_prior(n_params=10, seed=None):
    """Prior"""
    M = n_params-1

    # Smoothing prior on h; N(0, 1) on b0. Smoothness encouraged by
    # penalyzing 2nd order differences of filter elements
    D = np.diag(np.ones(M)) - np.diag(np.ones(M-1), -1)
    F = np.dot(D, D) + np.diag(1.0 * np.arange(M)/(M))**0.5  # Binv is block diagonal
    Binv = np.zeros(shape=(M+1,M+1))
    Binv[0,0] = 0.5    # offset (b0)
    Binv[1:,1:] = np.dot(F.T, F) # filter (h)

    # Prior params
    prior_mn = np.zeros((n_params, ))
    prior_prec = Binv

    return dd.Gaussian(m=prior_mn, P=prior_prec, seed=seed)

def smoothing_prior_highVar(n_params=10, seed=None):
    """Prior"""
    M = n_params-1

    # prior
    # smoothing prior on h; N(0, 1) on b0. Smoothness encouraged by penalyzing
    # 2nd order differences of elements of filter
    D = np.diag(np.ones(M)) - np.diag(np.ones(M-1), -1)
    F = np.dot(D, D)

    # Binv is block diagonal
    Binv = np.zeros(shape=(M+1, M+1))
    Binv[0,0] = 1  # offset (b0)
    Binv[1:,1:] = np.dot(F.T, F)  # filter (h)

    # Prior params
    prior_mn = np.zeros((n_params, ))
    prior_prec = Binv

    return dd.Gaussian(m=prior_mn, P=prior_prec, seed=seed)    

def pg_mcmc(true_params, obs, duration=100, dt=1, seed=None,
    prior_dist=None):
    """Polya-Gamma sampler for GLM

    Returns
    -------
    array : samples from posterior
    """

    if prior_dist is None:
        prior_dist = smoothing_prior(n_params=true_params.size, seed=seed)

    # seeding
    np.random.seed(seed)
    pg = PyPolyaGamma()  # seed=seed

    # observation
    I = obs['I'].reshape(1,-1)
    S_obs = obs['data'].reshape(-1)

    # simulation protocol
    num_param_inf = len(true_params)
    dt = 1
    t = np.arange(0, duration, dt)

    N = 1   # Number of trials
    M = num_param_inf-1   # Length of the filter

    # build covariate matrix X, such that X * h returns convolution of x with filter h
    X = np.zeros(shape=(len(t), M))
    for j in range(M):
        X[j:,j] = I[0,0:len(t)-j]

    # prior
    # smoothing prior on h; N(0, 1) on b0. Smoothness encouraged by penalyzing
    # 2nd order differences of elements of filter
    #prior_dist = prior(n_params=true_params.size, seed=seed)
    Binv = prior_dist.P

    # The sampler consists of two iterative Gibbs updates
    # 1) sample auxiliary variables: w ~ PG(N, psi)
    # 2) sample parameters: beta ~ N(m, V); V = inv(X'O X + Binv), m = V*(X'k), k = y - N/2
    nsamp = 500000   # samples to evaluate the posterior

    # add a column of 1s to the covariate matrix X, in order to model the offset too
    X = np.concatenate((np.ones(shape=(len(t), 1)), X), axis=1)

    beta = true_params*1.
    BETA = np.zeros((M+1,nsamp))

    for j in tqdm(range(1, nsamp)):
        psi = np.dot(X, beta)
        w = np.array([pg.pgdraw(N, b) for b in psi])
        O = np.diag(w)

        V = np.linalg.inv(np.dot(np.dot(X.T, O), X) + Binv)
        m = np.dot(V, np.dot(X.T, S_obs - N * 0.5))

        beta = np.random.multivariate_normal(np.ravel(m), V)

        BETA[:,j] = beta

    # burn-in
    burn_in = 100000
    BETA_sub_samp = BETA[:, burn_in:nsamp:30]

    # return sampling results
    return BETA_sub_samp
