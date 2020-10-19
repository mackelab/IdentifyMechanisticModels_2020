import numpy as np
import scipy.signal as ss
from scipy.stats import multivariate_normal as mvn

from delfi.simulator.BaseSimulator import BaseSimulator

import maprf.rfs.v1 as v1
import maprf.glm as glm
from maprf.generation import Generator
import theano


def cast_params(dct):
    output = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            output[key] = cast_params(value)
        else:
            output[key] = np.cast[theano.config.floatX](value)
    return output


class maprf(BaseSimulator):


    def __init__(self,
        duration=10,
        filter_shape=np.array((1,1,1)),
        dt=0.025,
        parametrization='full',
        params_ls=None,
        no_transform=False,
        seed=None):
        """GLM simulator

        Parameters
        ----------
        duration : int
            Duration of traces in ms
        filter_shape : (Nx, Ny, T)
            Size of spatiotemporal filter
        dt: float
            Length of temporal bins
        parametrization : string
            Parametrization of linear filter
        params_ls: dict
            Hierarchical dictionary with strings of variable names to include
        seed : int or None
            If set, randomness across runs is disabled
        """
        super(maprf, self).__init__(dim_param=np.prod(filter_shape).astype(np.int)+1,
                                    seed=seed)

        self.duration = duration
        self.filter_shape = filter_shape
        self.len_filter  = filter_shape[-1]
        self.size_filter = np.prod(filter_shape[:-1]).astype(np.int)
        self.parametrization = parametrization
        self.no_transform = no_transform

        # parameters that globally govern the simulations
        self.dt = dt
        self.t = np.arange(0, self.duration, self.dt)

        # input: gaussian white noise N(0, 1)
        self.rng_input = np.random.RandomState(seed=self.gen_newseed())
        self.I = np.cast[theano.config.floatX](self.rng_input.randn(len(self.t),self.size_filter))

        if parametrization=='full':

            self.params_dict = {'glm': {'binsize': self.dt,
                                        'bias': 0.},
                          'kernel': {'value': np.zeros(filter_shape)}}

            if params_ls is None:
                params_ls = {'glm': ('bias',),
                          'kernel':  ('value',)}
            else:
                self.params_ls = params_ls


            n_glm = len(self.params_ls['glm'])
            n_kst = np.prod(filter_shape) if 'value' in self.params_ls['kernel'] else 0
            self.n_param_div = (n_glm, n_kst)
            self.n_params = np.sum(self.n_param_div)

        elif  parametrization=='gaussian':

            self.params_dict = {'glm': {'binsize': self.dt,
                                        'bias': 0.},
                           'kernel': {'s': {'mx': 0.,
                                            'my': 0.,
                                            'sx': 0.,
                                            'sy': 0.,
                                            'c': 0.},
                                      't': {'value': np.zeros(self.len_filter)}}}

            if params_ls is None:
                params_ls = {'glm': ('bias',),
                          'kernel': {'s' : ('mx','my','sx','sy', 'c'),
                                     't' : ('value',)}}
            else:
                self.params_ls = params_ls

            self.n_param_div = (n_glm, n_ks, n_kt)
            self.n_params = np.sum(self.n_param_div)

        elif  parametrization=='gabor':

            kernel = v1.SimpleLinear_full_kt() # space-time seperable filter
            spikes = glm.Poisson()

            # initial values for all mapRF parameters
            self.params_dict = {'glm': {'binsize': self.dt,
                                        'bias': 0.},
                          'kernel': {'s': {'angle': 0.,
                                           'freq': 1.,
                                           'gain': 1.,
                                           'phase': 0.,
                                           'ratio': 1.,
                                           'width': 1.},
                                     'l': {'xo': 0.,
                                           'yo': 0.},
                                     't': {'value': 1.}}} #np.zeros(self.len_filter)}}}
            self.params_dict = cast_params(self.params_dict)

            # lookup-table for parameter dimensionalities
            self.par_lengths = {'glm': {'bias': 1,
                                        'binsize': 1},
                            'kernel': {'s': {'vec_A': 2,
                                           'vec_f': 2,
                                           'ratio': 1,
                                           'width': 1,
                                           'angle': 1,
                                           'phase': 1,
                                           'freq': 1,
                                           'gain': 1},
                                       'l': {'xo': 1,
                                            'yo': 1},
                                       't': {'tau': 1,
                                             'value': 1}}} #self.len_filter}}}

            # re-parametrization lookup table:
            self.reparams_ls = {'glm': {'bias': ('bias',)},
                                'kernel': {'s': {'vec_f': ('angle', 'freq'),
                                                 'vec_A': ('gain', 'phase'),
                                                 'angle': ('angle',),
                                                 'freq': ('freq',),
                                                 'phase': ('phase',),
                                                 'gain': ('gain',),
                                                 'ratio': ('ratio',),
                                                 'width': ('width',)},
                                           'l': {'xo' : ('xo',),
                                                 'yo' : ('yo',)},
                                           't': {'tau': ('tau',),
                                                 'value': ('value',)}}}

            # list of parameters varied for simulation
            if params_ls is None:
                params_ls = {
                 'glm': ('bias',),
                 'kernel': {'s' : ('vec_A', 'vec_f',
                                   'ratio','width'),
                            'l' : ('xo', 'yo'),
                            't' : ('value',)}}
            self.params_ls = params_ls

            self.str_to_incides()

            self._gen = Generator(kernel, spikes, seed=seed)

            # maprf grids hard-assume grid-element volumes dA=0.25
            #d = self.filter_shape[0]
            #axis_x = np.arange(-(d-1)/4., (d-1)/4.+1e-10, 0.5)
            #d = self.filter_shape[1]
            #axis_y = np.arange(-(d-1)/4., (d-1)/4.+1e-10, 0.5)
            self.axis_x = np.linspace(-1, 1, self.filter_shape[0])
            self.axis_y = np.linspace(-1, 1, self.filter_shape[1])

            self._gen.grid_x, self._gen.grid_y = np.meshgrid(self.axis_x,
                                                             self.axis_y)

            self._gen.axis_t = self.dt * np.arange(1,self.len_filter+1)

            self._gen.build()

            self._gen.x = self.I.reshape(len(self.t),
                                         self.filter_shape[0],
                                         self.filter_shape[1]).copy()

            # functions for evaluating receptive field shapes
            self._eval_kt = theano.function(list(self._gen.inputs.values()),
                                           self._gen.rf.kt,
                                           on_unused_input='ignore',
                                           allow_input_downcast=True)
            self._eval_ks = theano.function(list(self._gen.inputs.values()),
                                           self._gen.rf.ks,
                                           on_unused_input='ignore',
                                           allow_input_downcast=True)


        else:
            raise NotImplementedError

    def gen_single(self, params=None, params_ls=None):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of the forward run. Additional entries can be present.
        """

        if self.parametrization in ('full', 'gaussian'):

            assert self.len_filter == 1 # temporal filtering not implemented yet!
            assert self.filter_shape[2] == self.len_filter

            assert params.ndim == 1, 'params.ndim must be 1'

            params = np.asarray(params).reshape(-1)

            assert params.shape[0] == self.n_params, 'params.shape[0] must be dim params long'


            b0 = params[0]
            b0.astype(float)
            h = params[1:] if self.parametrization=='full' else self.params_to_rf(params[1:])
            h.astype(float)

            # simulation
            psi = b0 + np.dot(self.I, h) #ss.lfilter(h, 1, self.I, axis=0).sum(axis=1)

            # psi goes through a sigmoid non-linearity: firing probability
            z = 1 /(1 + np.exp(-psi)).reshape(-1,1)

            # sample the spikes
            N = 1   # number of trials
            y = self.rng.uniform(size=(len(self.t), N)) < z
            y = np.sum(y, axis=1)

        elif self.parametrization == 'gabor':

            if not params is None:

                assert params.ndim == 1, 'params.ndim must be 1'
                params = np.asarray(params).reshape(-1)
                assert params.shape[0] == self.n_params, 'params.shape[0] must be dim params long'
                self._set_pars_dict(params, params_ls)

            self._gen.set_params(self.params_dict)

            axis_x = self.axis_x - self.params_dict['kernel']['l']['xo']
            axis_y = self.axis_y - self.params_dict['kernel']['l']['yo']

            self._gen.grid_x, self._gen.grid_y = np.meshgrid(axis_x,
                                                             axis_y)
            self._gen._sample()
            y = self._gen.y.copy()

        return {'data': y, 'I': self.I, 'shape': (y.shape[0], *self.filter_shape[:-1])}

    def reparam_p2m(self, x):

        x = np.atleast_2d(x)
        params = x.copy()

        if self.no_transform:
            if params.shape[0]==1:
                params = params.reshape(-1)

            return params
        
        #if 'bias' in self.params_idx['glm'].keys():
        #    i = self.params_idx['glm']['bias']
        #    params[:,i] = np.log(x[:,i])

        ks = self.params_idx['kernel']['s']
        kl = self.params_idx['kernel']['l']

        if 'vec_f' in ks.keys():
            ij, ri, rj = ks['vec_f'], ks['angle'], ks['freq']
            u = self.cart2pol(x[:,ij])
            params[:,ri], params[:,rj] = u[:,0], u[:,1]

        if 'freq' in ks.keys() and not 'vec_f' in ks.keys():
            i = ks['freq']
            params[:,i] = np.exp(x[:,i])

        if 'angle' in ks.keys():
            i = ks['angle']
            params[:,i] = 2*np.pi / (1. + np.exp(-x[:,i]))

        if 'vec_A' in ks.keys():
            ij, ri, rj = ks['vec_A'], ks['gain'], ks['phase']
            u = self.cart2pol(x[:,ij])
            params[:,ri], params[:,rj] = u[:,0], u[:,1]

        if 'gain' in ks.keys():
            i = ks['gain']
            params[:,i] = np.exp(x[:,i])

        if 'phase' in ks.keys() and not 'vec_A' in ks.keys():
            i = ks['phase']
            params[:,i] = np.pi / (1. + np.exp(-x[:,i]))

        if 'ratio' in ks.keys():
            i = ks['ratio']
            params[:,i] = np.exp(x[:,i])

        if 'width' in ks.keys():
            i = ks['width']
            params[:,i] = np.exp(x[:,i])

        if 'xo' in kl.keys():
            i = kl['xo']
            params[:,i] = 2. / (1. + np.exp(-x[:,i])) - 1

        if 'yo' in kl.keys():
            i = kl['yo']
            params[:,i] = 2. / (1. + np.exp(-x[:,i])) - 1

        if params.shape[0]==1:
            params = params.reshape(-1)

        return params


    def reparam_m2p(self, x):

        x = np.atleast_2d(x)
        params = x.copy()

        if self.no_transform:
            return params
        
        #if 'bias' in self.params_idx['glm'].keys():
        #    i = self.params_idx['glm']['bias']
        #    params[:,i] = np.exp(x[:,i])

        ks = self.params_idx['kernel']['s']
        kl = self.params_idx['kernel']['l']

        if 'vec_f' in ks.keys():
            ij, ri, rj = ks['vec_f'], ks['angle'], ks['freq']
            params[:,ij] = self.pol2cart((x[:,(ri, rj)]))

        if 'freq' in ks.keys() and not 'vec_f' in ks.keys():
            i = ks['freq']
            params[:,i] = np.log(x[:,i])

        if 'angle' in ks.keys():
            i = ks['angle']
            params[:,i] =  np.log(x[:,i] / (2*np.pi - x[:,i]))

        if 'vec_A' in ks.keys():
            ij, ri, rj = ks['vec_A'], ks['gain'], ks['phase']
            params[:,ij] = self.pol2cart((x[:,(ri,rj)]))

        if 'gain' in ks.keys():
            i = ks['gain']
            params[:,i] = np.log(x[:,i])

        if 'phase' in ks.keys() and not 'vec_A' in ks.keys():
            i = ks['phase']
            params[:,i] =  np.log(x[:,i] / (np.pi - x[:,i]))

        if 'ratio' in ks.keys():
            i = ks['ratio']
            params[:,i] = np.log(x[:,i])

        if 'width' in ks.keys():
            i = ks['width']
            params[:,i] = np.log(x[:,i])

        if 'xo' in kl.keys():
            i = kl['xo']
            params[:,i] = np.log((1+x[:,i]) / (1. - x[:,i]))
        if 'yo' in kl.keys():
            i = kl['yo']
            params[:,i] = np.log((1+x[:,i]) / (1. - x[:,i]))


        if params.shape[0]==1:
            params = params.reshape(-1)

        return params

    @staticmethod
    def pol2cart(v):
        return v[:,[0]] * np.hstack([np.cos(v[:,[1]]), np.sin(v[:,[1]])])

    @staticmethod
    def cart2pol(v):
        u1 = np.sqrt(v[:,[0]]**2 + v[:,[1]]**2)
        u2 = np.arctan2(v[:,[1]], v[:,[0]])
        return np.hstack([u1, u2])


    def params_to_rf(self, params, params_ls=None):


        if self.parametrization=='full':

            h = params

        elif self.parametrization=='gaussian':

            c = params[4] * params[2] * params[3] # corr to cov
            C = np.array([[params[2]**2,c],[c,params[3]**2]])
            dx, dy = self.filter_shape[0]//2, self.filter_shape[1]//2
            assert self.filter_shape[0]>2*dx and self.filter_shape[1]>2*dy

            dx, dy = range(-dx,dx+1), range(-dy,dy+1)
            h = mvn.pdf(np.mgrid[dx, dy].reshape(2,-1).T,
                        mean=params[:2],
                        cov=C + 1e-5*np.eye(2))
            h -= h.mean()
            std = h.std()
            if std>0:
                h /= std

        elif self.parametrization=='gabor':

            self._set_pars_dict(params, params_ls)
            self.params = cast_params(self.params_dict)

            axis_x = self.axis_x - self.params_dict['kernel']['l']['xo']
            axis_y = self.axis_y - self.params_dict['kernel']['l']['yo']

            self._gen.grid_x, self._gen.grid_y = np.meshgrid(axis_x,
                                                             axis_y)


            ks = self._eval_ks(bias=np.cast[theano.config.floatX](self.params_dict['glm']['bias']),
                         angle=np.cast[theano.config.floatX](self.params_dict['kernel']['s']['angle']),
                         freq=np.cast[theano.config.floatX](self.params_dict['kernel']['s']['freq']),
                         gain=np.cast[theano.config.floatX](self.params_dict['kernel']['s']['gain']),
                         phase=np.cast[theano.config.floatX](self.params_dict['kernel']['s']['phase']),
                         ratio=np.cast[theano.config.floatX](self.params_dict['kernel']['s']['ratio']),
                         width=np.cast[theano.config.floatX](self.params_dict['kernel']['s']['width']))
            kt = self._eval_kt(value=np.cast[theano.config.floatX](self.params_dict['kernel']['t']['value']))

            h = (ks,kt)

        return h

    def str_to_incides(self):

        idx, j = {'glm': {}, 'kernel': {'s': {}, 'l': {}, 't': {}}}, 0

        for key in self.params_ls['glm']:
            j_ = j
            for rkey in self.reparams_ls['glm'][key]:
                dj = self.par_lengths['glm'][rkey]
                idx['glm'][rkey] = self._idx(j, dj)
                j += dj
            idx['glm'][key] = self._idx(j_,j-j_)

        ks = ('s', 'l', 't')
        for k in range(len(ks)):
            for key in self.params_ls['kernel'][ks[k]]:
                j_ = j
                for rkey in self.reparams_ls['kernel'][ks[k]][key]:
                    dj = self.par_lengths['kernel'][ks[k]][rkey]
                    idx['kernel'][ks[k]][rkey] = self._idx(j,dj)
                    j += dj
                idx['kernel'][ks[k]][key] = self._idx(j_,j-j_)
        self.n_params = j
        self.params_idx = idx

    @staticmethod
    def _idx(j, dj):
        if dj > 1:
            return np.arange(j, j+dj)
        elif dj == 1:
            return j
        else:
            return []

    def _set_pars_dict(self, params, params_ls=None):

        params_ls = self.params_ls if params_ls is None else params_ls

        reparams = self.reparam_p2m(params)

        glm = self.params_dict['glm']
        for key in params_ls['glm']:
            for rkey in self.reparams_ls['glm'][key]:
                glm[rkey] = reparams[self.params_idx['glm'][rkey]]

        kernels = self.params_dict['kernel']
        for k in ('s', 'l', 't'):
            for key in params_ls['kernel'][k]:
                for rkey in self.reparams_ls['kernel'][k][key]:
                    kernels[k][rkey] = reparams[self.params_idx['kernel'][k][rkey]]

        self.params_dict = cast_params(self.params_dict)
                    
                    
    def read_params_buffer(self):

        params = np.zeros(self.n_params)

        for key in self.params_ls['glm']:
            for rkey in self.reparams_ls['glm'][key]:
                params[self.params_idx['glm'][rkey]] = self.params_dict['glm'][rkey]

        for k in ('s', 'l', 't'):
            for key in self.params_ls['kernel'][k]:
                for rkey in self.reparams_ls['kernel'][k][key]:
                    params[self.params_idx['kernel'][k][rkey]] = self.params_dict['kernel'][k][rkey]

        return self.reparam_m2p(params)
