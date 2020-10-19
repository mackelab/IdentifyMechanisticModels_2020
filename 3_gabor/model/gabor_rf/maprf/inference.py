import theano
from theano import In
from tqdm import tqdm

from maprf.sampling.slice import EllipticalSliceSampling as ESS
from maprf.utils import *
from collections import OrderedDict
from numpy.core.umath import pi as π


class Inference:
	samplers = None

	def __init__(self, rf, emt, bias=None):
		self.rf = rf
		flt = rf.filter
		self.filter = flt
		self.emt = emt
		self.samplers = []
		self.updates = OrderedDict()

		self.inputs = {}
		self.buffer = {}

		self.bias_ = bias
		# for p in emt.params():
		# 	self.add_inputs(p)

	def compile(self):
		i = list(self.inputs.values())
		o = self.logL

		u = self.updates

		self.loglik = theano.function(i, o, updates=u, on_unused_input='warn', allow_input_downcast=True)

		buffer = self.filter.filt_s.buffer
		update = self.updates[buffer]
		o_cpy = theano.clone(self.logL, replace={update: buffer})
		self.loglik2 = theano.function(i, o_cpy, on_unused_input='ignore', allow_input_downcast=True)

	def build(self, data):

		try:
			prior = self.prior['glm']['bias']
			αo, βo = prior['alpha'], prior['beta']
		except:
			αo, βo = 1., 0.

		x = tt.as_tensor_variable(data[0], 'x')
		y = tt.as_tensor_variable(data[1], 'y')
		η = self.filter(x, self.updates)
		L = self.emt.likelihood_no_bias(η, y, αo, βo)

		self.logL = L

	def sample(self, nsamp):
		traces = {v: [] for v in self.buffer}
		L = np.zeros(nsamp)
		for i in tqdm(range(nsamp)):
			for sampler in self.samplers:
				sampler.a_step()
			for v, T in traces.items():
				x = self.buffer[v].get_value()
				T.append(x)
			L[i] = self.loglik()
		traces = {v: np.array(t) for v, t in traces.items()}
		return traces, L

	def sample_biases(self, data, traces, dt):

		# Gamma prior parameters
		alpha = self.priors['glm']['bias']['alpha']
		beta = self.priors['glm']['bias']['beta']

		# could try to grad the following also from the existing graph?
		x = tt.as_tensor_variable(data[0], 'x')
		y = tt.as_tensor_variable(data[1], 'y')
		η = self.filter(x, self.updates)

		α = theano.function([], tt.sum(y) + alpha,
		                    on_unused_input='warn',
							allow_input_downcast=True)

		# get binsize without adding it to self.inputs
		Δ = theano.shared(empty(self.emt.binsize.ndim), self.emt.binsize.name)
		in_Δ = In(self.emt.binsize, value=Δ.container, implicit=False)
		Δ.set_value(dt)

		i = list(self.inputs.values()) + [in_Δ]
		β = theano.function(i, self.emt.binsize * tt.sum(tt.exp(η)) + beta,
		                    on_unused_input='warn', allow_input_downcast=True)

		# sample exp(bias) = λo given all other parameters and the data
		nsamp = list(traces.values())[0].shape[0]
		traces['λo'] = np.zeros(nsamp)
		for i in tqdm(range(nsamp)):
		    for v in self.inputs:
		        self.buffer[v].set_value(traces[self.buffer[v].name][i])
		    traces['λo'][i] = np.random.gamma(shape = α(),
		                                      scale = 1./β())

		traces['bias'] = np.log(traces['λo'])

		return traces

	def add_sampler(self, sampler):
		if sampler not in self.samplers:
			self.samplers.append(sampler)
			sampler.attach(self)

	def add_buffer(self, p):
		self.buffer[p] = theano.shared(np.cast[tt.config.floatX](empty(p.ndim)), p.name)

	def add_inputs(self, p):
		try:
			b = self.buffer[p]
		except KeyError:
			self.add_buffer(p)
			b = self.buffer[p]

		self.inputs[p] = In(p, value=b.container, implicit=False)

	def pop_inputs(self, p, rm_buffer=False):
		self.inputs.pop(p)
		if rm_buffer:
			self.buffer.pop(p)

	def add_update(self, var, expr):
		self.buffer[var] = b = theano.shared(empty(var.ndim))
		self.updates[b] = expr


class GaborSampler(ESS):

	def attach(self, model):
		rf = model.rf
		kernel = rf.kernel['s']

		#v1, v2, v3, v4 = [T(x) for T, x in zip(self.forward, self.params)]


		vs = [T(x) for T, x in zip(self.forward, self.params+self.center)]

		if self.parametrization == 'vec_A':
			givens = {'ratio': vs[0], 'width': vs[1], 'freq': vs[2][0],
					  'angle': vs[2][1], 'gain': vs[3][0], 'phase': vs[3][1]}
		elif self.parametrization == 'logit_φ':
			givens = {'ratio': vs[0], 'width': vs[1], 'freq': vs[2],
					  'angle': vs[3], 'gain': vs[4], 'phase': vs[5]}

		center = {} if self.fix_position else {'xo' : vs[6], 'yo' : vs[7]}

		for name, var in givens.items():
			var.name = name
		for name, var in center.items():
			var.name = name
		for name, var in zip( ('logit_xo', 'logit_yo'), self.center ):
			var.name = name


		rf.center = [center['xo'], center['yo']]
		for name, var in givens.items():
			setattr(kernel, name, var)
		rf.update('s')

		# add the parameters of the gabor to the inputs
		for p in self.vars+self.center:
			model.add_inputs(p)
		for p in kernel.params():
			model.add_update(p, p)
		#for p in self.center:
		#	model.add_update(p, p)

		self.model = model
		self.kernel = kernel
		self.priors = model.priors['kernel']['s']

		self._set_priors()


	def __init__(self, model=None, fix_position=False, parametrization='vec_A'):

		if parametrization == 'vec_A':
			self.forward = [tt.exp, tt.exp, cart2pol, cart2pol]
			self.params = [tt.scalar('log_γ'), tt.scalar('log_b'),
						   tt.vector('vec_f'), tt.vector('vec_A')]
		elif parametrization == 'logit_φ':
			self.forward = [tt.exp, tt.exp,
							tt.exp, scaled_expit_fc,
#							identity, scaled_expit_hc]
							tt.exp, scaled_expit_hc]                            
			if not fix_position:
				self.forward = self.forward + [scaled_expit_i,scaled_expit_i]

			self.params = [tt.scalar('log_γ'), tt.scalar('log_b'),
					   	   tt.scalar('log_f'), tt.scalar('logit_θ'),
#					   	   tt.scalar('A'), tt.scalar('logit_φ')]
					   	   tt.scalar('log_A'), tt.scalar('logit_φ')]                           
		else:
			raise NotImplemented()

		#if not fix_position:
		#	self.forward = self.forward + [identity, identity]

		self.parametrization = parametrization
		self.center = [tt.scalar('logit_xo'), tt.scalar('logit_yo')]
		self.fix_position = fix_position
		vars = self.params if fix_position else self.params+self.center
		self.model = model
		self.kernel = None
		self.priors = None
		super().__init__(self.loglik, vars)

	def loglik(self, **kwargs):
		L = self.model.loglik(**kwargs)

		if self.fix_position:

			return L

		else:

			px, py = 0., 0. # using Gaussian priors (rather than Generalized Gaussian) !

			#z = (kwargs['xo'] - self.mu['xo']) / self.priors['xo']['alpha']
			#px = 0.5 * (z ** 2) - np.abs(z) ** self.priors['xo']['gamma']

			#z = (kwargs['yo'] - self.mu['yo']) / self.priors['yo']['alpha']
			#py = 0.5 * (z ** 2) - np.abs(z) ** self.priors['yo']['gamma']

			return L + px + py

	def _set_priors(self):

		for var in self.priors.keys():
			self._set_prior(self.priors[var])


	def _set_prior(self, prior):

		if prior['name'] == 'Rayleigh':
			# bivariate Gauss, yields Rayleigh over norm, uniform over angle
			self.mu[prior['varname']] = np.zeros(2)
			self.sd[prior['varname']] = prior['sigma'] * np.eye(2)
		elif prior['name'] == 'Rice':
			# bivariate Gauss, yields Rice over norm, uniform over angle
			assert 'kappa' in prior.keys()
			self.mu[prior['varname']] = np.zeros(2)
			self.sd[prior['varname']] = prior['sigma'] * np.eye(2)
		elif prior['name'] == 'lognormal':
			# *uni*variate log normal
			if 'a' in prior.keys() and 'b' in prior.keys() and 'f' in prior.keys():
				m, s = normal_from_ci(*zip(prior['b'], prior['a']), prior['f'])
			else:
				m, s = prior['mu'], prior['sigma']
			self.mu[prior['varname']] = m
			self.sd[prior['varname']] = s
		elif prior['name'] == 'logitnormal':
			# *uni*variate logit normal
			self.mu[prior['varname']] = prior['mu']
			self.sd[prior['varname']] = prior['sigma']
		elif prior['name'] == 'gennormal':
			# *uni*variate generalized normal
			r = prior['range']
			prior['alpha'] = 0.5 * (r[1] - r[0])
			self.mu[prior['varname']] = 0.5 * (r[1] + r[0])
			self.sd[prior['varname']] = np.sqrt(0.5) * prior['alpha']
		elif prior['name'] == 'normal':
			# *uni*variate normal
			self.mu[prior['varname']] = prior['mu']
			self.sd[prior['varname']] = prior['sigma']
		else:
			raise NotImplemented()


	def get_point(self):
		buffer = self.model.buffer
		return {v.name: buffer[v].get_value() for v in self.vars+self.center}

	def a_step(self):
		vars = self.vars
		v = self.get_point()

		# update Rice distribution hyper-parameters
		for var in self.priors.keys():
			prior = self.priors[var]
			if prior['name']=='Rice':
				varname = prior['varname']
				x, y = [var for var in v[varname]]
				ν = np.sqrt(x ** 2 + y ** 2)
				φ = np.arctan2(y, x)
				κ = (prior['kappa'] * ν) / (prior['sigma'] ** 2)
				θ = nr.vonmises(φ, κ)  # sample a new value for the prior direction
				self.mu[varname] = prior['kappa'] * np.stack([np.cos(θ), np.sin(θ)])

		# return a sample from the posterior
		return super().a_step(**v)


class KernelSampler(ESS):

	@property
	def kernel(self):
		return self.vars[0]

	def attach(self, model):
		filter = model.rf.filter

		# and insert an input for the new kernel
		kt = self.kernel
		kt = kt / tt.sqrt(tt.dot(kt, kt))
		filter.kernel['t'] = kt
		model.add_update(kt, kt)

		for p in self.vars:
			model.add_inputs(p)
		self.model = model
		prior = model.priors['kernel']['t']
		self.mu['kt'] = prior['mu']
		self.sd['kt'] = prior['sigma']


	def __init__(self):
		vars = [tt.vector('kt')]
		super().__init__(self.loglik, vars)

	def loglik(self, **kwargs):
		return self.model.loglik2(**kwargs)

	def get_point(self):
		buffer = self.model.buffer
		return {v.name: buffer[v].get_value() for v in self.vars}

	def a_step(self):
		v = self.get_point()
		return super().a_step(**v)


class CenterSampler(ESS):
	def __init__(self, model, center):
		self.model = model
		self.pars = {v.name: v for v in center}
		vars = {v: [0.0, 3] for v in center}
		super().__init__(self.loglik, vars, π)

	def loglik(self, **kwargs):
		# noinspection PyTypeChecker
		kwargs = {v: np.fix(x * 2) / 2.0 for v, x in kwargs.items()}
		return self.model.loglik(**kwargs)

	def get_point(self):
		buffer = self.model.buffer
		return {n: buffer[v].get_value() for n, v in self.pars.items()}

	def a_step(self):
		v = self.get_point()
		return super().a_step(**v)
