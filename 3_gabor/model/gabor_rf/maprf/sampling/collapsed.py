import theano
from theano import In
from tqdm import tqdm

from maprf.sampling.slice import EllipticalSliceSampling as ESS
from maprf.utils import *
from collections import OrderedDict


class Inference:
	samplers = None

	def __init__(self, rf, emt):
		self.rf = rf
		flt = rf.filter
		self.filter = flt
		self.emt = emt
		self.samplers = []
		self.updates = OrderedDict()

		self.inputs = {}
		self.buffer = {}

	def compile(self):
		i = list(self.inputs.values())
		o = self.logL

		self._loglik = theano.function(i, o, updates=self.updates,
		                               on_unused_input='warn',
									   allow_input_downcast=True)

	def loglik(self, **kwargs):
		if len(kwargs) == 0:
			return self.bufferCurrLoglik.get_value()
		else:
			return self._loglik(**kwargs)

	def build(self, data):
		x, y = data
		η = self.filter(x, self.updates)
		L = self.emt.likelihood_no_bias(η, y)

		maxL = L.max()
		self.logL = tt.log(tt.sum(tt.exp(L - maxL))) + maxL

		self.bufferL = theano.shared(empty(2), 'L2D')
		self.bufferCurrLoglik = theano.shared(np.nan, 'curr_logL')

		self.buffer['L2D'] = self.bufferL
		self.updates[self.bufferL] = L
		self.updates[self.bufferCurrLoglik] = self.logL

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
		L2D = traces.pop('L2D')
		return traces, L2D, L

	def add_sampler(self, sampler):
		if sampler not in self.samplers:
			self.samplers.append(sampler)
			sampler.attach(self)

	def add_buffer(self, p):
		self.buffer[p] = theano.shared(empty(p.ndim), p.name)

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
	forward = [tt.exp, tt.exp, cart2pol, cart2pol]

	def attach(self, model):
		rf = model.rf
		kernel = rf.kernel['s']

		v1, v2, v3, v4 = [T(x) for T, x in zip(self.forward, self.params)]
		givens = {
			'ratio': v1, 'width': v2, 'freq': v3[0],
			'angle': v3[1], 'gain': v4[0], 'phase': v4[1]
		}
		for name, var in givens.items():
			var.name = name

		for name, var in givens.items():
			setattr(kernel, name, var)
		rf.update('s')
		# add the parameters of the gabor to the inputs
		for p in self.vars:
			model.add_inputs(p)
		for p in kernel.params():
			model.add_update(p, p)

		self.model = model
		self.kernel = kernel
		self.priors = model.priors['kernel']['s']
		# set up mean and covariances
		prior = self.priors['ratio']
		# p1, p2 = [(b, a) for b, a in ]
		m, s = normal_from_ci(*zip(prior['b'], prior['a']), prior['f'])
		self.mu['log_γ'] = m
		self.sd['log_γ'] = s

		prior = self.priors['width']
		m, s = normal_from_ci(*zip(prior['b'], prior['a']), prior['f'])
		self.mu['log_b'] = m
		self.sd['log_b'] = s

		self.mu['vec_A'] = np.zeros(2)
		self.sd['vec_A'] = self.priors['gain']['sigma'] * np.eye(2)

		self.mu['vec_f'] = np.nan * np.zeros(2)
		self.sd['vec_f'] = self.priors['freq']['sigma'] * np.eye(2)

	def __init__(self, model=None):
		self.params = [tt.scalar('log_γ'), tt.scalar('log_b'),
		               tt.vector('vec_f'), tt.vector('vec_A')]
		vars = self.params
		self.model = model
		self.kernel = None
		self.priors = None
		super().__init__(self.loglik, vars)

	def loglik(self, **kwargs):
		return self.model.loglik(**kwargs)

	def get_point(self):
		buffer = self.model.buffer
		return {v.name: buffer[v].get_value() for v in self.vars}

	def a_step(self):
		vars = self.vars
		v = self.get_point()

		# update frequency's hyper-parameters
		prior = self.priors['freq']
		x, y = [var for var in v['vec_f']]
		ν = np.sqrt(x ** 2 + y ** 2)
		φ = np.arctan2(y, x)
		κ = (prior['kappa'] * ν) / (prior['sigma'] ** 2)
		θ = nr.vonmises(φ, κ)  # sample a new value for the prior direction
		self.mu['vec_f'] = prior['kappa'] * np.stack([np.cos(θ), np.sin(θ)])

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
		return self.model.loglik(**kwargs)

	def get_point(self):
		buffer = self.model.buffer
		return {v.name: buffer[v].get_value() for v in self.vars}

	def a_step(self):
		v = self.get_point()
		return super().a_step(**v)
