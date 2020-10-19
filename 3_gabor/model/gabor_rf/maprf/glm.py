from abc import abstractmethod, ABCMeta

from theano import tensor as tt
from theano.tensor import as_tensor_variable
from theano.tensor.shared_randomstreams import RandomStreams


# TODO: fix the code for the Bernoulli GLM

# TODO: implement Gaussian GLM


class GLM(metaclass=ABCMeta):

	def params(self):
		return self.bias, self.binsize

	def __init__(self, seed=None):
		self._random = RandomStreams(seed)
		self.bias = tt.scalar('bias')
		self.binsize = tt.scalar('binsize')

	@abstractmethod
	def invlink(self, x):
		pass

	@abstractmethod
	def prediction(self, z):
		r = self.invlink(z + self.bias)
		return as_tensor_variable(r, 'rate')

	@abstractmethod
	def generation(self, z):
		pass

	@abstractmethod
	def likelihood(self, z, y):
		pass


class Poisson(GLM):

	def invlink(self, x):
		return tt.exp(x)

	def prediction(self, z):
		return super().prediction(z)

	def generation(self, r, rng=None):
		if rng is None:
			rng = self._random
		λ = r * self.binsize
		# generate spikes
		y = rng.poisson(λ.shape, lam=λ, ndim=λ.ndim)
		return as_tensor_variable(y, name='y')

	def likelihood(self, z, y):
		η = z.flatten(min(2, z.ndim)) + self.bias
		Δ = self.binsize
		# 1st part of the likelihood
		L1 = tt.dot(y, η)
		if z.ndim > 1:
			ndim = z.ndim - 1
			shp_z = z.shape[-ndim:]
			L1 = L1.reshape(shp_z, ndim=ndim)
		# 2nd part of the likelihood
		λ = self.invlink(z + self.bias)
		L2 = Δ * tt.sum(λ, axis=0)
		# constant factors
		c1 = tt.sum(y) * tt.log(Δ)
		c2 = -tt.sum(tt.where(y > 1, tt.gammaln(y + 1), 0.0))
		const = c1 - c2

		L = L1 - L2 + const
		return as_tensor_variable(L, name='logL')

	def likelihood_no_bias(self, z, y, alpha, beta):

		#c1 = tt.gammaln(1 + tt.sum(y))
		#c2 = tt.sum(tt.gammaln(1 + y))
		#const = c1 - c2

		η = z.flatten(min(2, z.ndim))
		L1 = tt.dot(y, η)
		if z.ndim > 1:
			ndim = z.ndim - 1
			shp_z = z.shape[-ndim:]
			L1 = L1.reshape(shp_z, ndim=ndim)

		z_max = tt.max(z, axis=0)
		z_max_ = tt.shape_padleft(z_max, 1)
		log_sum_exp = z_max + tt.log(tt.sum(tt.exp(z - z_max_)) + beta * tt.exp(- z_max_))
		L2 = (alpha + tt.sum(y)) * log_sum_exp

		return L1 - L2 #+ const

	# def marg_log_lik(self, model, data):
	# 	backup = model.filter.conv
	# 	model.filter.conv = True
	# 	r = self.prediction(data['x'], model)
	# 	L = self.likelihood(r, data['y'])
	# 	model.filter.conv = backup
	# 	return L


def robust_expit(x):
	def expit_p(z):
		return 1 / (1 + tt.exp(-z))

	def expit_n(z):
		exp_z = tt.exp(z)
		return exp_z / (1 + exp_z)
	return tt.where(x > 0, expit_p, expit_n)


def expit(x):
	return 1 / (1 + tt.exp(-x))


class Bernoulli(GLM):

	invlink = expit

	def likelihood(self, y):
		η = self.nat_rate
		if η.ndim == 1:
			L1 = y.dot(η)
		else:
			shp_z = η.shape[-2:]
			L1 = y.dot(η.flatten(2))
			L1 = L1.reshape(shp_z, ndim=2)
		L = L1 - tt.sum(tt.log1p(tt.exp(η)), axis=0)
		return as_tensor_variable(L, name='logL')

	def generation(self, x):
		p = self.prediction(x) * self.binsize
		return self._random.binomial(p.shape, p=p, ndim=p.ndim)

	def prediction(self, x):
		r = super().prediction(x) / self.binsize
		return as_tensor_variable(r, name='r')


# def identity(x): return x
#
#
# class Gaussian(GLM):
#
# 	invlink = identity
#
# 	def __init__(self, sd, *args, **kwargs):
# 		super().__init__(*args, **kwargs)
# 		self.sd = sd
#
# 	def likelihood(self, x, y):
# 		super().likelihood(x, y)
#
# 	def generation(self, x):
# 		super().generation(x)
#
# 	def prediction(self, x):
# 		return super().prediction(x)

