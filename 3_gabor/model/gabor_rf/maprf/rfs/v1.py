from theano import shared as sv
from maprf.filters import SeparableRank1
from maprf.kernels import *
from ..utils import empty
import theano.tensor as tt


class ReceptiveFieldV1:

	kernel = None
	filter = None

	def __init__(self, ndim=2):
		super().__init__()
		self.grids = {
			self.kernel['s']: [
				sv(empty(ndim), 'grid_x', borrow=True),
				sv(empty(ndim), 'grid_y', borrow=True)],
			self.kernel['t']: [
				sv(empty(1), 'axis_t', borrow=True)]
		}
		self.grids['s'] = self.grids[self.kernel['s']]
		self.grids['t'] = self.grids[self.kernel['t']]

	def __call__(self, s):
		return self.filter(s)

	def predict(self, s):
		return self.filter(s)


class SimpleLinear(ReceptiveFieldV1):

	kernel = {'s': Gabor('cos', normalize=True),
	          't': AdelsonBergen(normalize=True)}

	@property
	def ks(self):
		return self.filter.kernel['s']

	@property
	def kt(self):
		return self.filter.kernel['t']

	def __init__(self):
		super().__init__(ndim=2)
		# initialize the filter
		self.filter = SeparableRank1(
				ks=self.kernel['s'](self),
				kt=self.kernel['t'](self))

	def update(self, keys=None):
		for k, kernel in self.kernel.items():
			if keys is None or k in keys:
				self.filter.kernel[k] = kernel(self)

	# model's parameters
	width = kernel['s'].width
	angle = kernel['s'].angle
	ratio = kernel['s'].ratio
	phase = kernel['s'].phase
	freq = kernel['s'].freq
	gain = kernel['s'].gain
	tau_t = kernel['t'].tau


class SimpleLinear_full_kt(ReceptiveFieldV1):

	kernel = {'s': Gabor('cos', normalize=True),
	          't': NonParametric(normalize=True)}

	@property
	def ks(self):
		return self.filter.kernel['s']

	@property
	def kt(self):
		return self.filter.kernel['t']

	def __init__(self):
		super().__init__(ndim=2)
		# initialize the filter
		self.filter = SeparableRank1(
				ks=self.kernel['s'](self),
				kt=self.kernel['t'](self))

	def update(self, keys=None):
		for k, kernel in self.kernel.items():
			if keys is None or k in keys:
				self.filter.kernel[k] = kernel(self)

	# model's parameters
	width = kernel['s'].width
	angle = kernel['s'].angle
	ratio = kernel['s'].ratio
	phase = kernel['s'].phase
	freq = kernel['s'].freq
	gain = kernel['s'].gain
	kt = kernel['t'].value

