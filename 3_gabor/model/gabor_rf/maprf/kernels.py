import theano.tensor as tt
import theano
import numpy as np
from numpy import pi as π, log2, sqrt
from abc import ABCMeta, abstractmethod
from collections import OrderedDict


__all__ = ['Gabor', 'AdelsonBergen', 'NonParametric']


class Kernel(metaclass=ABCMeta):

	def params(self):
		return tuple(self._param_dict.values())

	def __init__(self):
		self._param_dict = OrderedDict()

	def __call__(self, model):
		grids = self.grids(model)
		return self[grids]

	def grids(self, model):
		return model.grids[self]

	@abstractmethod
	def __getitem__(self, grids):
		raise NotImplementedError()


class Parameter:

	def __init__(self, name):
		self.name = name

	def __get__(self, instance, owner):
		params = instance._param_dict
		return params[self.name]

	def __set__(self, instance, value):
		params = instance._param_dict
		params[self.name] = value


class Gabor(Kernel):

	width = Parameter('b')
	ratio = Parameter('γ')
	angle = Parameter('θ')
	phase = Parameter('ϕ')
	freq = Parameter('ν')
	gain = Parameter('A')

	def __init__(self, mode='cos', normalize=False, exclude=None):
		super().__init__()
		self.normalize = normalize
		assert mode in ['cos', 'sin']
		self.mode = mode
		self.cache = dict(params={}, values={})
		# create parameters for this object
		if exclude is None:
			exclude = []
		if 'width' not in exclude:
			self.width = tt.scalar('width')
		if 'ratio' not in exclude:
			self.ratio = tt.scalar('ratio')
		if 'angle' not in exclude:
			self.angle = tt.scalar('angle')
		if 'phase' not in exclude:
			self.phase = tt.scalar('phase')
		self.freq = tt.scalar('freq')
		self.gain = tt.scalar('gain')

	@property
	def sigma(self):
		b = getattr(self, 'width', 1.0)
		a = 2 ** b
		k = getattr(self, 'freq') * (2 * π)
		return sqrt(2 * np.log(2)) * ((a + 1) / (a - 1)) / k

	def __getitem__(self, grids):
		x, y = self.rotate(grids)
		win = self.window(x, y)
		if self.normalize:
			ω = (2 * π) * self.freq
			γ = getattr(self, 'ratio', 1.0)
			σ = self.sigma
			v1 = 4 * π * (σ ** 2 / γ)
			v2 = tt.exp(-(ω * σ) ** 2)
			v2 = v2 * tt.cos(2 * getattr(self, 'phase', 0))
			if self.mode == 'cos':
				VOL = v1 * (0.5 + v2)
			else:
				VOL = v1 * (0.5 - v2)
			# correcting for sampling densitiy on grid:
			dx = (grids[0][0,-1] - grids[0][0,0]) / (grids[0].shape[1]-1)
			dy = (grids[1][-1,0] - grids[1][0,0]) / (grids[1].shape[0]-1)
			dA = dx * dy

			VOL = VOL / ( 4. * dA )

			win = win / tt.sqrt(VOL)
		win = win * self.gain
		wav = self.grating(x)
		return tt.cast(win * wav, 'float32')

	def window(self, x, y):
		γ = getattr(self, 'ratio', 1.0)
		σ = self.sigma
		sq_rad = x ** 2 + (γ * y) ** 2
		return tt.exp(-0.5 * (sq_rad / σ ** 2))

	def grating(self, x):
		ω = (2 * π) * self.freq
		ϕ = getattr(self, 'phase', 0)
		wave = getattr(tt, self.mode)
		return wave(ω * x - ϕ)

	def grids(self, model):
		grids = super().grids(model)
		if hasattr(model, 'center'):
			print('model (kernels.py) using center po')
			grids = [p - po for p, po in zip(grids, model.center)]
		else:
			print('model (kernels.py) not using center po !')

		return grids

	def rotate(self, grids):
		θ = getattr(self, 'angle', None)
		if θ is None:
			return grids
		else:
			# rotate the axes according to theta
			x, y = grids
			cos_θ = tt.cos(θ)
			sin_θ = tt.sin(θ)
			_x = x * cos_θ - y * sin_θ
			_y = x * sin_θ + y * cos_θ
			return _x, _y


class AdelsonBergen(Kernel):

	tau = Parameter('τ')

	def __init__(self, normalize=False):
		super().__init__()
		self.normalize = normalize
		self.tau = tt.scalar('tau')

	def __getitem__(self, grids):
		if isinstance(grids, (list, tuple)):
			grids = grids[0]
		s = grids / self.tau
		k = np.exp(-s) * (s ** 5 / 120) * (1 - s ** 2 / 42)
		if self.normalize:
			norm = np.sqrt(k.dot(k))
			return k / norm
		else:
			return k



class NonParametric(Kernel):

	value = Parameter('value')

	def __init__(self, normalize=False):
		super().__init__()
		self.normalize = normalize
		self.value = tt.scalar('value')

	def __getitem__(self, grids):
		if isinstance(grids, (list, tuple)):
			grids = grids[0]
		k = self.value
		if self.normalize:
			norm = np.sqrt(k.dot(k))
			return k / norm
		else:
			return k
