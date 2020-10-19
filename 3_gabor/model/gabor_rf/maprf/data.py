from theano import config
from theano import shared
from theano.tensor import as_tensor_variable

from .utils import empty


class Data:
	def __init__(self, x, y):
		self._x = x
		self._y = y

	def predictor(self, model):
		r = model.prediction(self._x)
		return as_tensor_variable(r, name='r')

	def likelihood(self, model):
		L = model.likelihood(self._x, self._y)
		return as_tensor_variable(L, name='L')


class SymbolicData(Data):
	@property
	def x(self):
		return self._x.get_value(borrow=True)

	@x.setter
	def x(self, value):
		self._x.set_value(value, borrow=False)

	@property
	def y(self):
		return self._y.get_value(borrow=True)

	@y.setter
	def y(self, value):
		self._y.set_value(value, borrow=False)

	def __init__(self, kwx=None, kwy=None):
		if kwx is None: kwx = {}
		kwx.setdefault('ndim', 1)
		kwx.setdefault('dtype', config.floatX)
		_x = shared(empty(**kwx), 'x')

		if kwy is None: kwy = {}
		kwy.setdefault('ndim', 1)
		kwy.setdefault('dtype', config.floatX)
		_y = shared(empty(**kwy), 'y')

		super().__init__(_x, _y)
