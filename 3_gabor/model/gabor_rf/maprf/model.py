import numpy as np
from theano import shared as sv, In as sym_in
import theano.tensor as tt
from collections import OrderedDict


class Variable:
	def __init__(self, borrow=True, readonly=False):
		self.borrow = borrow
		self.readonly = readonly

	def __get__(self, instance, owner):
		try:
			v = instance.backend[self]
		except AttributeError:
			return self
		else:
			return v.get_value(borrow=self.borrow)

	def __set__(self, instance, value):
		if self.readonly:
			raise AttributeError()
		v = instance.backend[self]
		v.set_value(value, borrow=self.borrow)

	def add_backend(self, model):
		raise NotImplementedError()

	def register_at(self, model):
		raise NotImplementedError()

	def symbolic(self, model):
		raise NotImplementedError()

	def get_backend(self, model):
		return model.backend[self]

	def set_backend(self, model, value=None):
		model.backend[self] = value


class Data(Variable):

	def add_backend(self, model):
		self.set_backend(model)

	def register_at(self, model):
		model.add_data(self)

	def symbolic(self, model):
		return model.backend[self]


class Parameter(Variable):

	def __init__(self, symvar, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.symvar = symvar

	def add_backend(self, model):
		v = self.symvar
		x = np.nan if v.ndim == 0 else np.array([], ndmin=v.ndim)
		self.set_backend(model, sv(x, self.name))

	def register_at(self, model):
		model.add_param(self)

	def symbolic(self, model):
		assert self in model._vars
		return self.symvar


def var(*args, **kwargs):
	args = list(args)
	try:
		a0 = args.pop()
	except IndexError:
		return Data(**kwargs)
	else:
		assert len(args) == 0
		return Parameter(a0, **kwargs)


class ModelType(type):
	_pars = None
	_vars = None

	def define(cls, item):
		try:
			x = getattr(cls, item)
		except AttributeError:
			setattr(cls, item, {})
		else:
			assert isinstance(x, dict)

	def __new__(mcs, name, bases, attrs):
		attrs.setdefault('_vars', set())
		attrs.setdefault('_pars', set())
		attrs.setdefault('_data', set())

		cls = super().__new__(mcs, name, bases, attrs)
		return cls

	def __init__(cls, name, bases, attrs):
		super().__init__(name, bases, attrs)
		for field, value in attrs.items():
			if isinstance(value, Variable):
				value.name = field
				cls._vars.add(value)
				value.register_at(cls)


sympar = Parameter
simvar = Variable


class Model(metaclass=ModelType):

	_vars = set()
	_pars = set()
	_data = set()

	@classmethod
	def add_data(cls, var):
		cls._data.add(var)

	@classmethod
	def add_param(cls, var):
		cls._pars.add(var)

	def __init__(self, **kwargs):
		self.__create_backend()
		for name, value in kwargs.items():
			setattr(self, name, value)

	def __create_backend(self):
		# create backend shared variables
		self.backend = {}
		for var in type(self).iter_vars():
			var.add_backend(self)

	def vars(self):
		return [v for v in self._vars]

	@classmethod
	def iter_vars(cls):
		for v in cls._vars: yield v

	a = Parameter(tt.scalar('a1'))


class SymbolicView:

	def __init__(self, model):
		self.model = model
		self.var_names = [v.name for v in type(model).iter_vars()]

	def __getattr__(self, item):
		if item in self.var_names:
			prop = getattr(type(self.model), item)
			return prop.symbolic(self.model)
		else:
			return getattr(self.model, item)


def symview(model):
	return SymbolicView(model)


if __name__ == '__main__':
	import theano.tensor as tt

	class ModelInst(Model):
		b = Data('b')
		a = Parameter(tt.scalar('a'))

		def __init__(self, **kwargs):
			super(ModelInst, self).__init__(**kwargs)
			self.backend[type(self).b] = sv(0.0, 'b')

	m1 = ModelInst()
	m2 = ModelInst(a=2.0)

	s1 = symview(m1)
	s2 = symview(m2)

	print("{} == {}? {}".format(m1._pars, m2._pars, m1._pars == m2._pars))
	print(m1.backend[list(m1._pars)[0]] == m2.backend[list(m1._pars)[0]])

	print(type(m1).a == type(m2).a)
	print("type(m{}).a = {}, m{}.a = {}".format(1, type(m1).a, 1, m1.a))
	print("type(m{}).a = {}, m{}.a = {}".format(2, type(m2).a, 2, m2.a))
	print("sym(m1).a = {}".format(type(s1.a)))
	print("sym(m1).a == sym(m2).a: {}".format(s1.a == s2.a))
	print("m1.b ({}) == m2.b ({}): {}".format(m1.b, m2.b, m1.b == m2.b))
	print("sym(m1).b = {}".format(type(s1.b)))
	print("sym(m1).b == sym(m2).b: {}".format(s1.b == s2.b))


