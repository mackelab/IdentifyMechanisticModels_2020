import numpy as np
import theano
import theano.tensor as tt
from theano import In
from theano.tensor.shared_randomstreams import RandomStreams

from maprf.utils import empty


class Generator:

	_sample = None

	@property
	def x(self):
		return self._x.get_value()

	@x.setter
	def x(self, value):
		self._x.set_value(value, borrow=True)

	@property
	def y(self):
		return self._y.get_value()

	@property
	def grid_x(self):
		g = self.rf.grids['s'][0]
		return g.get_value()

	@grid_x.setter
	def grid_x(self, value):
		g = self.rf.grids['s'][0]
		g.set_value(value)

	@property
	def grid_y(self):
		g = self.rf.grids['s'][1]
		return g.get_value()

	@grid_y.setter
	def grid_y(self, value):
		g = self.rf.grids['s'][1]
		g.set_value(value)

	@property
	def axis_t(self):
		g = self.rf.grids['t'][0]
		return g.get_value()

	@axis_t.setter
	def axis_t(self, value):
		g = self.rf.grids['t'][0]
		g.set_value(value)

	def __init__(self, rf, emt, seed=None):
		self.rf = rf
		self.emt = emt
		self.rng = RandomStreams(seed)

		self._x = theano.shared(empty(3, dtype='float32'), 'x', allow_downcast=True)
		self._y = theano.shared(empty(1, dtype='int64'), 'y', allow_downcast=True)

		self.inputs = {}
		self.buffer = {}
		for p in emt.params():
			self.add_inputs(p)
		for p in rf.kernel['s'].params():
			self.add_inputs(p)
		for p in rf.kernel['t'].params():
			self.add_inputs(p)

	def _set_params(self, obj, p):
		for name, value in p.items():
			v = getattr(obj, name)
			b = self.buffer[v]
			b.set_value(np.cast[theano.config.floatX](value))

	def set_params(self, params):
		if 'glm' in params:
			self._set_params(self.emt, params['glm'])
		if 'kernel' in params:
			p = params['kernel']
			if 's' in params['kernel']:
				self._set_params(self.rf.kernel['s'], p['s'])
			if 't' in params['kernel']:
				self._set_params(self.rf.kernel['t'], p['t'])

	def build(self):
		z = self.rf.predict(self._x)
		r = self.emt.prediction(z)
		y = self.emt.generation(r, self.rng)
		# import pdb; pdb.set_trace()
		i = list(self.inputs.values())
		u = [y.update] + [(self._y, tt.cast(y, 'int64'))]
		self._sample = theano.function(i, r, updates=u, mode=theano.Mode(optimizer="fast_compile"), on_unused_input='ignore', allow_input_downcast=True)
		#self._sample = theano.function(i, r, updates=u, mode='FAST_COMPILE', on_unused_input='ignore', allow_input_downcast=True)
		# self._sample = theano.function(i, r, mode='FAST_COMPILE', on_unused_input='ignore', allow_input_downcast=True)

	def simulate(self):
		return self._sample()
		# rate = self._sample()
		# count = rate * self.buffer[self.emt.binsize]
		# spikes = nr.poisson(count)
		# self._y.set_value(spikes)

	def add_buffer(self, p):
		self.buffer[p] = theano.shared(np.cast[theano.config.floatX](empty(p.ndim)), p.name)

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
