from theano import tensor as tt
from theano import scan, shared
from theano.tensor.nnet import conv2d
from theano.tensor import as_tensor_variable
from maprf.utils import empty
from theano import as_op
from scipy.signal import lfilter
from numpy import array, inf


def infer_conv_shape(node, input_shapes):
	x_shp, k_shp = input_shapes
	return [x_shp[:]]


@as_op(itypes=[tt.dvector, tt.dvector],
       otypes=[tt.dvector],
       infer_shape=infer_conv_shape)
def conv(x, k):
	return lfilter(k, array([1.0], ndmin=1), x)


PROJECTION = 0
CONVOLUTION = 1


class Projection:

	@property
	def kernel(self):
		return self.parent.kernel[self.key]

	def __init__(self, key, parent):
		self.key = key
		self.parent = parent
		self.buffer = shared(empty(1), 'zs')

	def __call__(self, x):
		x = tt.flatten(x, 2)
		k = self.kernel.flatten()
		return tt.dot(x, k)


class Convolution:
	@property
	def kernel(self):
		return self.parent.kernel[self.key]

	def __init__(self, key, parent):
		self.key = key
		self.parent = parent
		self.buffer = shared(empty(3), 'zs')

	def __call__(self, x):
		x = x.dimshuffle(0, 'x', 1, 2)
		shp_x = (None, 1, None, None)
		k = tt.shape_padleft(self.kernel, 2)
		shp_k = (1, 1, None, None)
		y = conv2d(x, k, shp_x, shp_k, 'half', filter_flip=False)
		return tt.squeeze(y)


class SeparableRank1:

	@property
	def filt_s(self):
		return self.stage1[self.mode]

	def __init__(self, ks=None, kt=None, mode=PROJECTION):
		self.kernel = {'s': tt.matrix('ks') if ks is None else ks,
		               't': tt.vector('kt') if kt is None else kt}
		self.buffer = {ks: shared(empty(2), '__ks'),
		               kt: shared(empty(1), '__kt')}
		self.stage1 = [Projection('s', self), Convolution('s', self)]
		self.mode = mode

	def params(self):
		return self.kernel['s'], self.kernel['t']

	def filt_t(self, x):
		def step(h, accum, i, buffer):
			increment = buffer[:-i] * h
			zt = tt.inc_subtensor(accum[i:], increment)
			return zt, i + 1

		kt = self.kernel['t']
		return x * kt
        
		#i_info = [kt[1:]]
		#o_info = [x * kt[0], 1]
		#out, _ = scan(step, i_info, o_info, [x])
		#return out[0][-1]

	def __call__(self, x, updates=None):
		y = as_tensor_variable(self.filt_s(x), name='zs')
		if updates is not None:
			updates[self.filt_s.buffer] = y

		z = as_tensor_variable(self.filt_t(y), name='zt')
		return as_tensor_variable(z.clip(-inf, 4), 'zf')
