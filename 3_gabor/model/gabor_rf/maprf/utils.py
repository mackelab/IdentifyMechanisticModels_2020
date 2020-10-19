import numpy as np
import numpy.random as nr
from numpy import nan as NAN
import theano.tensor as tt
import scipy.special as special


def empty(ndim, **kwargs):
	if ndim == 0:
		return NAN
	else:
		return np.array([], ndmin=ndim, **kwargs)


def pol2cart(v):
	return v[0] * tt.stack([tt.cos(v[1]), tt.sin(v[1])])


def cart2pol(v):
	u1 = tt.sqrt(v[0]**2 + v[1]**2)
	u2 = tt.arctan2(v[1], v[0])
	return tt.stack([u1, u2])


def identity(v):
	return 1. * v

def expit(v):
	return tt.inv(1. + tt.exp(tt.neg(v)))

def scaled_expit_qc(v):
	return np.pi/2 * expit(v)

def scaled_expit_hc(v):
	return np.pi * expit(v)

def scaled_expit_fc(v):
	return 2*np.pi * expit(v)	

def scaled_expit_i(v):
	return 2 * expit(v) - 1


def augment_expit(v):
	return tt.concatenate([tt.constant(np.array([1.]), name='constant1', ndim=1), 
						  np.pi/2. * expit(v)])

def normal_from_ci(p1, p2, f=None):
	β, α = [x for x in zip(p1, p2)]
	if f is not None:
		try:
			T = getattr(np, f)
		except AttributeError:
			T = getattr(special, f)
		β = [T(x) for x in β]
	ζ = [special.erfinv(2 * x - 1) for x in α]

	den = ζ[1] - ζ[0]
	σ = np.sqrt(.5) * (β[1] - β[0]) / den
	# μ = (β[1] * ζ[1] - β[0] * ζ[0]) / den
	μ = 0.5 * (β[1] + β[0]) + np.sqrt(0.5) * σ * (ζ[0] + ζ[1])

	return μ, σ
