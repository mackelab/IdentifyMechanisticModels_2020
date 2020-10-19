import numpy as np
import numpy.random as nr
from numpy.core.umath import pi as π


def randn(μ, Σ):
	try:
		x = nr.normal(size=μ.shape)
	except AttributeError:
		x = nr.normal()
	return μ + np.dot(Σ, x)


class EllipticalSliceSampling:
	def __init__(self, logp, vars, width=2 * π):
		self.logp = logp
		self.vars = vars
		self.width = width
		self.mu = {v.name: None for v in vars}
		self.sd = {v.name: None for v in vars}

	def a_step(self, **q0):
		# set the slice
		logL = self.logp(**q0)
		ϑ = logL - nr.standard_exponential()

		# draw from prior
		nu = {}
		for v in self.vars:
			μ = self.mu[v.name]
			Σ = self.sd[v.name]
			nu[v.name] = randn(μ, Σ)

		# set up a bracket around the current point
		φ = nr.uniform(low=0.0, high=self.width)
		φmin = φ - self.width
		φmax = φ

		while True:
			c, s = np.cos(φ), np.sin(φ)
			q1 = {}
			for v in self.vars:
				x, ν, μ = q0[v.name], nu[v.name], self.mu[v.name]
				q1[v.name] = (x - μ) * c + (ν - μ) * s + μ

			logL = self.logp(**q1)
			if logL > ϑ:
				return q1
			if φ > 0:
				φmax = φ
			elif φ < 0:
				φmin = φ
			else:
				raise RuntimeError()
			φ = nr.uniform(low=φmin, high=φmax)
