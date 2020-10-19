from theano import function, In as _in


class Activity:
	@property
	def x(self):
		return self.data['x'].get_value(borrow=True)

	@x.setter
	def x(self, value):
		self.data['x'].set_value(value, borrow=True)

	@property
	def y(self):
		return self.data['y'].get_value()

	@y.setter
	def y(self, value):
		self.data['y'].set_value(value)

	def __init__(self, model, factory):
		self.data = model.data[factory]
		inputs = []
		for p in type(model)._pars:
			new_i = _in(p.symvar, value=p.get_backend(model).container)
			inputs.append(new_i)
		outputs = model.results[factory]
		updates = model.updates[factory]
		# build predictor
		y = outputs['y']
		self.generator = function(inputs, y, updates=updates[y], allow_input_downcast=True)
		# build simulator
		r = outputs['r']
		self.predictor = function(inputs, r, allow_input_downcast=True)
		# build log-likelihood
		L = outputs['L']
		self.inference = function(inputs, L, allow_input_downcast=True)

	def simulate(self):
		return self.generator()

	def predict(self):
		return self.predictor()

	def loglik(self):
		return self.inference()


class ActivityFactory:

	def create(self, model):
		model.updates[self] = {}
		model.results[self] = {}
		self.add_predictor(model)
		self.add_generator(model)
		self.add_inference(model)
		return Activity(model, self)

	def add_data(self, model, x_dim):
		raise NotImplementedError()

	def add_predictor(self, model):
		raise NotImplementedError()

	def add_generator(self, model, rng=None):
		raise NotImplementedError()

	def add_inference(self, model):
		raise NotImplementedError()
