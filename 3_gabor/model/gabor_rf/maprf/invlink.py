import theano.tensor as tt


def explin(x):
	return tt.where(x >= 0, 1 + x, tt.exp(x))


def log_exp1p(x):
	return tt.log1p(tt.exp(x))

