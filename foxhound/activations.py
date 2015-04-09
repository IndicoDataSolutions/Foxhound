import theano
import theano.tensor as T

class Softmax(object):

	def __init__(self):
		pass

	def __call__(self, x):
	    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
	    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

class Rectify(object):

	def __init__(self):
		pass

	def __call__(self, x):
		return (x + abs(x)) / 2.0

class LeakyRectify(object):

	def __init__(self, leak=0.25):
		self.leak = leak

	def __call__(self, x):
		f1 = 0.5 * (1 + self.leak)
		f2 = 0.5 * (1 - self.leak)
		return f1 * x + f2 * abs(x)

class Tanh(object):

	def __init__(self):
		pass

	def __call__(self, x):
		return T.tanh(x)

class Sigmoid(object):

	def __init__(self):
		pass

	def __call__(self, x):
		return T.nnet.sigmoid(x)

class Linear(object):

	def __init__(self):
		pass

	def __call__(self, x):
		return x

class SteeperSigmoid(object):

	def __init__(self, scale=3.75):
		self.scale = scale

	def __call__(self, x):
		return 1./(1. + T.exp(-self.scale * x))