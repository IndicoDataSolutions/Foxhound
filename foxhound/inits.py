import numpy as np

import theano
import theano.tensor as T

from theano_utils import sharedX, floatX, intX

class Uniform(object):
	def __init__(self, scale=0.05, np_rng=np.random.RandomState(42)):
		self.scale = 0.05
		self.np_rng = np_rng

	def __call__(self, shape):
		return sharedX(self.np_rng.uniform(low=-self.scale, high=self.scale, size=shape))

class Normal(object):
	def __init__(self, loc=0., scale=0.05, np_rng=np.random.RandomState(42)):
		self.scale = scale
		self.loc = loc
		self.np_rng = np_rng

	def __call__(self, shape):
		return sharedX(self.np_rng.normal(loc=self.loc, scale=self.scale, size=shape))

class Orthogonal(object):
	""" benanne lasagne ortho init (faster than qr approach)"""
	def __init__(self, scale=1.1, np_rng=np.random.RandomState(42)):
		self.scale = scale
		self.np_rng = np_rng

	def __call__(self, shape):
	    flat_shape = (shape[0], np.prod(shape[1:]))
	    a = self.np_rng.normal(0.0, 1.0, flat_shape)
	    u, _, v = np.linalg.svd(a, full_matrices=False)
	    q = u if u.shape == flat_shape else v # pick the one with the correct shape
	    q = q.reshape(shape)
	    return sharedX(self.scale * q[:shape[0], :shape[1]])

class Constant(object):

	def __init__(self, c=0.):
		self.c = c

	def __call__(self, shape):
		return sharedX(np.ones(shape) * self.c)

class Identity(object):

	def __init__(self, scale=0.5):
		self.scale = scale
		pass

	def __call__(self, shape):
		return sharedX(np.identity(shape[0]) * self.scale)