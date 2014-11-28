# code for weights initiallzation
from numpy.random import normal, uniform
from numpy import ones

from foxhound.utils import sharedX

class Uniform(object):

	def __init__(self, low=0, high=1):
		self.low = low
		self.high = high

	def __call__(self, shape):
		return sharedX(uniform(low=self.low, high=self.high, size=shape))

class Gaussian(object):

	def __init__(self, mean=0, std_dev=0.01):
		self.mean = mean
		self.std_dev = std_dev

	def __call__(self, shape):
		return sharedX(normal(loc=self.mean, scale=self.std_dev, size=shape))

class Constant(object):

	def __init__(self, value):
		self.value = value

	def __call__(self, shape):
		return sharedX(ones(shape) * self.value)

Normal = Gaussian
