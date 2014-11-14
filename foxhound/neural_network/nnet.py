import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from sklearn import metrics
from time import time

from foxhound.utils import floatX, sharedX, shuffle
from foxhound.utils.load import mnist
from foxhound.utils.costs import categorical_crossentropy
from foxhound.utils.updates import Adadelta, NAG
from foxhound.utils.activations import rectify, tanh, softmax

srng = RandomStreams()

trX,teX,trY,teY = mnist(onehot=False)
print trX.shape,teX.shape,trY.shape,teY.shape

class Net(object):

	def __init__(self, layers):
		self.layers = [Input(ndim=2)]
		for layer in layers:
			layer.connect(self.layers[-1])
			self.layers.append(layer)
