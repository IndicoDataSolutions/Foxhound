import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from sklearn import metrics
from time import time

from foxhound.utils import floatX, sharedX, shuffle, iter_data
from foxhound.utils.load import mnist
from foxhound.utils.costs import CCE, BCE, MSE, MAE, hinge, squared_hinge
from foxhound.utils.updates import Adadelta, NAG
from foxhound.utils.activations import rectify, tanh, softmax
from foxhound.neural_network.layers import Dense, Input

trX,teX,trY,teY = mnist(onehot=True)
print trX.shape,teX.shape,trY.shape,teY.shape

def get_params(layer):
	params = []
	while not hasattr(layer, 'X'):
		if hasattr(layer, 'params'):
			params.extend(layer.params)
		layer = layer.l_in
	return params

cost_map = {
	'cce':CCE,
	'bce':BCE,
	'mse':MSE,
	'mae':MAE,
	'hinge':hinge,
	'squared_hinge':squared_hinge
}

class Net(object):

	def __init__(self, layers, n_epochs=100, cost='cce'):
		self._layers = layers
		self.n_epochs = n_epochs
		self.cost_fn = cost_map[cost]

	def _init(self, trX, trY):
		self.layers = [Input(shape=trX.shape)]
		for i, layer in enumerate(self._layers):
			if i == len(self._layers)-1:
				layer.size = trY.shape[1]
			layer.connect(self.layers[-1])
			self.layers.append(layer)
		tr_out = self.layers[-1].output(dropout_active=True)
		te_out = self.layers[-1].output(dropout_active=False)
		Y = T.fmatrix()
		cost = self.cost_fn(Y, tr_out)
		self.params = get_params(self.layers[-1])
		grads = T.grad(cost, self.params)
		updates = Adadelta(self.params, grads)
		self._train = theano.function([self.layers[0].X, Y], cost, updates=updates)
		self._cost = theano.function([self.layers[0].X, Y], cost)
		self._predict = theano.function([self.layers[0].X], te_out)
		
		metric = T.eq(T.argmax(te_out, axis=1), T.argmax(Y, axis=1)).mean()
		self._metric = theano.function([self.layers[0].X, Y], metric)

	def fit(self, trX, trY, teX=None, teY=None):
		self._init(trX, trY)

		trX = floatX(trX)
		trY = floatX(trY)

		teX = floatX(teX)
		teY = floatX(teY)

		for e in range(self.n_epochs):
			for x, y in iter_data(trX, trY):
				cost = self._train(x, y)
			print self._metric(teX[:1024], teY[:1024])

layers = [
	Dense(size=512, activation=rectify, p_drop=0.2),
	Dense(size=512, activation=rectify, p_drop=0.5),
	Dense(activation=softmax, p_drop=0.5)
]

model = Net(layers=layers, n_epochs=100, cost='hinge')
model.fit(trX, trY, teX, teY)
