import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from sklearn import metrics
from time import time

from foxhound.utils import floatX, sharedX, shuffle, iter_data, iter_indices
from foxhound.utils.load import mnist
from foxhound.utils import costs
from foxhound.utils import updates
from foxhound.utils.activations import rectify, tanh, softmax, linear
from foxhound.neural_network.layers import Dense, Input
from foxhound.utils import config
import foxhound.utils.gpu as gpu

def get_params(layer):
    params = []
    while not hasattr(layer, 'X'):
        if hasattr(layer, 'params'):
            params.extend(layer.params)
        layer = layer.l_in
    return params

class Net(object):

    def __init__(self, layers, n_epochs=100, cost='cce', update='adadelta', lr=1., lr_decay=0.995):
        self._layers = layers
        self.n_epochs = n_epochs
        self.lr = sharedX(lr)
        self.lr_decay = lr_decay
        self.cost_fn = getattr(costs, cost)
        self.update_fn = getattr(updates, update)

    def setup(self, trX, trY):
        self.chunk_size = gpu.n_chunks(self.max_gpu_mem, trX)
        self.gpuX = sharedX(trX[:self.chunk_size])
        self.gpuY = theano.shared(trY[:self.chunk_size])

        self.layers = [Input(shape=trX.shape)]
        for i, layer in enumerate(self._layers):
            if i == len(self._layers)-1:
                layer.size = trY.shape[1]
            layer.connect(self.layers[-1])
            self.layers.append(layer)
        tr_out = self.layers[-1].output(dropout_active=True)
        te_out = self.layers[-1].output(dropout_active=False)
        te_pre_act = self.layers[-1].output(dropout_active=False, pre_act=True)
        X = self.layers[0].X
        Y = T.fmatrix()
        cost = self.cost_fn(Y, tr_out)
        self.params = get_params(self.layers[-1])
        grads = T.grad(cost, self.params)
        updates = self.update_fn(self.params, grads, lr=self.lr)

        idx = T.lscalar('idx')
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        givens = {
            X : self.gpuX[start:end],
            Y : self.gpuY[start:end]
        }

        self._train = theano.function(
            [idx], cost, updates=updates, givens=givens, allow_input_downcast=True
        )
        self._cost = theano.function(
            [idx], cost, givens=givens, allow_input_downcast=True
        )
        self._predict = theano.function(
            [idx], te_out, givens=givens, allow_input_downcast=True, on_unused_input='ignore'
        )
        self._predict_pre_act = theano.function(
            [idx], te_pre_act, givens=givens, allow_input_downcast=True, on_unused_input='ignore'
        )

        metric = T.eq(T.argmax(te_out, axis=1), T.argmax(Y, axis=1)).mean()
        self._metric = theano.function([X, Y], metric)

    def fit(self, trX, trY, teX=None, teY=None, max_gpu_mem=config.max_gpu_mem, batch_size=128):
        self.max_gpu_mem = max_gpu_mem
        self.batch_size = batch_size
        self.setup(trX, trY)

        trX = floatX(trX)
        trY = floatX(trY)

        teX = floatX(teX)
        teY = floatX(teY)

        t = time()
        for e in range(self.n_epochs):
            for chunkX, chunkY in iter_data(trX, trY, size=self.chunk_size):
                self.gpuX.set_value(chunkX)
                self.gpuY.set_value(chunkY)
                for batch_idx in iter_indices(chunkX, size=self.batch_size):
                    cost = self._train(batch_idx)
            self.lr.set_value(floatX(self.lr.get_value() * self.lr_decay))

    def predict_proba(self, X):
        results = []
        for chunk in iter_data(X, size=self.chunk_size):
            self.gpuX.set_value(chunk)
            for batch_idx in iter_indices(chunk, size=self.batch_size):
                results.append(self._predict(batch_idx))
        return np.vstack(results)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_pre_act(self, X):
        results = []
        for chunk in iter_data(X, size=self.chunk_size):
            self.gpuX.set_value(chunk)
            for batch_idx in iter_indices(chunk, size=self.batch_size):
                results.append(self._predict_pre_act(batch_idx))
        return np.vstack(results)
