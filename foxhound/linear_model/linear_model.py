import theano
import theano.tensor as T
from theano.compat import OrderedDict
import numpy as np

from foxhound.utils import sharedX, floatX, iter_data, iter_indices
from foxhound.utils.updates import adadelta
import foxhound.utils.config as config
import foxhound.utils.gpu as gpu

class LinearModel(object):

    def __init__(self, rng=None, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.X = T.fmatrix()
        self.Y = T.fmatrix()
        self.rng = rng if rng else np.random.RandomState()

    def setup(self, X, Y):

        # calculate chunk sizes
        self.chunk_size = gpu.n_chunks(self.max_gpu_mem, X, Y) 
        self.gpuX = sharedX(X[:self.chunk_size])
        self.gpuY = theano.shared(Y[:self.chunk_size])

        assert len(X.shape) is 2
        n_examples, n_features = X.shape
        n_outputs = Y.shape[1]

        assert len(Y.shape) is 2
        self.init_params(n_features, n_outputs)

        self.pred = self.activation(T.dot(self.X, self.W) + self.b)
        self.error = self.cost() + self.regularization()
        self.grads = T.grad(self.error, self.params)
        self.updates = adadelta(self.params, self.grads, lr=self.lr)

        idx = T.lscalar('idx')
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        self.givens = {
            self.X : self.gpuX[start:end],
            self.Y : self.gpuY[start:end]
        }

        self._predict = theano.function(
            [idx], self.pred, givens=self.givens, on_unused_input='ignore'
        )
        self._fit = theano.function(
            [idx], updates=self.updates, givens=self.givens
        )

    def init_params(self, n_features, n_outputs=1, variance=0.01):
        self.W = sharedX(self.rng.randn(n_features, n_outputs) * variance)
        self.b = sharedX(np.zeros(n_outputs))
        self.params = [self.W, self.b]

    def activation(self, preactivation):
        return preactivation

    def cost(self):
        raise NotImplementedError

    def regularization(self):
        return 0

    def fit(self, X, Y, lr=1., epochs=100, batch_size=128, max_gpu_mem=config.max_gpu_mem):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_gpu_mem = max_gpu_mem

        X = np.atleast_2d(floatX(X))
        if len(Y.shape) is 1:
            Y = floatX(Y).reshape(-1, 1)

        self.setup(X, Y)
        for epoch in xrange(epochs):
            for chunkX, chunkY in iter_data(X, Y, size=self.chunk_size):
                self.gpuX.set_value(chunkX)
                self.gpuY.set_value(chunkY)
                for batch_idx in iter_indices(X, size=self.batch_size):
                    self._fit(batch_idx)

        return self

    def predict(self, X):
        X = floatX(X)

        results = []
        for chunkX in iter_data(X, size=self.chunk_size):
            self.gpuX.set_value(chunkX)
            for batch_idx in iter_indices(X, size=self.batch_size):
                preds = self._predict(batch_idx)
                if len(preds.shape) is 1:
                    preds = preds.reshape(-1, 1)
                results.append(preds)

        return np.vstack(results)

    def decision_function(self, *args, **kwargs):
        return self.predict(X, *args, **kwargs)

    def get_params(self):
        return (self.W.get_value(), self.W.get_value())

    def set_params(self, **params):
        for k, v in params.values():
            getattr(self, k).set_value(v)
