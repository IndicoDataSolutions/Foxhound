import theano
import theano.tensor as T
from theano.compat import OrderedDict
import numpy as np

from foxhound.utils import sharedX, floatX, iter_data, shuffle
from foxhound.utils.updates import SGD,Adadelta

class LinearModel(object):

    def __init__(self, rng=None, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.X = T.fmatrix()
        self.Y = T.fmatrix()
        self.rng = rng if rng else np.random.RandomState()

    def setup(self, X, Y):

        assert len(X.shape) is 2
        n_examples, n_features = X.shape
        n_outputs = Y.shape[1]

        assert len(Y.shape) is 2
        self.init_params(n_features, n_outputs)

        self.pred = self.activation(T.dot(self.X, self.W) + self.b)
        self.error = self.cost() + self.regularization()
        self.grads = T.grad(self.error, self.params)
        self.updates = Adadelta(self.params, self.grads, lr=self.lr)

        self.fprop = theano.function([self.X], self.pred)
        self.train = theano.function(
            [self.X, self.Y], updates=self.updates
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

    def fit(self, X, Y, lr=1., epochs=100, batch_size=128):
        self.lr = lr
        X = np.atleast_2d(floatX(X))
        Y = np.atleast_2d(floatX(Y))

        self.setup(X, Y)
        for epoch in xrange(epochs):
            for bX, bY in iter_data(X, Y, size=batch_size, shuffle=True):
                self.train(bX, bY)

        return self

    def predict(self, X):
        X = np.atleast_2d(floatX(X))
        return self.fprop(X)

    def decision_function(self, *args, **kwargs):
        return self.predict(X, *args, **kwargs)

    def get_params(self):
        return (self.W.get_value(), self.W.get_value())

    def set_params(self, **params):
        for k, v in params.values():
            getattr(self, k).set_value(v)
