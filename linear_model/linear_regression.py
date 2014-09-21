import theano
import theano.tensor as T
from theano.compat import OrderedDict
import numpy as np

from utils import sharedX, downcast_float
from utils.costs import MSE
from utils.updates import SGD

class LinearRegression(object):

    def __init__(self):
        self.X = T.fmatrix()
        self.Y = T.fmatrix()

    def setup(self, X, y):

        assert len(X.shape) is 2
        n_examples, n_features = X.shape

        self.init_params(n_features)

        self.pred = T.dot(self.X, self.W) + self.b
        self.error = self.cost()
        self.grads = T.grad(self.error, self.params)
        self.updates = SGD(self.params, self.grads, lr=0.01)

        self.fprop = theano.function([self.X], self.pred)
        self.train = theano.function(
            [self.X, self.Y], updates=self.updates
        )

    def init_params(self, n_features, variance=0.01):
        self.W = sharedX(np.random.randn(n_features, 1) * variance)
        self.b = sharedX(np.zeros(1))
        self.params = [self.W, self.b]

    def cost(self):
        return MSE(self.Y, self.pred)

    def fit(self, X, y, epochs=10):
        X = np.atleast_2d(downcast_float(X))
        y = np.atleast_2d(downcast_float(y)).T

        self.setup(X, y)
        for epoch in xrange(epochs):
            self.train(X, y)

        return self

    def predict(self, X):
        X = np.atleast_2d(downcast_float(X))
        return self.fprop(X)


if __name__ == "__main__":
    X = np.random.random((100, 100))
    y = X.sum(axis=1)

    model = LinearRegression()
    model.fit(X, y)

    X = np.linspace(0, 1, 100)
    print model.predict(X)
