import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from sklearn import metrics
from time import time
import random

from foxhound.utils import floatX, sharedX, shuffle
from foxhound.utils.load import mnist
from foxhound.utils import activations
from foxhound.utils.distances import euclidean


srng = RandomStreams()

tensor_map = {
    2:T.matrix(),
    3:T.tensor3(),
    4:T.tensor4(),
}

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


class Layer(object):

    def __repr__(self):
        size = self.size or None
        return "%s(size=%s)" % (self.__class__.__name__, str(self.size))

class Input(Layer):

    def __init__(self, shape):
        self.X = tensor_map[len(shape)+1]
        self.output_shape = shape
        self.size = shape[0]

    def output(self, dropout_active=True):
        return self.X


class Dense(Layer):

    def __init__(self, size=512, activation='rectify', p_drop=0., w_std=0.01, b_init=0.):
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.p_drop = p_drop
        self.size = size
        self.b_init = b_init
        self.w_std = w_std

    def weight_init(self):
        return sharedX(np.random.randn(self.n_in, self.size) * self.w_std)

    def bias_init(self):
        return sharedX(np.ones(self.size) * self.b_init)

    def get_params(self):
        return [self.w, self.b]

    def setup(self, l_in, trX, trY):
        self.l_in = l_in
        self.trX = trX
        self.trY = trY

        self.n_in = np.prod(l_in.output_shape)
        self.w = self.weight_init()
        self.b = self.bias_init()
        self.params = self.get_params()
        self.output_shape = (self.size,)

    def preactivation(self, dropout_active=True):
        X = self.l_in.output(dropout_active=dropout_active)
        if X.ndim > 2:
            X = T.flatten(X, outdim=2)
        if dropout_active and (self.p_drop > 0.):
            X = dropout(X, p = self.p_drop)
        return self.transform(X)

    def output(self, dropout_active=True):
        return self.activation(
            self.preactivation(dropout_active=dropout_active)
        )

    def transform(self, X):
        return T.dot(X, self.w) + self.b

    def __repr__(self):
        size = self.size or None
        return "%s(size=%s, activation='%s')" % (
            self.__class__.__name__, str(self.size), self.activation_str
        )


class Euclidean(Dense):

    def __init__(self, size=10, activation='linear', **kwargs):
        Dense.__init__(self, size=size, activation=activation, **kwargs)

    def transform(self, X):
        return euclidean(X, self.w)

    def weight_init(self):
        return sharedX(random.sample(self.trX, self.size))

    def bias_init(self):
        return None

    def get_params(self):
        return [self.w]
