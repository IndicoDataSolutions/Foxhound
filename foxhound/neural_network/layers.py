import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from sklearn import metrics
from time import time

from foxhound.utils import floatX, sharedX, shuffle
from foxhound.utils.load import mnist
from foxhound.utils import activations

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

class Input(object):
    def __init__(self, shape):
        self.X = tensor_map[len(shape)]
        self.output_shape = shape[1:]

    def output(self, dropout_active=True):
        return self.X

class Dense(object):
    def __init__(self, size=512, activation='rectify', p_drop=0., w_std=0.01, b_init=0.):
        self.activation = getattr(activations, activation)
        self.p_drop = p_drop
        self.size = size
        self.b_init = b_init
        self.w_std = w_std

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = np.prod(l_in.output_shape)
        self.w = sharedX(np.random.randn(self.n_in, self.size) * self.w_std)
        self.b = sharedX(np.ones(self.size) * self.b_init)
        self.params = [self.w, self.b]
        self.output_shape = (self.size,)
        print 
        print 'in  shape', l_in.output_shape
        print 'out shape',self.output_shape

    def output(self, dropout_active=True):
        X = self.l_in.output(dropout_active=dropout_active)
        if X.ndim > 2:
            X = T.flatten(X, outdim=2)
        if dropout_active and (self.p_drop > 0.):
            X = dropout(X, p = self.p_drop) 
        return self.activation(T.dot(X, self.w) + self.b)
