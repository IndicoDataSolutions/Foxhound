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

class Input(object):
    def __init__(self, ndim=2):
        self.X = T.TensorType(theano.config.floatX, (False,)*ndim)
        self.output_shape = [784]

    def output(self, dropout_active=True):
        return self.X

class Dense(object):
    def __init__(self, size, activation=rectify, p_drop=0., w_std=0.01, b_init=0.):
        self.activation = activation
        self.p_drop = p_drop
        self.output_shape = [size]
        print 
        print l_in.output_shape
        print self.output_shape

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = np.prod(l_in.output_shape)
        self.w = sharedX(np.random.randn(self.n_in, size) * w_std)
        self.b = sharedX(np.ones(size) * b_init)
        self.params = [self.w, self.b]

    def output(self, dropout_active=True):
        X = self.l_in.output(dropout_active=dropout_active)
        if X.ndim > 2:
            X = T.flatten(X, outdim=2)
        if dropout_active and (self.p_drop > 0.):
            X = dropout(X, p = self.p_drop) 
        return self.activation(T.dot(X, self.w) + self.b)
