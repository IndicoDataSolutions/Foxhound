import numpy as np
import theano
import theano.tensor as T

def floatX(X):
	return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X):
    return theano.shared(np.asarray(X, dtype=theano.config.floatX))

def downcast_float(X):
    return np.asarray(X, dtype=np.float32)
