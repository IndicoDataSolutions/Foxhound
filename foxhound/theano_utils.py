import numpy as np
import theano
import theano.tensor as T

def l2norm(x, e=1e-8):
    return T.sqrt(T.sum(T.sqr(x), axis=1)+e)

def cosine(a, b):
    return T.sum(a*b, axis=1)/(l2norm(a)*l2norm(b))

def euclidean(a, b, e=1e-8):
	return T.sqrt(T.sum(T.sqr(a-b), axis=1)+e)

def intX(X):
    return np.asarray(X, dtype=np.int32)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def sharedNs(shape, n, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape)*n, dtype=dtype, name=name)

def downcast_float(X):
    return np.asarray(X, dtype=np.float32)
