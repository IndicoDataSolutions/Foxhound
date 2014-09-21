import numpy as np
import theano
import theano.tensor as T

def floatX(X):
	return np.asarray(X,dtype=theano.config.floatX)