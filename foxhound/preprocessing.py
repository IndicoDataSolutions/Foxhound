import numpy as np
import costs

from utils import numpy_array

def standardize_X(shape, X):
	if not numpy_array(X):
		X = np.asarray(X)

	if len(shape) == 4 and len(X.shape) == 2:
		return X.reshape(-1, shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2)
	else:
		return X

def standardize_Y(shape, Y):
	if not numpy_array(Y):
		Y = np.asarray(Y)
	if len(Y.shape) == 1:
		Y = Y.reshape(-1, 1)
	if len(Y.shape) == 2 and len(shape) == 2:
		if shape[-1] != Y.shape[-1]:
			return one_hot(Y, n=shape[-1])
		else:
			return Y
	else:
		return Y

def one_hot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh