import theano
import theano.tensor as T

def categorical_crossentropy(y_target,y_pred):
	return T.nnet.categorical_crossentropy(y_pred,y_target).mean()

def binary_crossentropy(y_target,y_pred):
	return T.nnet.binary_crossentropy(pred, target).mean()

def mean_squared_error(y_target,y_pred):
	return T.sqr(y_target-y_pred).mean()