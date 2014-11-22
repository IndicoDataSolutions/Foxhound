import theano
import theano.tensor as T

def softmax(x):
    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def temp_softmax(x, t):
	e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))/t
	return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

rectify = lambda x: (x + abs(x)) / 2.0
tanh = T.tanh
sigmoid = T.nnet.sigmoid
linear = lambda x: x

cost_mapping = {
	linear: 'mse',
	tanh: 'mse',
	softmax: 'cce',
	sigmoid: 'bce'
}
