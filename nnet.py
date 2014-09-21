import theano
import theano.tensor as T
from load import mnist
import numpy as np
from costs import categorical_crossentropy
from updates import Adadelta

batch_size = 128
X = T.matrix()
Y = T.matrix()
n_in = 28*28
n_hidden = 512
n_out = 10

w_in = theano.shared(floatX(np.random.randn(n_in,n_hidden)*0.01))
w_out = theano.shared(floatX(np.random.randn(n_hidden,n_out)*0.01))
b_in = theano.shared(floatX(np.zeros((n_hidden))))
b_out = theano.shared(floatX(np.zeros((n_out))))

def model(X):
	h = T.tanh(T.dot(X,w_in)+b_in)
	y = T.nnet.softmax(T.dot(h,w_out)+b_out)
	return y

out = model(X)
err = categorical_crossentropy(Y,out)
params = [w_in,b_in,w_out,b_out]
grads = T.grad(err,params)
updates = Adadelta(params,grads)

train = theano.function([X,Y],err,updates=updates)
predict = theano.function([X],out)