import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from foxhound.utils import floatX,shuffle
from foxhound.utils.load import mnist
from foxhound.utils.costs import categorical_crossentropy
from foxhound.utils.updates import Adadelta,NAG
from foxhound.utils.activations import rectify,tanh,softmax

import numpy as np
from sklearn import metrics
from time import time

srng = RandomStreams()

trX,teX,trY,teY = mnist(onehot=False)
print trX.shape,teX.shape,trY.shape,teY.shape

gX = theano.shared(trX,borrow=True)
gY = theano.shared(trY,borrow=True)
print gX.type,gY.type

batch_size = 128
X = T.matrix()
Y = T.ivector()
n_in = 28*28
n_hidden = 2048
n_out = 10
n_train = trX.shape[0]
vin = T.scalar()
vh = T.scalar()
vh2 = T.scalar()
pdin = T.scalar()
pdh = T.scalar()
pdh2 = T.scalar()
lr = theano.shared(np.asarray(0.01,dtype=theano.config.floatX))
momentum = 0.95

w_in = theano.shared(floatX(np.random.randn(n_in,n_hidden)*0.01))
b_in = theano.shared(floatX(np.zeros((n_hidden))))

w = theano.shared(floatX(np.random.randn(n_hidden,n_hidden)*0.01))
b = theano.shared(floatX(np.zeros((n_hidden))))

w2 = theano.shared(floatX(np.random.randn(n_hidden,n_hidden)*0.01))
b2 = theano.shared(floatX(np.zeros((n_hidden))))

w_out = theano.shared(floatX(np.random.randn(n_hidden,n_out)*0.01))
b_out = theano.shared(floatX(np.zeros((n_out))))

def noise(X, v = 0.):
	if v != 0:
		X += srng.normal(size=X.shape, std=T.sqrt(v))
	return X

def dropout(X, p = 0.):
	if p != 0:
		retain_prob = 1 - p
		X = X / retain_prob * srng.binomial(X.shape, p=retain_prob, dtype='int32').astype('float32')
	return X

def model(X,vin,vh,vh2,pdin,pdh,pdh2):
	X = dropout(noise(X,vin),pdin)
	h = dropout(noise(rectify(T.dot(X,w_in)+b_in),vh),pdh)
	h2 = dropout(noise(rectify(T.dot(h,w)+b),vh2),pdh2)
	y = softmax(T.dot(h2,w_out)+b_out)
	return y

out = model(X,vin,vh,vh2,pdin,pdh,pdh2)
err = categorical_crossentropy(Y,out)
params = [w_in,b_in,w,b,w_out,b_out]
grads = T.grad(err,params)
# updates = Adadelta(params,grads,lr=lr)
updates = NAG(params,grads,lr,momentum)

idx = T.lscalar('idx')

givens = {
	X: gX[idx * batch_size: (idx + 1) * batch_size],
	Y: gY[idx * batch_size: (idx + 1) * batch_size],
}

train = theano.function([idx,vin,vh,vh2,pdin,pdh,pdh2],err,givens=givens,updates=updates)
predict = theano.function([X,vin,vh,vh2,pdin,pdh,pdh2],out)

t = time()
n = 0
for e in range(1000):
	trX,trY = shuffle(trX,trY)
	gX.set_value(trX)
	gY.set_value(trY)
	for b in range(n_train/batch_size):
		train(b,0.1,0.15,0.15,0.1,0.33,0.33)
		n += batch_size
	if e % 5 == 0:
		tr_pred = predict(trX,0,0,0,0,0,0)
		te_pred = predict(teX,0,0,0,0,0,0)
		tr_acc = metrics.accuracy_score(trY, np.argmax(tr_pred,axis=1))
		te_acc = metrics.accuracy_score(teY, np.argmax(te_pred,axis=1))
		print e,tr_acc,te_acc,n/(time()-t),round(time()-t),lr.get_value()
	lr.set_value((lr.get_value()*0.998).astype(theano.config.floatX))
