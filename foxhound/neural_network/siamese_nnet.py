import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from scipy.spatial.distance import cosine as cosine_scipy
from scipy.spatial.distance import euclidean as euclidean_scipy
import random
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt

from foxhound.utils import floatX
from foxhound.utils.costs import binary_crossentropy
from foxhound.utils.updates import Adadelta,NAG
from foxhound.utils.distances import cosine,euclidean
from foxhound.utils.load import mnist,lfw,gimg_top_n_for_n_labels
from foxhound.utils.vis import color_grid_vis,color_weight_vis,unit_scale

srng = RandomStreams()

def rectify(x):
	return T.maximum(x,0.)

def minibatch(data,labels,batch_size):
	x1mb = np.zeros((batch_size,n_in))
	x2mb = np.zeros_like(x1mb)
	ymb = np.zeros((batch_size,1))
	for i in range(batch_size):
		if i % 2 == 0:
			l1,l2 = random.sample(labels,2)
			x1mb[i] = random.choice(data[l1])
			x2mb[i] = random.choice(data[l2])
			ymb[i] = [0.]
		else:
			l = random.choice(labels)
			x1,x2 = random.sample(data[l],2)
			# x1,x2 = data[l][random.randint(0,len(data[l])-1)],data[l][random.randint(0,len(data[l])-1)]
			x1mb[i] = x1
			x2mb[i] = x2
			ymb[i] = [1.]
	return floatX(x1mb),floatX(x2mb),floatX(ymb)

# def dropout(*x):

def label_split(X,Y,test_size=0.2):
	labels = np.unique(Y)
	labels = np.random.permutation(labels)
	test_size = round(len(labels)*test_size)
	tr_labels,te_labels = labels[test_size:],labels[:test_size]
	tr_labels = set(tr_labels.tolist())
	te_labels = set(te_labels.tolist())
	trX = []
	teX = []
	trY = []
	teY = []
	for x,y in zip(X,Y):
		if y in tr_labels:
			trX.append(x)
			trY.append(y)
		else:
			teX.append(x)
			teY.append(y)
	trX = floatX(trX)
	teX = floatX(teX)
	trY = floatX(trY)
	teY = floatX(teY)
	return trX,teX,trY,teY

# trX,teX,trY,teY = mnist()
# X,Y,n_to_i,i_to_n = lfw(n_imgs='all')
# print len(np.unique(Y))
X,Y = gimg_top_n_for_n_labels(n_imgs=20,n_labels=5000,flatten=True,w=32,h=32)
# trX,teX,trY,teY = train_test_split(X,Y,test_size=0.25,random_state=42)
trX,teX,trY,teY = label_split(X,Y,test_size=0.2)
print trX.shape,teX.shape,trY.shape,teY.shape

tr_labels = np.unique(trY).tolist()
te_labels = np.unique(teY).tolist()
tr_data = dict(zip(tr_labels,[trX[trY==y] for y in tr_labels]))
te_data = dict(zip(te_labels,[teX[teY==y] for y in te_labels]))

n_samples,n_in = trX.shape
n_hidden = 2048
n_out = 1
batch_size = 128

dropout = T.scalar()
X = T.matrix()
Y = T.matrix()

w_in = theano.shared(floatX(np.random.randn(n_in,n_hidden)*0.01))
b_in = theano.shared(floatX(np.zeros((n_hidden))))
w = theano.shared(floatX(np.random.randn(n_hidden,n_hidden)*0.01))
b = theano.shared(floatX(np.zeros((n_hidden))))
w_out = theano.shared(floatX(np.random.randn(1,1)*0.01))
b_out = theano.shared(floatX(np.zeros((1))))

trX1_eval,trX2_eval,trY_eval = minibatch(tr_data,tr_labels,4096)
print trX1_eval.shape,trX2_eval.shape,trY_eval.shape
teX1_eval,teX2_eval,teY_eval = minibatch(te_data,te_labels,4096)

def model(x1,x2,dropout):
	hx1 = rectify(T.dot(x1,w_in)+b_in)
	hx2 = rectify(T.dot(x2,w_in)+b_in)
	if dropout != 0.:
		retain_prob = 1 - dropout
		dmask = srng.binomial(hx1.shape, p=retain_prob, dtype='int32').astype(theano.config.floatX)
        hx1 = hx1 / retain_prob * dmask
        hx2 = hx2 / retain_prob * dmask
	h2x1 = rectify(T.dot(hx1,w)+b)
	h2x2 = rectify(T.dot(hx2,w)+b)
	if dropout != 0.:
		retain_prob = 1 - dropout
		dmask = srng.binomial(h2x1.shape, p=retain_prob, dtype='int32').astype(theano.config.floatX)
        h2x1 = h2x1 / retain_prob * dmask
        h2x2 = h2x2 / retain_prob * dmask
	dists = cosine(h2x1,h2x2).dimshuffle(0, 'x')
	y = T.nnet.sigmoid(T.dot(dists,w_out)+b_out)
	return y,h2x1

X1 = T.matrix()
X2 = T.matrix()

p_sim,features = model(X1,X2,dropout)
err = binary_crossentropy(Y,p_sim)

# p_sim = model(X1,X2)
# predict = theano.function([X1,X2],p_sim)
# print predict(trX1_eval,trX2_eval).shape

params = [w_in, b_in, w, b, w_out, b_out]
grads = T.grad(err,params)
updates = Adadelta(params,grads,lr=1.)
# updates = NAG(params,grads,lr=0.05, momentum=0.9)


train = theano.function([X1,X2,Y,dropout],err,updates=updates)
predict = theano.function([X1,X2,dropout],p_sim)
transform = theano.function([X1,dropout],features)

for e in range(1000):
	for b in range(n_samples/batch_size):
		x1mb, x2mb, ymb = minibatch(tr_data, tr_labels, batch_size)
		train(x1mb, x2mb, ymb, 0.5)
	if e % 5 == 0:
		tr_pred = predict(trX1_eval, trX2_eval, 0.)
		te_pred = predict(teX1_eval, teX2_eval, 0.)
		tr_acc = metrics.accuracy_score(trY_eval, tr_pred > 0.5)
		te_acc = metrics.accuracy_score(teY_eval, te_pred > 0.5)
		img = color_grid_vis(w_in.get_value().T,transform=lambda x:color_weight_vis(x.reshape(3,32,32).transpose(1,2,0)),show=False,save='imgs/%s.png'%e)
		# img = color_grid_vis(w_in.get_value().T,transform=lambda x:color_weight_vis(x.reshape(32,32,3)),show=False,save='imgs/%s.png'%e)
		print e,tr_acc,te_acc


