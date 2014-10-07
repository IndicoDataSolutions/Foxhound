import theano
import theano.tensor as T

def l2norm(x):
	return T.sqrt(T.sum(T.sqr(x), axis=1))

def cosine(a,b,e=1e-6):
	return T.sum(a*b,axis=1)/T.maximum(l2norm(a)*l2norm(b),e)

def euclidean(a,b,e=1e-6):
	return T.sqrt(T.sum(T.sqr(a-b), axis=1))