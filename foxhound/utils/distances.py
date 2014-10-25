import theano
import theano.tensor as T
from scipy.spatial.distance import cdist
from time import time

def l2norm(x):
	return T.sqrt(T.sum(T.sqr(x), axis=1))

def cosine(a, b, e=1e-6):
	return T.sum(a*b,axis=1)/T.maximum(l2norm(a)*l2norm(b),e)

"""May be useful where shapes same?"""
# def euclidean(a, b):
# 	return T.sqrt(T.sum(T.sqr(a-b), axis=1))

def euclidean(x, y):
	xx = T.sqr(T.sqrt((x*x).sum(axis=1)))
	yy = T.sqr(T.sqrt((y*y).sum(axis=1)))
	dist = T.dot(x, y.T)
	dist *= -2
	dist += xx.dimshuffle(0, 'x')
	dist += yy.dimshuffle('x', 0)
	dist = T.sqrt(T.maximum(dist, 1e-6))
	return dist
