import theano
import theano.tensor as T
import numpy as np

def max_norm(w, n):
	if n > 0:
		norms = T.sqrt(T.sum(T.sqr(w), axis=0))
		desired = T.clip(norms, 0, n)
		w = w * (desired/ (1e-7 + norms))
	return w

def clip_norm(g, n):
	if n > 0:
		norm = T.sqrt(T.sum(T.sqr(g)))
		desired = T.clip(norm, 0, n)
		g = g * (desired/ (1e-7 + norm))
	return g

def clip_norms(gs, n):
	return [clip_norm(g, n) for g in gs]

def regularize(p, l1=0., l2=0., maxnorm=0.):
	p = max_norm(p, maxnorm)
	p -= p * l2
	p -= l1
	return p

def sgd(params, grads, lr=0.01, l1=0., l2=0., maxnorm=0., clipnorm=0):
	updates = []
	grads = clip_norms(grads, clipnorm)
	for p,g in zip(params,grads):
		updated_p = p - lr *g
		updated_p = regularize(updated_p, l1=l1, l2=l2, maxnorm=maxnorm)
		updates.append((p, updated_p))
	return updates

def momentum(params, grads, lr=0.01, momentum=0.9, l1=0., l2=0., maxnorm=0., clipnorm=0):
	updates = []
	grads = clip_norms(grads, clipnorm)
	for p,g in zip(params,grads):
		m = theano.shared(p.get_value() * 0.)
		v = (momentum * m) - (lr * g)
		updates.append((m, v))

		updated_p = p + v
		updated_p = regularize(updated_p, l1=l1, l2=l2, maxnorm=maxnorm)
		updates.append((p, updated_p))
	return updates

def nag(params, grads, lr=0.01, momentum=0.9, l1=0., l2=0., maxnorm=0., clipnorm=0):
	updates = []
	grads = clip_norms(grads, clipnorm)
	for p,g in zip(params,grads):
		m = theano.shared(p.get_value() * 0.)
		v = (momentum * m) - (lr * g)
		updates.append((m,v))

		updated_p = p + momentum * v - lr * g
		updated_p = regularize(updated_p, l1=l1, l2=l2, maxnorm=maxnorm)
		updates.append((p, updated_p))
	return updates

def rmsprop(params, grads, lr=0.001, rho=0.9, epsilon=1e-6, l1=0., l2=0., maxnorm=0., clipnorm=0):
	updates = []
	grads = clip_norms(grads, clipnorm)
	for p,g in zip(params,grads):
		acc = theano.shared(p.get_value() * 0.)
		acc_new = rho * acc + (1 - rho) * g ** 2
		updates.append((acc, acc_new))

		updated_p = p - lr * g / T.sqrt(acc_new + epsilon))
		updated_p = regularize(updated_p, l1=l1, l2=l2, maxnorm=maxnorm)
		updates.append((p, updated_p))
	return updates

def adagrad(params, grads, lr=0.01, epsilon=1e-6, l1=0., l2=0., maxnorm=0., clipnorm=0):
	updates = []
	grads = clip_norms(grads, clipnorm)
	for p,g in zip(params,grads):
		acc = theano.shared(p.get_value() * 0.)
		acc_new += g ** 2
		updates.append((acc, acc_new))

		updated_p = p - (lr / T.sqrt(acc_new + epsilon)) * g
		updated_p = regularize(updated_p, l1=l1, l2=l2, maxnorm=maxnorm)
		updates.append((p, updated_p))
	return updates		

def adadelta(params, grads, lr=1.0, rho=0.95, epsilon=1e-6, l1=0., l2=0., maxnorm=0., clipnorm=0):
	updates = []
	grads = clip_norms(grads, clipnorm)
	for p,g in zip(params,grads):
		acc = theano.shared(p.get_value() * 0.)
		acc_delta = theano.shared(p.get_value() * 0.)
		acc_new = rho * acc + (1 - rho) * g ** 2
		updates.append((acc,acc_new))

		update = g * T.sqrt(acc_delta + epsilon) / T.sqrt(acc_new + epsilon)
		updated_p = p - lr * update
		updated_p = regularize(updated_p, l1=l1, l2=l2, maxnorm=maxnorm)
		updates.append((p, updated_p))

		acc_delta_new = rho * acc_delta + (1 - rho) * update ** 2
		updates.append((acc_delta,acc_delta_new))
	return updates
