import theano
import theano.tensor as T

def max_norm(w, n):
	norms = T.sqrt(T.sum(T.sqr(w), axis=0))
	desired = T.clip(norms, 0, n)
	w = w * (desired/ (1e-7 + norms))
	return w

def sgd(params, grads, lr=0.01):
	updates = []
	for p,g in zip(params,grads):
		updates.append((p,p-lr*g))
	return updates

def momentum(params, grads, lr, momentum):
	updates = []
	for p,g in zip(params,grads):
		m = theano.shared(p.get_value()*0.)
		v = momentum*m-lr*g
		updates.append((m,v))
		updates.append((p,p+v))
	return updates

def nag(params, grads, lr, momentum, l2=0.):
	updates = []
	for p,g in zip(params,grads):
		m = theano.shared(p.get_value()*0.)
		if l2 != 0:
			g = g + l2*p
		v = momentum*m-lr*g
		w = p + momentum*v-lr*g
		updates.append((m,v))
		updates.append((p,w))
	return updates

def rmsprop(params, grads, lr, rho=0.9, epsilon=1e-6):
	updates = []
	for p,g in zip(params,grads):
		acc = theano.shared(p.get_value()*0.)
		acc_new = rho*acc+(1-rho)*g**2
		updates.append((acc,acc_new))
		updates.append((p,p-lr*g/T.sqrt(acc_new + epsilon)))
	return updates

def adadelta(params, grads, lr=1.0, rho=0.95, epsilon=1e-6, maxnorm=2.5):
	updates = []
	for p,g in zip(params,grads):
		acc = theano.shared(p.get_value()*0.)
		acc_delta = theano.shared(p.get_value()*0.)
		acc_new = rho*acc+(1-rho)*g**2
		updates.append((acc,acc_new))

		update = g*T.sqrt(acc_delta+epsilon)/T.sqrt(acc_new+epsilon)
		updated_p = p-lr*update
		if maxnorm > 0:
			updated_p = max_norm(updated_p, maxnorm)
		updates.append((p, updated_p))

		acc_delta_new = rho*acc_delta+(1-rho)*update**2
		updates.append((acc_delta,acc_delta_new))
	return updates
