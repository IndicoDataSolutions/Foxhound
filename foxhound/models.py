import sys
import numpy as np
from time import time

import theano
import theano.tensor as T

import ops
import costs
import activations
import iterators
import async_iterators
from utils import instantiate
from preprocessing import standardize_X, standardize_Y
from theano_utils import pair_cosine, euclidean

def init(model):
    for i in range(1, len(model)):
        model[i].connect(model[i-1])
        if hasattr(model[i], 'init'):
            model[i].init()
    return model

def collect_updates(model, cost):
    updates = []
    for op in model[1:]:
        if hasattr(op, 'update'):
            updates.extend(op.update(cost))
    return updates

def collect_infer_updates(model):
    updates = []
    for op in model[1:]:
        if hasattr(op, 'infer_update'):
            updates.extend(op.infer_update)
    return updates

def collect_reset_updates(model):
    updates = []
    for op in model[1:]:
        if hasattr(op, 'reset_update'):
            updates.extend(op.reset_update())
    return updates

def collect_cost(model):
    cost = 0
    for op in model[1:]:
        if hasattr(op, 'cost'):
            cost += op.cost()
    return cost

class Network(object):

    def __init__(self, model, acts=[0], cost=None, verbose=2, iterator='linear'):

        if cost is not None:
            self.cost = instantiate(costs, cost)
        else:
            if isinstance(model[-1], ops.Activation):
                if isinstance(model[-1].activation, activations.Sigmoid):
                    self.cost = instantiate(costs, 'bce')
                elif isinstance(model[-1].activation, activations.Softmax):
                    self.cost = instantiate(costs, 'cce')
                else:
                    self.cost = instantiate(costs, 'mse')
            else:
                self.cost = instantiate(costs, 'mse')

        self.verbose = verbose
        self.model = init(model)
        try:
            self.iterator = instantiate(iterators, iterator)
        except:
            self.iterator = instantiate(async_iterators, iterator)

        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        acts = [l.op({'dropout':False, 'bn_active':False, 'infer':False}) for l in [self.model[idx] for idx in acts]]
        self.X = self.model[0].X
        self.Y = T.TensorType(theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        cost = self.cost(self.Y, y_tr)

        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X, self.Y], cost, updates=self.updates)
        self._predict = theano.function([self.X], y_te)
        self._acts = theano.function([self.X], acts)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, trY, n_iter=1):
        out_shape = self.model[-1].out_shape
        n = 0.
        t = time()
        costs = []
        for e in range(n_iter):
            epoch_costs = []
            for xmb, ymb in self.iterator.iterXY(trX, trY):
                # print xmb.shape, ymb.shape
                c = self._train(xmb, ymb)
                epoch_costs.append(c)
                n += len(ymb)
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = len(trY)*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rIter %d Seen %d samples Avg cost %0.4f Examples per second %d Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)
            n_per_sec = n / (time() - t)
            n_left = len(trY)*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f Examples per second %d Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
        return costs

    def infer_iterator(self, X):
        for xmb in self.iterator.iterX(X):
            self._infer(xmb)

    def infer_idxs(self, X):
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._infer(xmb)

    def infer(self, X):
        self._reset()
        if isinstance(self.iterator, iterators.Linear):
            return self.infer_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.infer_idxs(X)
        else:
            raise NotImplementedError

    def acts(self, X):
        states = None
        for xmb in self.iterator.iterX(X):
            mb_states = self._acts(xmb)
            if states is None:
                states = [[state] for state in mb_states]
            else:
                for i, state in enumerate(mb_states):
                    states[i].append(state)
        return [np.vstack(state) for state in states]

    def predict(self, X):
        if isinstance(self.iterator, iterators.Linear):
            return self.predict_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.predict_idxs(X)
        else:
            raise NotImplementedError

    def predict_iterator(self, X):
        preds = []
        for xmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
        # return np.vstack(preds)
        # return preds
        return np.concatenate(preds, axis=1)

    def predict_idxs(self, X):
        preds = []
        idxs = []
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return np.vstack(preds)[np.argsort(idxs)]

class SimNetwork(object):

    def __init__(self, model, iterator, verbose=2):

        self.model = init(model)
        self.iterator = iterator
        self.verbose = verbose

        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        self.X = self.model[0].X

        cos_sim = pair_cosine(y_tr[::4], y_tr[1::4])
        cos_diff = pair_cosine(y_tr[2::4], y_tr[3::4])

        cost = T.mean(T.maximum(0, 1 - cos_sim + cos_diff))

        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X], cost, updates=self.updates)
        self._transform = theano.function([self.X], y_te)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, n_iter=1):
        n = 0.
        t = time()
        costs = []
        for e in range(n_iter):
            epoch_costs = []
            for xmb in self.iterator.train(trX):
                c = self._train(xmb)
                epoch_costs.append(c)
                n += xmb.shape[self.model[0].out_shape.index('x')]/4
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = self.iterator.batches*self.iterator.size*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rIter %d Seen %d samples Avg cost %0.4f Examples per second %d Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)
            n_per_sec = n / (time() - t)
            n_left = self.iterator.batches*self.iterator.size*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f Examples per second %d Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
        return costs

    def infer(self, X):
        self._reset()
        for xmb in self.iterator.predict(X):
            self._infer(xmb)

    def transform(self, X):
        Xt = []
        for xmb in self.iterator.predict(X):
            xt = self._transform(xmb)
            Xt.append(xt)
        return np.vstack(Xt)

class MaskSimNetwork(object):

    def __init__(self, model, iterator, verbose=2):

        self.model = init(model)
        self.iterator = iterator
        self.verbose = verbose

        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})['X']
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})['X']
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})['X']
        self.X = self.model[0].X
        self.mask = self.model[0].mask

        cos_sim = pair_cosine(y_tr[::4], y_tr[1::4])
        cos_diff = pair_cosine(y_tr[2::4], y_tr[3::4])

        cost = T.mean(T.maximum(0, 1 - cos_sim + cos_diff))

        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X, self.mask], cost, updates=self.updates)
        self._transform = theano.function([self.X, self.mask], y_te)
        self._infer = theano.function([self.X, self.mask], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, n_iter=1):
        n = 0.
        t = time()
        costs = []
        for e in range(n_iter):
            epoch_costs = []
            for xmb, mask in self.iterator.train(trX):
                c = self._train(xmb, mask)
                epoch_costs.append(c)
                n += xmb.shape[self.model[0].out_shape.index('x')]/4
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = self.iterator.batches*self.iterator.size*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rIter %d Seen %d samples Avg cost %0.4f Examples per second %d Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)
            n_per_sec = n / (time() - t)
            n_left = self.iterator.batches*self.iterator.size*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f Examples per second %d Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
        return costs

    def infer(self, X):
        self._reset()
        for xmb, mask in self.iterator.predict(X):
            self._infer(xmb, mask)

    def transform(self, X):
        Xt = []
        for xmb, mask in self.iterator.predict(X):
            xt = self._transform(xmb, mask)
            Xt.append(xt)
        return np.vstack(Xt)

class EmbeddingNetwork(object):

    def __init__(self, model, iterator, alpha=0.2, verbose=2):

        self.model = init(model)
        self.iterator = iterator
        self.alpha = alpha
        self.verbose = verbose

        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        self.X = self.model[0].X

        anchor = y_tr[::3]
        positive = y_tr[1::3]
        negative = y_tr[2::3]

        cost = T.mean(euclidean(anchor, positive) - euclidean(anchor, negative) + self.alpha)

        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X], cost, updates=self.updates)
        self._transform = theano.function([self.X], y_te)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, n_iter=1):
        n = 0.
        t = time()
        costs = []
        for e in range(n_iter):
            epoch_costs = []
            for xmb in self.iterator.train(trX):
                c = self._train(xmb)
                epoch_costs.append(c)
                n += xmb.shape[self.model[0].out_shape.index('x')]/3
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = self.iterator.batches*self.iterator.size*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rIter %d Seen %d samples Avg cost %0.4f Examples per second %d Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)
            n_per_sec = n / (time() - t)
            n_left = self.iterator.batches*self.iterator.size*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f Examples per second %d Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
        return costs

    def infer(self, X):
        self._reset()
        for xmb in self.iterator.predict(X):
            self._infer(xmb)

    def transform(self, X):
        Xt = []
        for xmb in self.iterator.predict(X):
            xt = self._transform(xmb)
            Xt.append(xt)
        return np.vstack(Xt)

class MaskNetwork(object):

    def __init__(self, model, cost=None, verbose=2, iterator='linear'):

        if cost is not None:
            self.cost = instantiate(costs, cost)
        else:
            if isinstance(model[-1], ops.Activation):
                if isinstance(model[-1].activation, activations.Sigmoid):
                    self.cost = instantiate(costs, 'bce')
                elif isinstance(model[-1].activation, activations.Softmax):
                    self.cost = instantiate(costs, 'cce')
                else:
                    self.cost = instantiate(costs, 'mse')
            else:
                self.cost = instantiate(costs, 'mse')

        self.verbose = verbose
        self.model = init(model)
        try:
            self.iterator = instantiate(iterators, iterator)
        except:
            self.iterator = instantiate(async_iterators, iterator)

        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})['X']
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})['X']
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})['X']
        self.X = self.model[0].X
        self.mask = self.model[0].mask
        self.Y = T.TensorType(theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        cost = self.cost(self.Y, y_tr)

        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X, self.Y, self.mask], cost, updates=self.updates)
        self._predict = theano.function([self.X, self.mask], y_te)
        self._infer = theano.function([self.X, self.mask], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, trY, n_iter=1):
        out_shape = self.model[-1].out_shape
        n = 0.
        t = time()
        costs = []
        for e in range(n_iter):
            epoch_costs = []
            for xmb, ymb, mask in self.iterator.iterXY(trX, trY):
                c = self._train(xmb, ymb, mask)
                epoch_costs.append(c)
                n += len(ymb)
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = len(trY)*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rIter %d Seen %d samples Avg cost %0.4f Examples per second %d Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)
            n_per_sec = n / (time() - t)
            n_left = len(trY)*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f Examples per second %d Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
        return costs

    def infer_iterator(self, X):
        for xmb, mask in self.iterator.iterX(X):
            self._infer(xmb, mask)

    def infer_idxs(self, X):
        for xmb, mask, idxmb in self.iterator.iterX(X):
            pred = self._infer(xmb, mask)

    def infer(self, X):
        self._reset()
        if isinstance(self.iterator, (iterators.Linear, iterators.MaskLinear)):
            return self.infer_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.infer_idxs(X)
        else:
            raise NotImplementedError

    def predict(self, X):
        if isinstance(self.iterator, (iterators.Linear, iterators.MaskLinear)):
            return self.predict_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.predict_idxs(X)
        else:
            raise NotImplementedError

    def predict_iterator(self, X):
        preds = []
        for xmb, mask in self.iterator.iterX(X):
            pred = self._predict(xmb, mask)
            preds.append(pred)
        return np.vstack(preds)

    def predict_idxs(self, X):
        preds = []
        idxs = []
        for xmb, mask, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb, mask)
            preds.append(pred)
            idxs.extend(idxmb)
        return np.vstack(preds)[np.argsort(idxs)]

class KDNetwork(object):

    def __init__(self, model, temp, weight=0.5, verbose=2, iterator='kdlinear'):

        self.verbose = verbose
        self.model = init(model)
        try:
            self.iterator = instantiate(iterators, iterator)
        except:
            self.iterator = instantiate(async_iterators, iterator)

        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        
        self.X = self.model[0].X

        self.Yhard = T.TensorType(theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        self.Ysoft = T.TensorType(theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        print weight, temp
        cost = weight*costs.cce(self.Yhard, y_tr[0]) + costs.cce(self.Ysoft, y_tr[1])*temp**2

        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)

        self._train = theano.function([self.X, self.Yhard, self.Ysoft], cost, updates=self.updates)
        self._predict = theano.function([self.X], y_te[0])
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, trYh, trYs, n_iter=1):
        out_shape = self.model[-1].out_shape
        n = 0.
        t = time()
        costs = []
        for e in range(n_iter):
            epoch_costs = []
            for xmb, ymbh, ymbs in self.iterator.iterXY(trX, trYh, trYs):
                c = self._train(xmb, ymbh, ymbs)
                epoch_costs.append(c)
                n += len(ymbh)
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = len(trYh)*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rIter %d Seen %d samples Avg cost %0.4f Examples per second %d Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)
            n_per_sec = n / (time() - t)
            n_left = len(trYh)*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f Examples per second %d Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), n_per_sec, time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
        return costs

    def infer_iterator(self, X):
        for xmb in self.iterator.iterX(X):
            self._infer(xmb)

    def infer_idxs(self, X):
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._infer(xmb)

    def infer(self, X):
        self._reset()
        if isinstance(self.iterator, (iterators.KDLinear, iterators.Linear)):
            return self.infer_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.infer_idxs(X)
        else:
            raise NotImplementedError

    def predict(self, X):
        if isinstance(self.iterator, (iterators.KDLinear, iterators.Linear)):
            return self.predict_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.predict_idxs(X)
        else:
            raise NotImplementedError

    def predict_iterator(self, X):
        preds = []
        for xmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
        return np.vstack(preds)

    def predict_idxs(self, X):
        preds = []
        idxs = []
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return np.vstack(preds)[np.argsort(idxs)]