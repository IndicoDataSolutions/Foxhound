import sys
import numpy as np
from time import time

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import ops
import costs
import activations
import iterators
from utils import instantiate
from preprocessing import standardize_X, standardize_Y

def init(model):
    print model[0].out_shape
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

class Network(object):

    def __init__(self, model, cost=None, verbose=2, iterator='linear', seed=42):

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
        self.seed = seed
        self.model = init(model)
        self.iterator = instantiate(iterators, iterator)

        t_rng = RandomStreams(self.seed)
        y_tr = self.model[-1].op({'t_rng':t_rng, 'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'t_rng':t_rng, 'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'t_rng':t_rng, 'dropout':False, 'bn_active':True, 'infer':True})
        self.X = self.model[0].X
        self.Y = T.TensorType(theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        cost = self.cost(self.Y, y_tr)

        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X, self.Y], cost, updates=self.updates)
        self._predict = theano.function([self.X], y_te)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, trY, n_iter=1):
        out_shape = self.model[-1].out_shape
        trY = standardize_Y(out_shape, trY)
        n = 0.
        t = time()
        costs = []
        for e in range(n_iter):
            epoch_costs = []
            for xmb, ymb in self.iterator.iterXY(trX, trY):
                c = self._train(xmb, ymb)
                epoch_costs.append(c)
                n += len(ymb)
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = len(trY)*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rIter %d Seen %d samples Avg cost %0.4f Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)

            status = "Iter %d Seen %d samples Avg cost %0.4f Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status
        return costs

    def infer(self, X):
        self._reset()
        for xmb in self.iterator.iterX(X):
            self._infer(xmb)

    def predict(self, X, argmax=True):
        preds = []
        for xmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
        preds = np.vstack(preds)
        if argmax:
            preds = np.argmax(preds, axis=1)
        return preds

    def predict_proba(self, X):
        return self.predict(X, argmax=False)