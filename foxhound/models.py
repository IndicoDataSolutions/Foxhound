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
        else:
            print 'no update op'
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

        self.verbose = verbose
        self.seed = seed
        self.model = init(model)
        self.iterator = instantiate(iterators, iterator)

        t_rng = RandomStreams(self.seed)
        y_tr = self.model[-1].op({'t_rng':t_rng, 'dropout':True})
        y_te = self.model[-1].op({'t_rng':t_rng, 'dropout':False})
        
        self.X = self.model[0].X
        self.Y = T.TensorType(theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        cost = self.cost(self.Y, y_tr)

        self.updates = collect_updates(self.model, cost)
        self._train = theano.function([self.X, self.Y], cost, updates=self.updates)
        self._predict = theano.function([self.X], y_te)

    def fit(self, trX, trY, n_iter=1):
        trX = standardize_X(self.model[0].out_shape, trX)
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
        print
        return costs

    def predict(self, X, argmax=True):
        X = standardize_X(self.model[0].out_shape, X)
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