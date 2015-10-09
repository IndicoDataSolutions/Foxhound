import sys
import numpy as np
from time import time
import json
from sklearn import metrics

import theano
import theano.tensor as T

import ops
import costs
import activations
import iterators
import async_iterators
from utils import instantiate

def init(model):
    print model[0].out_shape
    for i in range(1, len(model)):
        # print model[i]
        model[i].connect(model[i-1])
        if hasattr(model[i], 'init'):
            # print 'about to init'
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

    def __init__(self, model, log_path, log_fields=['n_updates', 'n_examples', 'tr_cost', 'te_cost'], log_freq=100, cost=None, verbose=2, iterator='linear'):
        self.n_updates = 0
        self.n_examples = 0
        self.n_epochs = 0
        self.f_log = open(log_path, 'wb')
        self.log_freq = log_freq
        self.log_fields = log_fields
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
        self.uses_batchnorm = any([isinstance(op, ops.BatchNorm) for op in self.model])
        print 'uses batchnorm?', self.uses_batchnorm
        print 'MODEL INIT FINISHED'
        try:
            self.iterator = instantiate(iterators, iterator)
        except:
            self.iterator = instantiate(async_iterators, iterator)

        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        y_mon = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':False})
        self.X = self.model[0].X
        self.Y = T.TensorType(theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        cost = self.cost(self.Y, y_tr)
        mon_cost = self.cost(self.Y, y_mon)

        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X, self.Y], cost, updates=self.updates)
        self._predict = theano.function([self.X], y_te)
        self._score = theano.function([self.X, self.Y], mon_cost)
        self._mon_predict = theano.function([self.X], y_mon)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, trY, vaX, vaY, teX, teY, infX=None, n_iter=1, n_log=100):
        out_shape = self.model[-1].out_shape
        n = 0.
        t = time()
        costs = []
        for e in range(n_iter):
            epoch_costs = []
            for xmb, ymb in self.iterator.iterXY(trX, trY):
                if self.n_updates % self.log_freq == 0:
                    log_data = {}
                    if 'tr_cost' in self.log_fields:
                        log_data['tr_cost'] = float(self.score(vaX, vaY))
                    if 'te_cost' in self.log_fields:
                        log_data['te_cost'] = float(self.score(teX, teY))
                    if 'tr_acc' in self.log_fields:
                        if self.uses_batchnorm:
                            self.infer(infX)
                        va_pred = self.predict(vaX)
                        if self.model[-1].out_shape[-1] == 1:
                            log_data['tr_acc'] = metrics.accuracy_score(vaY, va_pred > 0.5)
                        elif self.model[-1].out_shape[-1] > 1:
                            log_data['tr_acc'] = metrics.accuracy_score(vaY, np.argmax(va_pred, axis=1))
                        else:
                            raise ValueError('Out shape is not standard.')
                    if 'te_acc' in self.log_fields:
                        if self.uses_batchnorm:
                            self.infer(infX)
                        te_pred = self.predict(teX)
                        if self.model[-1].out_shape[-1] == 1:
                            log_data['te_acc'] = metrics.accuracy_score(teY, te_pred > 0.5)
                        elif self.model[-1].out_shape[-1] > 1:
                            log_data['te_acc'] = metrics.accuracy_score(teY, np.argmax(te_pred, axis=1))
                        else:
                            raise ValueError('Out shape is not standard.')
                    if 'n_examples' in self.log_fields:
                        log_data['examples'] = self.n_examples
                    if 'n_updates' in self.log_fields:
                        log_data['updates'] = self.n_updates
                    row = json.dumps(log_data)
                    self.f_log.write(row+'\n')
                    self.f_log.flush()
                c = self._train(xmb, ymb)
                epoch_costs.append(c)
                n += len(ymb)
                self.n_examples += len(ymb)
                self.n_updates += 1
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
            self.n_epochs += 1
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

    def iter_fn(self, fn, X):
        res = []
        for xmb in self.iterator.iterX(X):
            pred = fn(xmb)
            res.append(pred)
        return np.vstack(res)        

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
        return np.vstack(preds)

    def predict_idxs(self, X):
        preds = []
        idxs = []
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return np.vstack(preds)[np.argsort(idxs)]

    def score(self, X, Y):
        scores = []
        for xmb, ymb in self.iterator.iterXY(X, Y):
            score = self._score(xmb, ymb)
            scores.append(score)
        return np.mean(scores)