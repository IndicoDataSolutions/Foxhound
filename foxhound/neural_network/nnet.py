from time import time
from itertools import izip

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib

from foxhound.utils import floatX, sharedX, shuffle, iter_data, iter_indices
from foxhound.utils.load import mnist
from foxhound.neural_network.layers import Dense, Input
from foxhound.utils import config
import foxhound.utils.gpu as gpu
from foxhound.utils.activations import cost_mapping
from foxhound.utils import updates, costs, case_insensitive_import


class Net(object):

    def __init__(self, layers, n_epochs=100, cost=None, update='adadelta', regularizer=None, 
                 max_gpu_mem=config.max_gpu_mem, batch_size=128):
        self._layers = layers
        self._values = False
        self.n_epochs = n_epochs
        self.max_gpu_mem = max_gpu_mem
        self.batch_size = batch_size

        if not cost:
            try:
                cost = cost_mapping[layers[-1].activation]
            except KeyError:
                raise ValueError(
                    "Model must be initialized with a valid cost function."
                )

        if isinstance(cost, basestring):
            self.cost_fn = case_insensitive_import(costs, cost)
            self.cost_fn = self.cost_fn()
        else:
            self.cost_fn = cost

        if isinstance(update, basestring):
            self.update_fn = case_insensitive_import(updates, update)()
        else:
            self.update_fn = update

        if regularizer:
            self.update_fn.regularizer = regularizer


    def setup(self, X, Y=None):
        dataX = floatX(X)
        dataY = floatX(Y)

        self.chunk_size = gpu.n_chunks(self.max_gpu_mem, dataX)
        self.gpuX = sharedX(dataX[:self.chunk_size])
        if dataY is not None:
            self.gpuY = theano.shared(dataY[:self.chunk_size])

        if isinstance(self._layers[0], Input):
            self.layers = [self._layers[0]]
            self._layers = self._layers[1:]
        else:
            self.layers = [Input(shape=dataX.shape[1:])]

        for i, layer in enumerate(self._layers):
            if dataY is not None and i == len(self._layers) - 1:
                layer.size = dataY.shape[1]
            layer.setup(self.layers[-1], dataX, dataY)
            self.layers.append(layer)


        tr_out = self.layers[-1].output(dropout_active=True)
        te_out = self.layers[-1].output(dropout_active=False)
        te_pre_act = self.layers[-1].preactivation(dropout_active=False)
        X = self.layers[0].X
        cost = self.cost_fn.get_cost(tr_out)
        self.params = self.get_params()
        self.update_params()
        updates = self.update_fn.get_updates(self.params, cost)

        idx = T.lscalar('idx')
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        givens = {
            X : self.gpuX[start:end],
        }

        if dataY is not None:
            givens[self.cost_fn.target] = self.gpuY[start:end]

        try:
            self._train = theano.function(
                [idx], cost, updates=updates, givens=givens, allow_input_downcast=True
            )
            self._cost = theano.function(
                [idx], cost, givens=givens, allow_input_downcast=True
            )
        except theano.gof.fg.MissingInputError:
            pass

        self._predict = theano.function(
            [idx], te_out, givens=givens, allow_input_downcast=True, on_unused_input='ignore'
        )
        self._predict_pre_act = theano.function(
            [idx], te_pre_act, givens=givens, allow_input_downcast=True, on_unused_input='ignore'
        )

    def fit(self, trX, trY=None, teX=None, teY=None):
        self.setup(trX, trY)

        data = [trX]
        if trY is not None:
            data.append(trY)
            
        print "Total Epochs:", self.n_epochs
        for e in range(self.n_epochs):
            print "Epoch:", e
            for batch_idx in self.batches(*data):
                cost = self._train(batch_idx)

    def predict_proba(self, X):
        return np.vstack(
            [self._predict(idx) for idx in self.batches(X)]
        )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_pre_act(self, X):
        return np.vstack(
            [self._predict_pre_act(idx) for idx in self.batches(X)]
        )

    def batches(self, *args):
        args = tuple([floatX(arg) for arg in args])
        self.setup(*args)
        for chunk in iter_data(*args, size=self.chunk_size):
            if len(chunk) is 1:
                X = chunk[0]
                self.gpuX.set_value(X)
            elif len(chunk) is 2:
                X, Y = chunk
                self.gpuX.set_value(X)
                self.gpuY.set_value(Y)
            for batch_idx in iter_indices(X, size=self.batch_size):
                yield batch_idx

    def get_params(self):
        params = []
        layer = self.layers[-1]
        while not hasattr(layer, 'X'):
            if hasattr(layer, 'params'):
                params.extend(layer.params)
            layer = layer.l_in
        return params

    def __repr__(self):
        template = "%s(\n  %s\n)"
        layer_str = ",\n  ".join([str(layer) for layer in self.layers])
        return template % (self.__class__.__name__, layer_str)

    def save(self, filename):
        values = [p.get_value() for p in self.params]
        joblib.dump(values, filename, compress=3)

    def load(self, filename):
        self._values = joblib.load(filename)

    def update_params(self):
        if self._values:
            for p, v in izip(self.params, self._values):
                p.set_value(v)
            self._values = False

if __name__ == '__main__':
    trX, teX, trY, teY = mnist(onehot=True)

    layers = [
        Input(shape=trX[0].shape),
        Dense(size=512),
        Dense(size=512),
        Dense(activation='softmax')
    ]

    update = updates.Adadelta(regularizer=updates.Regularizer(l1=1.0))
    model = Net(layers=layers, cost='cce', update='rmsprop', n_epochs=1)
    model.fit(trX, trY)
    model.save("save.pkl")

    # print metrics.accuracy_score(np.argmax(teY, axis=1), model.predict(teX))
