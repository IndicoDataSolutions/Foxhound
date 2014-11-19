import theano
import theano.tensor as T
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn import metrics

from foxhound.utils import iter_data
from foxhound.utils.distances import euclidean
from foxhound.utils.vis import grayscale_grid_vis
from foxhound.utils import sharedX, floatX, shuffle, iter_data, iter_indices
from foxhound.utils.updates import SGD,Adadelta, NAG
from foxhound.utils.costs import MSE, MAE
from foxhound.utils.vis import grayscale_grid_vis
import foxhound.utils.config as config
import foxhound.utils.gpu as gpu

class KMeans(object):

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, epochs=10, batch_size=128, max_gpu_mem=config.max_gpu_mem):
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_gpu_mem = max_gpu_mem

        X = np.atleast_2d(floatX(X))

        self.centers = sharedX(random.sample(X, self.k))
        self.counts = sharedX(np.zeros((self.k)))
        self.setup(X)

        for e in range(self.epochs):
            for chunk in iter_data(X, size=self.chunk_size):
                self.gpuX.set_value(chunk)
                for batch_index in iter_indices(chunk, size=self.batch_size):
                    err = self._train(batch_index)

    def setup(self, data):

        # calculate chunk sizes
        self.chunk_size = gpu.n_chunks(self.max_gpu_mem, data) 
        self.gpuX = sharedX(data[:self.chunk_size])

        X = T.matrix()
        out = euclidean(X, self.centers)
        err = MSE(0, T.min(out, axis=1))
        params = [self.centers]
        grads = T.grad(err, params)
        updates = NAG(params, grads, lr = 0.01, momentum = 0.9)

        idx = T.lscalar('idx')
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        givens = {
            X: self.gpuX[start:end]
        }

        self._predict = theano.function([idx], T.argmin(out, axis=1), givens=givens)
        self._train = theano.function([idx], err, givens=givens, updates=updates)

    def predict(self, X):
        X = np.atleast_2d(floatX(X))

        results = []
        for chunk in iter_data(X, size=self.chunk_size):
            self.gpuX.set_value(chunk)
            for batch_index in iter_indices(chunk, size=self.batch_size):
                preds = self._predict(batch_index)
                if len(preds.shape) is 1:
                    preds = preds.reshape(-1, 1)
                results.append(preds)
        return np.vstack(results)

if __name__ == '__main__':
    from foxhound.utils.load import mnist

    trX, teX, trY, teY = mnist()

    kmeans = KMeans(k=10)
    kmeans.fit(trX)
    print kmeans.predict(teX)
