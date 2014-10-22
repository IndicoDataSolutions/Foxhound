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
from foxhound.utils import sharedX, floatX, shuffle, iter_data
from foxhound.utils.updates import SGD,Adadelta, NAG
from foxhound.utils.costs import MSE, MAE
from foxhound.utils.vis import grayscale_grid_vis

class KMeans(object):

    def __init__(self, k=10, epochs=10, batch_size=128, max_gpu_mem=1e8):
        self.k = k
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_gpu_mem = max_gpu_mem

    def fit(self, X):
        X = np.atleast_2d(floatX(X))
        # self.n_batches_gpu = np.floor(self.max_gpu_mem / X[0].size / self.batch_size) 
        # self.chunk_size = self.n_batches_gpu * self.batch_size
        # print self.n_batches_gpu, self.chunk_size
        # self.gpuX = sharedX(X[:self.chunk_size])
        self.centers = sharedX(random.sample(X, self.k))
        self.counts = sharedX(np.zeros((self.k)))
        self.setup()
        for e in range(self.epochs):
            X = shuffle(X)
            for batch in iter_data(X, size=self.batch_size):
                err = self.train(batch)

    def setup(self):
        X = T.matrix()
        out = euclidean(X, self.centers)
        err = MSE(0, T.min(out, axis=1))
        params = [self.centers]
        grads = T.grad(err, params)
        updates = NAG(params, grads, lr = 0.01, momentum = 0.9)
        self.predict = theano.function([X], T.argmin(out, axis=1))
        self.train = theano.function([X], err, updates=updates)

if __name__ == '__main__':
    from foxhound.utils.load import mnist

    trX, teX, trY, teY = mnist()

    kmeans = KMeans(k=10, batch_size=128)
    kmeans.fit(trX)