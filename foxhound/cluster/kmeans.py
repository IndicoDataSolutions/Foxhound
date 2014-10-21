import theano
import theano.tensor as T
import numpy as np
import random

from foxhound.utils.distances import euclidean
from foxhound.utils.vis import grayscale_grid_vis
from foxhound.utils import sharedX, floatX, shuffle, iter_data
from foxhound.utils.updates import SGD,Adadelta
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

class KMeans(object):

    def __init__(self, k=10, epochs=100, batch_size=128, max_gpu_mem=1e8):
        self.k = k
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_gpu_mem = max_gpu_mem

    def fit(self, X):
        X = np.atleast_2d(floatX(X))
        Xnp = np.copy(X)
        # self.n_batches_gpu = np.floor(self.max_gpu_mem / X[0].size / self.batch_size) 
        # self.chunk_size = self.n_batches_gpu * self.batch_size
        # print self.n_batches_gpu, self.chunk_size
        # self.gpuX = sharedX(X[:self.chunk_size])
        self.centers = sharedX(random.sample(X, self.k))
        self.counts = sharedX(np.zeros((self.k)))
        self.train = self._train()
        for e in range(self.epochs):
            X = shuffle(X)
            for batch in iter_data(X, size=self.batch_size):
                self.train(batch)
                plt.scatter(Xnp[:,0], Xnp[:,1])
                centers_np = self.centers.get_value()
                plt.scatter(centers_np[:,0], centers_np[:,1], s=100, c='r')
                plt.show()


    # def _train(self):
    #     X = T.fmatrix()
    #     dist = euclidean(self.centers, X)
    #     nearest = T.argmin(dist, axis=0)
    #     self.counts += T.extra_ops.bincount(nearest, minlength=self.k).astype(theano.config.floatX)
    #     mu = (1 / self.counts).dimshuffle(0,'x')
    #     updates = [[self.centers, (1-mu)*self.centers + mu*X]]
    #     return theano.function([X],None,updates=updates)

    def _train(self):
        X = T.fmatrix()
        dist = euclidean(self.centers, X)
        weighted = 1. / dist
        vector = self.centers.dimshuffle(0,'x',1)-X
        vector = vector / T.sqrt(T.sum(T.sqr(vector), axis=2)).dimshuffle(0,1,'x')
        vector = vector * weighted.dimshuffle(0,1,'x')
        updates = [[self.centers, self.centers - T.mean(vector, axis=1)]]
        return theano.function([X],None,updates=updates)

    # def _train(self):

if __name__ == '__main__':
    from foxhound.utils.load import mnist

    # trX, teX, trY, teY = mnist()
    X, Y = make_blobs()
    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.show()

    kmeans = KMeans(k=3, batch_size=33)
    kmeans.fit(X)

    # X = trX[:128]
    # centers = trX[-10:]
    # dist = cdist(centers, X)
    # weighted = 1. / dist
    # vector = centers[:,np.newaxis,:]-X
    # vector = vector / np.sqrt(np.sum(np.square(vector), axis=2))[:,:,np.newaxis]
    # vector = vector * weighted[:,:,np.newaxis]
    # update = centers - np.mean(vector, axis=1)
    # print 'did'