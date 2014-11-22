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
from foxhound.neural_network import Net
from foxhound.neural_network.layers import Euclidean
from foxhound.utils.vis import grayscale_grid_vis

class KMeans(Net):

    def __init__(self, k=10, *args, **kwargs):
        layers = [Euclidean(size=k), Euclidean]
        Net.__init__(self, layers=layers, cost='kmeans', *args, **kwargs)

    def fit(self, trX, **kwargs):
        super(KMeans, self).fit(trX, **kwargs)

if __name__ == '__main__':
    from foxhound.utils.load import mnist

    trX, teX, trY, teY = mnist()

    kmeans = KMeans(k=10)
    kmeans.fit(trX)
    print np.bincount(kmeans.predict(teX))
    grayscale_grid_vis(kmeans._layers[-1].w.get_value(), transform=lambda x: x.reshape(28, 28))
