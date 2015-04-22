import numpy as np

from utils import shuffle, iter_data
from theano_utils import floatX, intX

class Linear(object):
    """
    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    x_dtype is for casting input data
    y_dtype is for casting target data
    """

    def __init__(self, size=128, shuffle=True, x_dtype=floatX, y_dtype=floatX, y_seq=False, train_transform=None, test_transform=None):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.y_seq = y_seq
        self.train_transform = train_transform
        self.test_transform = test_transform

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            if self.test_transform is not None:
                xmb = self.test_transform(xmb)
            xmb = self.x_dtype(xmb)
            yield xmb

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            # print np.asarray(xmb).shape, np.asarray(ymb).shape
            if self.train_transform is not None:
                xmb = self.train_transform(xmb)
            xmb = self.x_dtype(xmb)
            ymb = self.y_dtype(ymb)               
            yield xmb, ymb