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

    def __init__(self, size=64, shuffle=True, x_dtype=floatX, y_dtype=floatX, y_seq=False):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.y_seq = y_seq

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb = self.x_dtype(xmb)
            yield xmb

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            xmb = self.x_dtype(xmb)
            ymb = self.y_dtype(ymb)               
            yield xmb, ymb