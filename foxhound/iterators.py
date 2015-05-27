import numpy as np
import theano

from transforms import SeqPadded
from utils import shuffle, iter_data
from theano_utils import floatX, intX
from rng import py_rng, np_rng

class Linear(object):
    """
    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    """

    def __init__(self, size=128, shuffle=True, trXt=floatX, teXt=floatX, trYt=floatX):
        self.size = size
        self.shuffle = shuffle
        self.trXt = trXt
        self.teXt = teXt
        self.trYt = trYt

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb = self.teXt(xmb)
            yield xmb

    def iterXY(self, X, Y):

        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            xmb = self.trXt(xmb)
            ymb = self.trYt(ymb)            
            yield xmb, ymb

class GeneralizedLinear(object):
    """
    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    """

    def __init__(self, size=128, shuffle=True, trXt=[floatX], teXt=[floatX], trYt=[floatX]):
        self.size = size
        self.shuffle = shuffle
        self.trXt = trXt
        self.teXt = teXt
        self.trYt = trYt

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb = self.teXt(xmb)
            yield xmb

    def iterXY(self, X, Y):

        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            xmb = self.trXt(xmb)
            ymb = self.trYt(ymb)            
            yield xmb, ymb

class KDLinear(object):
    """
    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    """

    def __init__(self, size=128, shuffle=True, trXt=floatX, teXt=floatX, trYth=floatX, trYts=floatX):
        self.size = size
        self.shuffle = shuffle
        self.trXt = trXt
        self.teXt = teXt
        self.trYth = trYth
        self.trYts = trYts

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb = self.teXt(xmb)
            yield xmb

    def iterXY(self, X, Yh, Ys):

        if self.shuffle:
            X, Yh, Ys = shuffle(X, Yh, Ys)

        for xmb, ymbh, ymbs in iter_data(X, Yh, Ys, size=self.size):
            xmb = self.trXt(xmb)
            ymbh = self.trYth(ymbh)    
            ymbs = self.trYts(ymbs)          
            yield xmb, ymbh, ymbs

class MaskLinear(object):
    """
    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    """

    def __init__(self, tr_mask, te_mask, size=128, shuffle=True, trXt=floatX, teXt=floatX, trYt=floatX):
        self.size = size
        self.shuffle = shuffle
        self.tr_mask = tr_mask
        self.te_mask = te_mask
        self.trXt = trXt
        self.teXt = teXt
        self.trYt = trYt

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            mask = self.te_mask(xmb)
            xmb = self.teXt(xmb)
            yield xmb, mask

    def iterXY(self, X, Y):

        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            mask = self.tr_mask(xmb)
            xmb = self.trXt(xmb)
            ymb = self.trYt(ymb)            
            yield xmb, ymb, mask

class SortedPadded(object):

    def __init__(self, size=128, shuffle=True, trXt=floatX, teXt=floatX, trYt=floatX):
        self.size = size
        self.shuffle = shuffle
        self.trXt = trXt
        self.teXt = teXt
        self.trYt = trYt

    def iterX(self, X):
        for x_chunk, chunk_idxs in iter_data(X, np.arange(len(X)), size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            chunk_idxs = [chunk_idxs[idx] for idx in sort]
            for xmb, idxmb in iter_data(x_chunk, chunk_idxs, size=self.size):
                xmb = self.teXt(xmb)
                yield xmb, idxmb   

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for x_chunk, y_chunk in iter_data(X, Y, size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            y_chunk = [y_chunk[idx] for idx in sort]
            mb_chunks = [[x_chunk[idx:idx+self.size], y_chunk[idx:idx+self.size]] for idx in range(len(x_chunk))[::self.size]]
            py_rng.shuffle(mb_chunks)
            for xmb, ymb in mb_chunks:
                xmb = self.trXt(xmb)
                ymb = self.trYt(ymb)
                yield xmb, ymb

class Sampler(object):

    def __init__(self, sampler, batches=128, size=128, trXt=floatX, teXt=floatX):
            self.sampler = sampler
            self.batches = batches
            self.size = size
            self.trXt = trXt
            self.teXt = teXt

    def predict(self, X):
        for xmb in iter_data(X, size=self.size):
            xmb = self.teXt(xmb)
            yield xmb

    def train(self, data):
        for batch in range(self.batches):
            xmb = self.sampler(data, size=self.size)
            xmb = self.trXt(xmb)
            yield xmb

class MaskSampler(object):

    def __init__(self, tr_mask, te_mask, sampler, batches=128, size=128, trXt=floatX, teXt=floatX):
            self.sampler = sampler
            self.batches = batches
            self.size = size
            self.tr_mask = tr_mask
            self.te_mask = te_mask
            self.trXt = trXt
            self.teXt = teXt

    def predict(self, X):
        for xmb in iter_data(X, size=self.size):
            mask = self.te_mask(xmb)
            xmb = self.teXt(xmb)
            yield xmb, mask

    def train(self, data):
        for batch in range(self.batches):
            xmb = self.sampler(data, size=self.size)
            mask = self.tr_mask(xmb)
            xmb = self.trXt(xmb)          
            yield xmb, mask