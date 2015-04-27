import numpy as np
import theano

from utils import shuffle, iter_data
from theano_utils import floatX, intX

def noop(x):
    return x

def padded(seqs):
    lens = map(len, seqs)
    max_len = max(lens)
    seqs_padded = []
    for seq, seq_len in zip(seqs, lens):
        n_pad = max_len - seq_len 
        seq = [0] * n_pad + seq
        seqs_padded.append(seq)
    return np.asarray(seqs_padded).transpose(1, 0)

class Linear(object):
    """
    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    x_dtype is for casting input data
    y_dtype is for casting target data
    """

    def __init__(self, size=128, shuffle=True, x_dtype=floatX, y_dtype=floatX, trX=noop, teX=noop, trY=noop):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.trX = trX
        self.teX = teX
        self.trY = trY

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb = self.teX(xmb)
            xmb = self.x_dtype(xmb)
            yield xmb

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            xmb = self.trX(xmb)
            ymb = self.trY(ymb)
            xmb = self.x_dtype(xmb)
            ymb = self.y_dtype(ymb)              
            yield xmb, ymb

class SortedPadded(object):

    def __init__(self, size=128, shuffle=True, x_dtype=intX, y_dtype=floatX):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

    def iterX(self, X):
        for x_chunk, chunk_idxs in iter_data(X, np.arange(len(X)), size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            chunk_idxs = [chunk_idxs[idx] for idx in sort]
            for xmb, idxmb in iter_data(x_chunk, chunk_idxs, size=self.size):
                xmb = padded(xmb)
                yield self.x_dtype(xmb), idxmb   

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for x_chunk, y_chunk in iter_data(X, Y, size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            y_chunk = [y_chunk[idx] for idx in sort]
            mb_chunks = [[x_chunk[idx:idx+self.size], y_chunk[idx:idx+self.size]] for idx in range(len(x_chunk))[::self.size]]
            mb_chunks = shuffle(mb_chunks)
            for xmb, ymb in mb_chunks:
                xmb = padded(xmb)
                yield self.x_dtype(xmb), self.y_dtype(ymb)  

# class GPULinear(object):
#     """
#     size is the number of examples per minibatch
#     shuffle controls whether or not the order of examples is shuffled before iterating over
#     x_dtype is for casting input data
#     y_dtype is for casting target data
#     """

#     def __init__(self, chunk_size=128*64, batch_size=128, shuffle=True, x_dtype=floatX, y_dtype=floatX, y_seq=False, train_transform=lambda x:x, test_transform=lambda x:x):
#         self.batch_size = batch_size
#         self.chunk_size = chunk_size
#         self.shuffle = shuffle
#         self.x_dtype = x_dtype
#         self.y_dtype = y_dtype
#         self.y_seq = y_seq
#         self.train_transform = train_transform
#         self.test_transform = test_transform
#         self.X_chunk = theano.shared(floatX(np.asarray(np.zeros((self.chunk_size, 3, 28, 28)))), borrow=True)
#         self.Y_chunk = theano.shared(floatX(np.asarray(np.zeros((self.chunk_size, 1000)))), borrow=True)

#     def iterX(self, X):

#         for x_chunk in iter_data(X, size=self.chunk_size):
#             n = len(x_chunk)
#             self.X_chunk.set_value(self.x_dtype(self.test_transform(x_chunk)))
#             for idx in range(n/self.batch_size):
#                 yield idx

#     def iterXY(self, X, Y):
        
#         if self.shuffle:
#             X, Y = shuffle(X, Y)

#         for x_chunk, y_chunk in iter_data(X, Y, size=self.chunk_size):
#             n = len(x_chunk)
#             self.X_chunk.set_value(self.x_dtype(self.test_transform(x_chunk)))
#             self.Y_chunk.set_value(self.y_dtype(y_chunk))
#             for idx in range(n/self.batch_size):
#                 yield idx
