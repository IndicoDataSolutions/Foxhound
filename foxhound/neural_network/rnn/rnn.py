import numpy as np
import theano
import theano.tensor as T
from sklearn.cross_validation import train_test_split
import unicodedata
from locale import getpreferredencoding
from collections import Counter
import random
from time import time

from foxhound.utils import floatX, sharedX
from foxhound.utils.updates import Adadelta

def one_hot(X, n):
    Xoh = np.zeros((len(X), n))
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def encode(X, encoder=None, onehot=False):
    if encoder is None:
        c = Counter(X)
        encoder = dict(zip(c.keys(), range(len(c.keys()))))
    X = [encoder[c] for c in X]
    if onehot:
        X = one_hot(X, len(encoder))
    return X, encoder

def sample_weights_classic(sizeX, sizeY, sparsity=-1, scale=0.01):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = np.minimum(sizeY, sparsity)
    sparsity = np.minimum(sizeY, sparsity)
    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = np.random.permutation(sizeY)
        new_vals = np.random.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
    return values.astype(theano.config.floatX)

def sample_weights(sizeX, sizeY, sparsity=-1, scale=0.01):
    """
    Initialization that fixes the largest singular value.
    """
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = np.minimum(sizeY, sparsity)
    sparsity = np.minimum(sizeY, sparsity)
    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = np.random.permutation(sizeY)
        new_vals = np.random.uniform(low=-scale, high=scale, size=(sparsity,))
        vals_norm = np.sqrt((new_vals**2).sum())
        new_vals = scale*new_vals/vals_norm
        values[dx, perm[:sparsity]] = new_vals
    _,v,_ = np.linalg.svd(values)
    values = scale * values/v[0]
    return values.astype(theano.config.floatX)

class SeqRNN(object):

    def __init__(self, nh=512, seq_len=100, batch_size=128, n_epochs=1000, n_batches=128):
        self.nh = nh
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.seq_len = seq_len

    def _init(self, X):
        x, self.encoder = encode(X)
        self.decoder = dict(zip(self.encoder.values(), self.encoder.keys()))

        self.nin = len(self.encoder)
        self.nout = self.nin

        X = T.tensor3()
        Y = T.tensor3()
        h0 = T.matrix()

        self.w_i = sharedX(sample_weights_classic(self.nin, self.nh))
        self.w_h = sharedX(sample_weights(self.nh, self.nh))
        self.w_o = sharedX(sample_weights_classic(self.nh, self.nout))

        def step(x_t, h_tm1, w_i, w_h, w_o):
          h_t = T.tanh(T.dot(x_t, w_i) + T.dot(h_tm1, w_h))
          y_t = T.nnet.softmax(T.dot(h_t, w_o))
          return h_t, y_t

        [h, y], _ = theano.scan(step,
                        sequences=X,
                        outputs_info=[h0, None],
                        non_sequences=[self.w_i, self.w_h, self.w_o])

        params = [self.w_i, self.w_h, self.w_o]

        # self.W = sharedX(sample_weights_classic(self.nin,self.nh))
        # self.W_r = sharedX(sample_weights_classic(self.nin,self.nh))
        # self.W_z = sharedX(sample_weights_classic(self.nin,self.nh))

        # self.U = sharedX(sample_weights(self.nh, self.nh))
        # self.Ub = sharedX(floatX(np.zeros((self.nh))))

        # self.U_r = sharedX(sample_weights(self.nh, self.nh))
        # self.Ub_r = sharedX(floatX(np.zeros((self.nh))))

        # self.U_z = sharedX(sample_weights(self.nh, self.nh))
        # self.Ub_z = sharedX(floatX(np.zeros((self.nh))))

        # self.W_out = sharedX(sample_weights_classic(self.nh, self.nout))
        # self.b_out = sharedX(floatX(np.zeros((self.nout))))

        # def step(xt, htm1, U, Ub, U_z, Ub_z, U_r, Ub_r, W, W_z, W_r, W_out, b_out):
        #     z = T.nnet.sigmoid(T.dot(xt, W_z) + T.dot(htm1, U_z) + Ub_z)
        #     r = T.nnet.sigmoid(T.dot(xt, W_r) + T.dot(htm1, U_r) + Ub_r)
        #     h_tilda_t = T.tanh(T.dot(xt, W) + T.dot(r*htm1, U) + Ub)
        #     h_t = z*htm1+(1-z)*h_tilda_t
        #     y_t = T.nnet.softmax(T.dot(h_t, W_out) + b_out)
        #     return h_t, y_t

        # [h, y], _ = theano.scan(step,
        #                 sequences=X,
        #                 outputs_info=[h0, None],
        #                 non_sequences=[self.U, self.Ub, self.U_z, self.Ub_z, self.U_r, self.Ub_r, self.W, self.W_z, self.W_r, self.W_out, self.b_out])

        # params = [self.U, self.Ub, self.U_z, self.Ub_z, self.U_r, self.Ub_r, self.W, self.W_z, self.W_r, self.W_out, self.b_out]

        cost = T.nnet.categorical_crossentropy(y, Y).mean()
        grads = T.grad(cost, params)
        updates = Adadelta(params, grads)

        self._fit = theano.function([X, Y, h0], cost, updates=updates)
        self._cost = theano.function([X, Y, h0], cost)
        self._predict = theano.function([X, h0], [h, y])
        return x

    def minibatch(self, X, size):
        xmb = []
        ymb = []
        idxs = np.random.randint(0, len(X), size=size)
        for i in idxs:
            j = random.randint(0, len(X[i])-self.seq_len-1)
            xmb.append(one_hot(X[i][j:j+self.seq_len], self.nin))
            ymb.append(one_hot(X[i][j+1:j+1+self.seq_len], self.nin))
        xmb = np.asarray(xmb, dtype=theano.config.floatX).transpose(1, 0, 2)
        ymb = np.asarray(ymb, dtype=theano.config.floatX).transpose(1, 0, 2)
        return xmb, ymb

    def fit(self, X):
        X = self._init(X)
        trX = X[:-1000000]
        teX = X[-1000000:]
        trX = [trX]
        teX = [teX]
        trX_eval, trY_eval = self.minibatch(trX, size=1024)
        teX_eval, teY_eval = self.minibatch(teX, size=1024)
        h0tr = np.zeros((self.batch_size, self.nh), dtype=theano.config.floatX)
        h0te = np.zeros((1024, self.nh), dtype=theano.config.floatX)

        t = time()
        nc = 0.
        for e in range(self.n_epochs):
            for b in range(self.n_batches):
                xmb, ymb = self.minibatch(trX, size=self.batch_size)
                cost = self._fit(xmb, ymb, h0tr)
                nc += self.seq_len*self.batch_size
            tr_cost = self._cost(trX_eval, trY_eval, h0te)
            te_cost = self._cost(teX_eval, teY_eval, h0te)
            print e, tr_cost, te_cost, nc/(time()-t)

trX = np.load('/media/datasets/wiki_text/wiki_letters_2G')[0]
print len(trX)
trX = trX[:10000000]
print len(trX)
trX = unicodedata.normalize("NFKD", unicode(trX, getpreferredencoding())).encode("ascii", errors="ignore")

model = SeqRNN()
model.fit(trX)