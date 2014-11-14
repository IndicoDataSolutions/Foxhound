import theano
import theano.tensor as T

rectify = lambda x: (x + abs(x)) / 2.0
tanh = T.tanh
softmax = T.nnet.softmax
sigmoid = T.nnet.sigmoid