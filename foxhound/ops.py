import theano
import theano.tensor as T
import inits
import activations
import updates
import numpy as np
from theano.tensor.extra_ops import repeat
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.dnn import dnn_conv

from utils import instantiate
from theano_utils import shared0s

def same_padding(n):
    return int(np.floor(n / 2.))

class Input(object):

    def __init__(self, shape):
        self.X = T.TensorType(theano.config.floatX, (False,)*(len(shape)))()
        print self.X.type
        self.out_shape = shape

    def op(self, state):
        return self.X

class Flatten(object):

    def __init__(self, dim):
        self.dim = dim

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        if self.dim == len(self.in_shape):
            self.out_shape = self.in_shape
        else:
            self.out_shape = self.in_shape[:self.dim-1] + [np.prod(self.in_shape[self.dim-1:])]
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return T.flatten(X, outdim=self.dim)

class MaxPool(object):
    def __init__(self, pool_size):
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self.pool_size = pool_size

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.in_shape[1],
            int(np.ceil(float(self.in_shape[2]) / self.pool_size[0])),
            int(np.ceil(float(self.in_shape[3]) / self.pool_size[1]))
        ]
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return max_pool_2d(X, self.pool_size)

class Conv(object):

    def __init__(self, n_filters=32, filter_shape=(3, 3), padding='same', stride=(1, 1), init_fn='orthogonal', update_fn='nag'):
        self.n_filters = n_filters

        if isinstance(filter_shape, int):
            filter_shape = (filter_shape, filter_shape)
        self.filter_shape = filter_shape

        if isinstance(padding, int):
            padding = (padding, padding) 
        elif padding == 'same':
            padding = (same_padding(filter_shape[0]), same_padding(filter_shape[1]))
        self.padding = padding

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.n_filters, 
            (self.in_shape[2] - self.filter_shape[0] + self.padding[0] * 2)/self.stride[0] + 1, 
            (self.in_shape[3] - self.filter_shape[1] + self.padding[1] * 2)/self.stride[1] + 1
        ]
        print self.out_shape

    def init(self):
        self.w = self.init_fn((self.n_filters, self.in_shape[1], self.filter_shape[0], self.filter_shape[1]))
        self.params = [self.w]

    def op(self, state):
        X = self.l_in.op(state=state)
        return dnn_conv(X, self.w, subsample=self.stride, border_mode=self.padding)

    def update(self, cost):
        return self.update_fn(self.params, cost)

class CPUConv(object):

    def __init__(self, n_filters=32, filter_shape=(3, 3), padding='same', stride=(1, 1), init_fn='orthogonal', update_fn='nag'):
        self.n_filters = n_filters

        if isinstance(filter_shape, int):
            filter_shape = (filter_shape, filter_shape)
        self.filter_shape = filter_shape

        if padding != 'same':
            raise NotImplementedError('Only same padding supported right now!')
        self.padding = padding

        if isinstance(stride, int):
            stride = (stride, stride)
        if stride != (1, 1):
            raise NotImplementedError('Only (1, 1) stride supported right now!')
        self.stride = stride

        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.n_filters, 
            self.in_shape[2], 
            self.in_shape[3]
        ]
        print self.out_shape

    def init(self):
        self.w = self.init_fn((self.n_filters, self.in_shape[1], self.filter_shape[0], self.filter_shape[1]))
        self.params = [self.w]

    def op(self, state):
        """ Benanne lasange same for cpu """
        X = self.l_in.op(state=state)
        out = T.nnet.conv2d(X, self.w, subsample=self.stride, border_mode='full')
        shift_x = (self.filter_shape[0] - 1) // 2
        shift_y = (self.filter_shape[1] - 1) // 2
        return out[:, :, shift_x:self.out_shape[2] + shift_x, shift_y:self.out_shape[3] + shift_y]

    def update(self, cost):
        return self.update_fn(self.params, cost)

class Project(object):

    def __init__(self, dim=256, init_fn='orthogonal', update_fn='nag'):
        self.dim = dim
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = [self.in_shape[0], self.dim]
        print self.out_shape

    def init(self):
        self.w = self.init_fn((self.in_shape[-1], self.out_shape[-1]))
        self.params = [self.w]

    def op(self, state):
        X = self.l_in.op(state=state)
        return T.dot(X, self.w)

    def update(self, cost):
        return self.update_fn(self.params, cost)

class Dropout(object):

    def __init__(self, p_drop=0.5):
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        retain_prob = 1 - self.p_drop
        t_rng = state['t_rng']   
        if state['dropout']:
            X = X / retain_prob * t_rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        return X

class Shift(object):

    def __init__(self, init_fn='constant', update_fn='nag'):
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape
        n_dim_in = len(l_in.out_shape)
        if n_dim_in == 4:
            self.conv = True
        elif n_dim_in == 2:
            self.conv = False
        else:
            raise NotImplementedError

    def init(self):
        if self.conv:
            self.b = self.init_fn(self.out_shape[1])
        else:
            self.b = self.init_fn(self.out_shape[-1])

        self.params = [self.b]

    def op(self, state):
        X = self.l_in.op(state=state)
        if self.conv:
            return X + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            return X + self.b  

    def update(self, cost):
        return self.update_fn(self.params, cost)

class Activation(object):

    def __init__(self, activation, init_fn=inits.Constant(c=0.25), update_fn='nag'):
        self.activation = instantiate(activations, activation)
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape

    def init(self):
        self.params = []

    def op(self, state):
        X = self.l_in.op(state=state)
        return self.activation(X)

    def update(self, cost):
        return self.update_fn(self.params, cost)

class BatchNormalize(object):

    def __init__(self, update_fn='nag', e=1e-8):
        self.update_fn = instantiate(updates, update_fn)
        self.e = e

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape

    def init(self):
        self.g = inits.Constant(c=1.)(self.out_shape[-1])
        self.b = inits.Constant(c=0.)(self.out_shape[-1])
        self.params = [self.g, self.b]

    def op(self, state):
        X = self.l_in.op(state=state)

        u = T.mean(X, axis=0)
        s = T.mean(T.sqr(X - u), axis=0)
        X = (X - u) / T.sqrt(s + self.e)
        return self.g*X + self.b

    def update(self, cost):
        return self.update_fn(self.params, cost)

class Dimshuffle(object):

    def __init__(self, shuffle):
        self.shuffle = shuffle

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = [self.in_shape[idx] for idx in self.shuffle]
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return X.dimshuffle(*self.shuffle)

class Slice(object):

    def __init__(self, fn=lambda x:x[-1], shape_fn=lambda x:x[1:]):
        self.fn = fn
        self.shape_fn = shape_fn

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.shape_fn(self.in_shape)
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return self.fn(X) 

# class Last(object):
#     def __init__(self):
#         pass

#     def connect(self, l_in):
#         self.l_in = l_in
#         self.in_shape = l_in.out_shape
#         self.out_shape = 

class RNN(object):

    def __init__(self, dim=256, activation='rectify', proj_init_fn='orthogonal', rec_init_fn='identity',
                 bias_init_fn='constant', update_fn='nag'):
        self.dim = dim
        self.activation = instantiate(activations, activation)
        self.proj_init_fn = instantiate(inits, proj_init_fn)
        self.rec_init_fn = instantiate(inits, rec_init_fn)
        self.bias_init_fn = instantiate(inits, bias_init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape[:-1] + [self.dim]
        print self.out_shape

    def init(self):
        self.w = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.u = self.rec_init_fn((self.dim, self.dim))
        self.b = self.bias_init_fn((self.dim))
        # self.h0 = shared0s((1, self.dim))
        # self.params = [self.w, self.u, self.b, self.h0]
        self.params = [self.w, self.u, self.b]

    def step(self, x_t, h_tm1):
        h_t = self.activation(x_t + T.dot(h_tm1, self.u))
        return h_t

    def op(self, state):
        X = self.l_in.op(state=state)
        x = T.dot(X, self.w) + self.b
        out, _ = theano.scan(self.step,
            sequences=[x],
            # outputs_info=[repeat(self.h0, x.shape[1], axis=0)],
            outputs_info=[T.zeros((x.shape[1], self.dim), dtype=theano.config.floatX)],
        )
        return out

    def update(self, cost):
        return self.update_fn(self.params, cost)

class GRU(object):

    def __init__(self, dim=256, activation='tanh', gate_activation='SteeperSigmoid', proj_init_fn='orthogonal',
                 rec_init_fn='orthogonal', bias_init_fn='constant', update_fn='nag'):
        self.dim = dim
        self.proj_init_fn = instantiate(inits, proj_init_fn)
        self.rec_init_fn = instantiate(inits, rec_init_fn)
        self.bias_init_fn = instantiate(inits, bias_init_fn)
        self.update_fn = instantiate(updates, update_fn)
        self.activation = instantiate(activations, activation)
        self.gate_activation = instantiate(activations, gate_activation)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape[:-1] + [self.dim]
        print self.out_shape

    def init(self):
        self.w_z = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.w_r = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.w_h = self.proj_init_fn((self.in_shape[-1], self.dim))

        self.u_z = self.rec_init_fn((self.dim, self.dim))
        self.u_r = self.rec_init_fn((self.dim, self.dim))
        self.u_h = self.rec_init_fn((self.dim, self.dim))

        self.b_z = self.bias_init_fn((self.dim))
        self.b_r = self.bias_init_fn((self.dim))
        self.b_h = self.bias_init_fn((self.dim))

        self.params = [self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]

    def step(self, xz_t, xr_t, xh_t, h_tm1):
        z = self.gate_activation(xz_t + T.dot(h_tm1, self.u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, self.u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, self.u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t

    def op(self, state):
        X = self.l_in.op(state=state)
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
            outputs_info=[T.zeros((x_h.shape[1], self.dim), dtype=theano.config.floatX)], 
        )
        return out

    def update(self, cost):
        return self.update_fn(self.params, cost)