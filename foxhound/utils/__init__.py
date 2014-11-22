import numpy as np
import theano
import theano.tensor as T

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = data[0].shape[0] / size + 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data]) 

def iter_indices(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size + 1
    for b in range(batches):
        yield b

def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return data[0][idxs]
    else:
        return [d[idxs] for d in data]

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX):
    return theano.shared(np.asarray(X, dtype=dtype))

def downcast_float(X):
    return np.asarray(X, dtype=np.float32)

def case_insensitive_import(module, name):
    mapping = dict((k.lower(), k) for k in dir(module))
    return getattr(module, mapping[name])
