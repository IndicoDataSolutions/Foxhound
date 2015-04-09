import inspect
import numpy as np

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data]) 

def iter_indices(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        yield b

def shuffle(*data, **kwargs):
    np_rng = kwargs.get('np_rng', np.random.RandomState(42))
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def case_insensitive_import(module, name):
    mapping = dict((k.lower(), k) for k in dir(module))
    return getattr(module, mapping[name.lower()])

def classes_of(module):
	return tuple(x[1] for x in inspect.getmembers(module, inspect.isclass))

def instantiate(module, obj):
    if isinstance(obj, basestring):
        return case_insensitive_import(module, obj)()
    elif isinstance(obj, classes_of(module)):
    	return obj
    else:
    	raise TypeError