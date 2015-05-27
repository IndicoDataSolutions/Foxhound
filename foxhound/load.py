import os
import cPickle
import numpy as np
from foxhound.preprocessing import one_hot

datasets_dir = '/home/alec/datasets/'

def mnist(ntrain=60000, ntest=10000, onehot=False):
	data_dir = os.path.join(datasets_dir,'mnist')
	fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000, -1)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000, 1))

	fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000, -1)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000, 1))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY)
		teY = one_hot(teY)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX, teX, trY, teY

def unpickle(file):
	fo = open(file, 'rb')
	d = cPickle.load(fo)
	fo.close()
	return d

def cifar10(onehot=False, ntrain=50000, ntest=10000):
	data_dir = '/home/alec/datasets/cifar10/'
	fs = os.listdir(data_dir)

	b_num = 0
	trX = np.zeros((50000, 32*32*3))
	trY = []
	for f in fs:
		if not 'readme' in f:
			data = unpickle(os.path.join(data_dir,f))
		if 'test' in f:
			teX = data['data']
			teY = np.array(data['labels'])
		elif '_batch_' in f:
			trX[b_num*10000:(b_num+1)*10000] = data['data']
			trY.extend(data['labels'])
			b_num += 1
		if 'meta' in f:
						label_names = data['label_names']
	trX = trX
	teX = teX
	trY = np.array(trY).astype(int)
	teY = np.array(teY).astype(int)

	if onehot:
		trY = one_hot(trY)
		teY = one_hot(teY)

	trX = trX[:ntrain]
	teX = teX[:ntest]
	trY = trY[:ntrain]
	teY = teY[:ntest]
	return trX, teX, trY, teY
