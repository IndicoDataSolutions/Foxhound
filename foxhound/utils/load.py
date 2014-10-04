"""
A bunch of loading functions for datasets used internally at indico for
testing and development. At some point we'll clean this up and have standard
interfaces for academic/personal datasets.
"""

import theano
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cross_validation import train_test_split
import os
import cPickle
from time import time,sleep
from sklearn.externals import joblib
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.misc import imread,imsave
from scipy.io import loadmat
from scipy.spatial.distance import cdist

from cv2 import imread as cv2_imread
from cv2 import resize as cv2_resize
from cv2 import INTER_AREA,INTER_LINEAR
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB

from skimage.transform import resize

import random
import re

# change to where datasets are kept for you
datasets_dir = '/media/datasets/'

def sort(l):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key = alphanum_key)

def one_hot(x):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),np.max(x)+1))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def gimg_top_n_for_n_labels(n_imgs=125,n_labels=100,onehot=False,w=64,h=64,flatten=False):
	data_dir = os.path.join(datasets_dir,'gimg/imgs')
	random.seed(42)
	np.random.seed(42)
	fs = ['_'.join(f.split('_')[:-1]) for f in os.listdir(data_dir)]
	labels = list(set(fs))
	print 'n files %.0f n labels %0.f'%(len(fs),len(labels))
	fs = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
	label2fs = dict(zip(labels,[[] for _ in range(len(labels))]))
	for f in fs:
		label2fs['_'.join(f.split('_')[:-1]).split('/')[-1]].append(f)
	labels = random.sample(labels,n_labels)
	label2fs = dict(zip(labels,(sort(label2fs[label])[:n_imgs] for label in labels)))

	X = [[] for _ in range(n_imgs*n_labels)]
	Y = [[] for _ in range(n_imgs*n_labels)]
	n = 0
	paths = []
	for i,label in enumerate(labels):
		for f in label2fs[label]:
			img = load_imagenet_img(f,w=w,h=h,resize=True)/127.5-1.
			if flatten:
				img = img.flatten()
			paths.append(f)
			X[n] = img
			Y[n] = i
			n += 1
	if onehot:
		Y = one_hot(np.array(Y).astype(int))
	else:
		Y = np.asarray(Y,dtype=theano.config.floatX)
	X = np.asarray(X,dtype=theano.config.floatX)
	return X,Y

def center_crop(img,n_pixels):
	img = img[n_pixels:img.shape[0]-n_pixels,n_pixels:img.shape[1]-n_pixels]
	return img

def lfw(n_imgs=1000,flatten=True):
	data_dir = os.path.join(datasets_dir,'lfw/lfw-deepfunneled')
	if n_imgs == 'all':
		n_imgs = 13233
	n = 0
	imgs = []
	Y = []
	n_to_i = {}
	for root, subFolders, files in os.walk(data_dir):
		if subFolders == []:
			if len(files) >= 2:
				for f in files:
					if n < n_imgs:
						if n % 1000 == 0: print n
						path = os.path.join(root,f)
						img = imread(path)/255.
						img = resize(center_crop(img,53),(32,32,3))-0.5
						if flatten:
							img = img.flatten()
						imgs.append(img)
						n += 1
						name = root.split('/')[-1]
						if name not in n_to_i:
							n_to_i[name] = len(n_to_i)
						Y.append(n_to_i[name])
					else:
						break
	imgs = np.asarray(imgs,dtype=theano.config.floatX)
	Y = np.asarray(Y)
	i_to_n = dict(zip(n_to_i.values(),n_to_i.keys()))
	return imgs,Y,n_to_i,i_to_n

def img_load(path,w,h,resize=True):
	img = cv2_imread(path)
	if len(img.shape) == 2:
		img = np.dstack((img,img,img))
	if resize:
		img = cv2_resize(img,(w,h)).astype(np.uint8)
	img = cvtColor(img,COLOR_BGR2RGB)
	return img

def load_imagenet_img(path,w=32,h=32,resize=True):
	img = img_load(path,w,h,resize=resize)
	img = img.transpose(2,0,1)
	return img


def mnist(ntrain=60000,ntest=10000,onehot=False):
	data_dir = os.path.join(datasets_dir,'mnist')
	fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trainX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trainY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	testX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	testY = loaded[8:].reshape((10000))

	trainX = trainX/255.
	testX = testX/255.

	trainX = trainX[:ntrain]
	trainY = trainY[:ntrain]

	testX = testX[:ntest]
	testY = testY[:ntest]

	if onehot:
		trainY = one_hot(trainY)
		testY = one_hot(testY)

	return trainX,testX,trainY,testY