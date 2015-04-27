import numpy as np
import pandas as pd
import random
from collections import Counter

from utils import numpy_array       

def FlatToImg(X, w, h, c):
	if not numpy_array(X):
		X = np.asarray(X)	
	return X.reshape(-1, w, h, c)	

def ImgToConv(X):
    if not numpy_array(X):
        X = np.asarray(X)
    return X.transpose(0, 3, 1, 2)

def Standardize(X):
    if not numpy_array(X):
        X = np.asarray(X)
    return X / 127.5 - 1.

def ZeroOneScale(X):
    if not numpy_array(X):
        X = np.asarray(X)
    return X / 255.

def Fliplr(X):
    Xt = []
    for x in X:
        if random.random() > 0.5:
            x = np.fliplr(x)
        Xt.append(x)
    return Xt

def Reflect(X):
	Xt = []
	for x in X:
		if random.random() > 0.5:
			x = np.flipud(x)
		if random.random() > 0.5:
			x = np.fliplr(x)
		Xt.append(x)
	return Xt

def FlipVertical(X):
	Xt = []
	for x in X:
		if random.random() > 0.5:
			x = np.flipud(x)
		Xt.append(x)
	return Xt

def FlipHorizontal(X):
	Xt = []
	for x in X:
		if random.random() > 0.5:
			x = np.fliplr(x)
		Xt.append(x)
	return Xt

def Rot90(X):
    Xt = []
    for x in X:
        x = np.rot90(x, random.randint(0, 3))
        Xt.append(x)
    return Xt

def ColorShift(X, p=1/3., scale=20):
    Xt = []
    for x in X:
        x = x.astype(np.int16)
        x[:, :, 0] += (random.random() < p)*random.randint(-scale, scale)
        x[:, :, 1] += (random.random() < p)*random.randint(-scale, scale)
        x[:, :, 2] += (random.random() < p)*random.randint(-scale, scale)
        x = np.clip(x, 0, 255).astype(np.uint8)
        Xt.append(x)
    return Xt

def Patch(X, pw, ph):
    Xt = []
    for x in X: 
        w, h = x.shape[:2]
        i = random.randint(0, w-pw)
        j = random.randint(0, h-ph)
        Xt.append(x[i:i+pw, j:j+pw])
    return Xt

def CenterCrop(X, pw, ph):
    Xt = []
    for x in X: 
        w, h = x.shape[:2]
        i = int(round((w-pw)/2.))
        j = int(round((h-ph)/2.))
        Xt.append(x[i:i+pw, j:j+pw])
    return Xt
