import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Pool

from cv2 import imread as cv2_imread
from cv2 import resize as cv2_resize
from cv2 import INTER_AREA, INTER_LINEAR, INTER_NEAREST
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB, COLOR_RGB2HSV, COLOR_HSV2RGB

from utils import numpy_array    
from rng import np_rng, py_rng   

def standardize(X):
    return np.asarray(X)/127.5 - 1.

def load_path(path):
    img = cv2_imread(path)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    img = cvtColor(img, COLOR_BGR2RGB)
    return img

def center_crop(img, nw, nh):
    w, h = img.shape[:2]
    w = int(round((w - nw) / 2.))
    h = int(round((h - nh) / 2.))
    return img[w:w+nw, h:h+nh]

def flip(img):
    """
    Randomly flip an image horizontally with probability 0.5.
    """
    if py_rng.random() > 0.5:
        img = np.fliplr(img)
    return img

def color_shift(x, p=1/3., scale=20):
    x = x.astype(np.int16)
    x[:, :, 0] += (py_rng.random() < p)*py_rng.randint(-scale, scale)
    x[:, :, 1] += (py_rng.random() < p)*py_rng.randint(-scale, scale)
    x[:, :, 2] += (py_rng.random() < p)*py_rng.randint(-scale, scale)
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def min_resize(img, size, interpolation=INTER_LINEAR):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    w, h = map(float, img.shape[:2])
    if w <= h:
        img = cv2_resize(img, (int(round((h/w)*size)), int(size)), interpolation=interpolation)
    else:
        img = cv2_resize(img, (int(size), int(round((w/h)*size))), interpolation=interpolation)
    return img

def _train_fc_imagenet(x):
    x = load_path(x)
    x = min_resize(x, 64)
    x = center_crop(x, 64, 64)
    x = color_shift(x)
    x = flip(x)
    x = x.flatten()
    return x    

def _test_fc_imagenet(x):
    x = load_path(x)
    x = min_resize(x, 64)
    x = center_crop(x, 64, 64)
    x = x.flatten()
    return x    

pool = Pool(8)

def train_fc_imagenet(X):
    return standardize(pool.map(_train_fc_imagenet, X))

def test_fc_imagenet(X):
    return standardize(pool.map(_test_fc_imagenet, X))