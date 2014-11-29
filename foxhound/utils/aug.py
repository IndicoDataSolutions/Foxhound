import numpy as np
import theano
import random

# from cv2 import imread as cv2_imread
# from cv2 import resize as cv2_resize
# from cv2 import INTER_AREA, INTER_LINEAR, INTER_NEAREST
# from cv2 import cvtColor
# from cv2 import COLOR_BGR2RGB

from skimage.transform import SimilarityTransform, warp, rotate

from matplotlib import pyplot as plt
from scipy.misc import imread, imsave

from foxhound.utils import floatX
from foxhound.utils.vis import color_grid_vis

def gprob_rounding(n):
    if n.is_integer():
        return n
    else:
        fs = [np.floor, np.ceil]
        return fs[random.randint(0,1)](n)

def center_crop_shape(img, nw, nh):
    iw,ih = img.shape[:2]
    ow = (iw - nw) / 2.
    oh = (ih - nh) / 2.
    ow = prob_rounding(ow)
    oh = prob_rounding(oh)
    return img[ow:ow+nw, oh:oh+nh]

# def cv2_load(path):
#     img = cv2_imread(path)
#     if len(img.shape) == 2:
#         img = np.dstack((img, img, img))
#     img = cvtColor(img, COLOR_BGR2RGB)
#     return img

def flip(img, lr=True, ud=False):
    """
    Randomly flips an image (horizontally, vertically, or both) with probability 0.5.
    """
    choices = []
    if lr:
        choices.append(np.fliplr)
    if ud:
        choices.append(np.flipud)
    if lr and ud:
        choices.append(lambda x:np.flipud(np.fliplr(x)))
    if random.uniform(0., 1.) > 0.5:
        img = choices[random.randint(0, len(choices)-1)](img)
    return img

# def min_resize(img, size, interpolation=INTER_LINEAR):
#     """
#     Resize an image so that it is size along the minimum spatial dimension.
#     """
#     w, h = map(float, img.shape[:2])
#     if w <= h:
#         img = cv2_resize(img, (int(round((h/w)*size)), int(size)), interpolation=interpolation)
#     else:
#         img = cv2_resize(img, (int(size), int(round((w/h)*size))), interpolation=interpolation)
#     return img

def patch(img, size):
    """
    Crop out a size X size subpatch from an image.
    """
    w,h = img.shape[:2]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return img[x:x+size, y:y+size]

# def load_path(path, img_size=73, patch_size=64):
#     img = cv2_load(path)
#     img = min_resize(img, img_size)
#     return img

# def load_path_aug(path, img_size=73, patch_size=64):
#     img = cv2_load(path)
#     img = min_resize(img, img_size)
#     img = patch(img, patch_size)
#     img = flip(img)
#     return img

# def load_path_center_cropped(path, img_size=73, patch_size=64):
#     img = cv2_load(path)
#     img = min_resize(img,img_size)
#     img = center_crop_shape(img,patch_size,patch_size)
#     img = img.transpose(2, 0, 1) / 127.5 - 1.
#     return img


def load_paths(paths):
    return pool.map(load_path, paths)

def load_paths_aug(paths):
    return floatX(pool.map(load_path_aug, paths)).transpose(0, 3, 1, 2) / 127.5 - 1.

def load_paths_center_cropped(paths):
    return floatX(pool.map(load_path_center_cropped, paths))

def image_aug(image, translate=0.125, rotate_degrees=0., flip_lr=True, flip_ud=False):
    image = flip(image, lr=flip_lr, ud=flip_ud)
    h, w = image.shape[:2]
    if translate != 0:
        h_t_range = prob_rounding(h * translate)
        v_t_range = prob_rounding(w * translate)
        h_t = random.randint(-h_t_range, h_t_range+1)
        v_t = random.randint(-v_t_range, v_t_range+1)
        t_form = SimilarityTransform(translation=(h_t, v_t))
        image = warp(image, t_form, mode='nearest')
    if rotate !=0:
        rotate_degrees = random.randint(-rotate_degrees, rotate_degrees)
        image = rotate(image, angle=rotate_degrees, mode='nearest')
    return image

if __name__ == '__main__':
    
    img = imread('/home/alec/Pictures/dog.jpg')/255.
    img = center_crop_shape(img, nw=256, nh=256)
    imgs = [image_aug(img, flip_ud=True, rotate_degrees=15) for _ in range(25)]
    print imgs[0].min(), imgs[0].max()
    img = color_grid_vis(imgs)
    imsave('dawgs.png', img)