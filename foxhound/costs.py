import theano
import theano.tensor as T
import numpy as np

def CategoricalCrossEntropy(y_true, y_pred):
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def BinaryCrossEntropy(y_true, y_pred):
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()

def MeanSquaredError(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

def MeanAbsoluteError(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean()

def SquaredHinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def Hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean()

def GaussianMLE(y_true, y_pred):
    mu, log_sigma = y_pred
    return -T.mean(-(0.5 * np.log(2 * np.pi) + log_sigma) - 0.5 * ((y_true - mu) / T.exp(log_sigma))**2)

cce = CCE = CategoricalCrossEntropy
bce = BCE = BinaryCrossEntropy
mse = MSE = MeanSquaredError
mae = MAE = MeanAbsoluteError
gmle = GMLE = GaussianMLE
