import theano
import theano.tensor as T

class Cost(object):

	def __init__(self, target=None):
		if target is None:
			self.target = T.fmatrix()
		else:
			self.target = target

	def get_cost(self, pred):
		return NotImplementedError


class CategoricalCrossEntropy(Cost):

	def get_cost(self, pred):
		if self.target.type is T.ivector:
			return -T.mean(T.log(pred)[T.arange(self.target.shape[0]), self.target])
		else:
			return T.nnet.categorical_crossentropy(pred, self.target).mean()


class BinaryCrossEntropy(Cost):

	def get_cost(self, pred):
		return T.nnet.binary_crossentropy(pred, self.target).mean()
		

class MeanSquaredError(Cost):

	def get_cost(self, pred):
		return T.sqr(pred - self.target).mean()


class MeanAbsoluteError(Cost):

	def get_cost(self, pred):
		return T.abs_(pred - self.target).mean()


class Hinge(Cost):

	def get_cost(self, pred):
		return T.maximum(1. - self.target * pred, 0.).mean()


class SquaredHinge(Cost):

	def get_cost(self, pred):
		return T.sqr(T.maximum(1. - self.target * pred, 0.)).mean()

# aliasing
cce = CCE = CategoricalCrossEntropy
bce = BCE = BinaryCrossEntropy
mse = MSE = MeanSquaredError
mae = MAE = MeanAbsoluteError
