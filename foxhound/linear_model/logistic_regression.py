import theano.tensor as T

from foxhound.linear_model import LinearModel
from foxhound.utils.costs import bce, mae, mse

class LogisticRegression(LinearModel):

    def __init__(self, l1=0.0, l2=1.0, *args, **kwargs):
        LinearModel.__init__(self, *args, **kwargs)
        self.l1 = l1
        self.l2 = l2

    def activation(self, preactivation):
    	return T.nnet.softmax(preactivation)

    def cost(self):
        return bce(self.Y, self.pred)

    def regularization(self):
		return self.l1 * mae(self.W, 0) + self.l2 * mse(self.W, 0)
