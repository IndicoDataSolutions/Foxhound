from foxhound.linear_model import LinearRegression
from foxhound.utils.costs import mse

class RidgeRegression(LinearRegression):

    def __init__(self, alpha=1.0, *args, **kwargs):
        LinearRegression.__init__(self, *args, **kwargs)
        self.alpha = alpha

    def cost(self):
        return mse(self.Y, self.pred)

    def regularization(self):
        return self.alpha * mse(self.W, 0)
