from foxhound.linear_model import LinearRegression
from foxhound.utils.costs import mse, mae

class LassoRegression(LinearRegression):

    def __init__(self, l1=1.0, *args, **kwargs):
        LinearRegression.__init__(self, *args, **kwargs)
        self.l1 = l1

    def cost(self):
        return mse(self.Y, self.pred)

    def regularization(self):
        return self.l1 * mae(self.W, 0)
