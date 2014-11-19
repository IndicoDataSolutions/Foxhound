from foxhound.linear_model import LinearRegression
from foxhound.utils.costs import mse, mae

class ElasticNet(LinearRegression):

    def __init__(self, l1=0.5, l2=0.25, *args, **kwargs):
        LinearRegression.__init__(self, *args, **kwargs)
        self.l1 = l1
        self.l2 = l2

    def regularization(self):
        return self.l1 * mae(self.W, 0) + self.l2 * mse(self.W, 0)
