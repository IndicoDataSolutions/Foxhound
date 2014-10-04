from foxhound.linear_model import LinearRegression
from foxhound.utils.costs import MSE, MAE

class ElasticNet(LinearRegression):

    def __init__(self, l1=0.5, l2=0.25, *args, **kwargs):
        LinearRegression.__init__(self, *args, **kwargs)
        self.l1 = l1
        self.l2 = l2

    def regularization(self):
        return self.l1 * MAE(self.W, 0) + self.l2 * MSE(self.W, 0)
