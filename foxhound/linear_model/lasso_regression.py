from foxhound.linear_model import LinearRegression
from foxhound.utils.costs import MSE, MAE

class LassoRegression(LinearRegression):

    def __init__(self, alpha=1.0, *args, **kwargs):
        LinearRegression.__init__(self, *args, **kwargs)
        self.alpha = alpha

    def cost(self):
        return MSE(self.Y, self.pred)

    def regularization(self):
        return self.alpha * MAE(self.W, 0)
