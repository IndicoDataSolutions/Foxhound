import numpy as np

from linear_model import LinearRegression
from utils.costs import MSE

class LassoRegression(LinearRegression):

    def __init__(self, alpha=1.0, *args, **kwargs):
        LinearRegression.__init__(self, *args, **kwargs)
        self.alpha = alpha

    def cost(self):
        return MSE(self.Y, self.pred) + self.alpha * MAE(self.W, 0)
