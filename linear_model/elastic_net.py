import numpy as np

from linear_model import LinearRegression
from utils.costs import MSE

class ElasticNet(LinearRegression):

    def __init__(self, alpha=1.0, l1_ratio=0.5, *args, **kwargs):
        LinearRegression.__init__(self, *args, **kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.l2_ratio = 1 - l1_ratio

    def cost(self):
        return (MSE(self.Y, self.pred) + 
                self.alpha * self.l1_ratio * MAE(self.W, 0) + 
                self.alpha * self.l2_ratio * MSE(self.W, 0))
