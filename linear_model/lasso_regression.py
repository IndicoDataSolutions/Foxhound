import numpy as np

from linear_model import LinearRegression
from utils.costs import MSE

class LassoRegression(LinearRegression):

    def __init__(self, alpha=1.0, *args, **kwargs):
        LinearRegression.__init__(self, *args, **kwargs)
        self.alpha = alpha

    def cost(self):
        return MSE(self.Y, self.pred) + self.alpha * MAE(self.W, 0)

if __name__ == "__main__":
    X = np.random.random((100, 100))
    y = X.sum(axis=1)

    model = LassoRegression()
    model.fit(X, y)

    X = np.linspace(0, 1, 100)
    model.predict(X)
