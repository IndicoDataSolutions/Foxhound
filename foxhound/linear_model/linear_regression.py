from foxhound.linear_model import LinearModel
from foxhound.utils.costs import MSE

class LinearRegression(LinearModel):

    def cost(self):
        return MSE(self.Y, self.pred)
