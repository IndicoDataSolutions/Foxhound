from foxhound.linear_model import LinearModel
from foxhound.utils.costs import mse

class LinearRegression(LinearModel):

    def cost(self):
        return mse(self.Y, self.pred)
