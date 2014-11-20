from foxhound.linear_model import LinearModel
from foxhound.utils.costs import mse

class LinearRegression(LinearModel):

    def __init__(self, *args, **kwargs):
    	LinearModel.__init__(self, cost='mse', **kwargs)
