from foxhound.linear_model import LinearModel
from foxhound.utils.updates import Adadelta

class RidgeRegression(LinearModel):

    def __init__(self, l2=1.0, *args, **kwargs):
    	update = Adadelta(l2=l2)
        LinearModel.__init__(
        	self, cost='mse', update=update, **kwargs
        )
