from foxhound.linear_model import LinearModel
from foxhound.utils.updates import Adadelta

class LassoRegression(LinearModel):

    def __init__(self, l1=1.0, *args, **kwargs):
    	update = Adadelta(l1=l1)
        LinearModel.__init__(
        	self, *args, cost='mse', update=update, **kwargs
        )
