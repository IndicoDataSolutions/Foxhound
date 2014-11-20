from foxhound.linear_model import LinearModel
from foxhound.utils.updates import Adadelta

class ElasticNet(LinearModel):

    def __init__(self, l1=0.5, l2=0.25, *args, **kwargs):
    	update = Adadelta(l1=l1, l2=l2)
        LinearModel.__init__(
        	self, *args, cost='mse', update=update, **kwargs
        )
