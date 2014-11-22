from foxhound.linear_model import LinearModel
from foxhound.utils.updates import Adadelta, Regularizer

class RidgeRegression(LinearModel):

    def __init__(self, l2=1.0, *args, **kwargs):
    	regularizer = Regularizer(l2=l2)
    	update = Adadelta()
        LinearModel.__init__(
        	self, update=update, regularizer=regularizer, **kwargs
        )
