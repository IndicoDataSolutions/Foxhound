from foxhound.linear_model import LinearModel
from foxhound.utils.updates import Adadelta, Regularizer

class LassoRegression(LinearModel):

    def __init__(self, l1=1.0, *args, **kwargs):
    	regularizer = Regularizer(l1=l1)
    	update = Adadelta()
        LinearModel.__init__(
        	self, update=update, regularizer=regularizer, **kwargs
        )
