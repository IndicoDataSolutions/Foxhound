from foxhound.linear_model import LinearModel
from foxhound.utils.updates import Adadelta
from foxhound.neural_network.layers import Dense

class LogisticRegression(LinearModel):

    def __init__(self, l1=0.0, l2=1.0, *args, **kwargs):
        update = Adadelta(l1=l1, l2=l2)
        layers = [
            Dense(1, activation='softmax')
        ]
        LinearModel.__init__(
            self, layers=layers, cost='bce', update=update, **kwargs
        )
