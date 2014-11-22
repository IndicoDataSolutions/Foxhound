from foxhound.neural_network import Net
from foxhound.neural_network.layers import Dense

class LinearModel(Net):

    def __init__(self, *args, **kwargs):

        layers = kwargs.pop('layers', None) or [Dense(1, activation='linear')]
        Net.__init__(self, layers=layers, *args, **kwargs)
