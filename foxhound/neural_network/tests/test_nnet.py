import foxhound.neural_network.tests.utils as utils
from foxhound.neural_network import Net
from foxhound.neural_network.layers import Dense

def test_train_nnet():
	layers = [Dense(10)]
	utils.test_train_model(Net, layers)
