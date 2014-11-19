from foxhound.linear_model import SoftmaxRegression
import foxhound.utils.testing as testing

def test_train_softmax_regression():
	testing.test_train_model(SoftmaxRegression)

def test_repeatable_softmax_regression():
	testing.test_repeatable_model(SoftmaxRegression)
