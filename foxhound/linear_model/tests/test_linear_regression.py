from foxhound.linear_model import LinearRegression
import foxhound.linear_model.tests.utils as utils

def test_train_linear_regression():
	utils.test_train_model(LinearRegression())
