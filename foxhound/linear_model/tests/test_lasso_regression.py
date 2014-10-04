from foxhound.linear_model import LassoRegression
import foxhound.linear_model.tests.utils as utils

def test_train_lasso_regression():
	utils.test_train_model(LassoRegression())
