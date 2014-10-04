from foxhound.linear_model import RidgeRegression
import foxhound.linear_model.tests.utils as utils

def test_train_ridge_regression():
	utils.test_train_model(RidgeRegression)

def test_repeatable_ridge_regression():
	utils.test_repeatable_model(RidgeRegression)
