from foxhound.linear_model import LogisticRegression
import foxhound.linear_model.tests.utils as utils

def test_train_logistic_regression():
	utils.test_train_model(LogisticRegression)

def test_repeatable_logistic_regression():
	utils.test_repeatable_model(LogisticRegression)
