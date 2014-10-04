from foxhound.linear_model import ElasticNet
import foxhound.linear_model.tests.utils as utils

def test_train_elastic_net():
	utils.test_train_model(ElasticNet)

def test_repeatable_elastic_net():
	utils.test_repeatable_model(ElasticNet)
