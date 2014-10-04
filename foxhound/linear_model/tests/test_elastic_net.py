from foxhound.linear_model import ElasticNet
import foxhound.linear_model.tests.utils as utils

def test_train_elastic_net():
	utils.test_train_model(ElasticNet())
