import numpy as np
from numpy.random import RandomState


def generate_dataset():
	X = np.random.random((10, 10)).astype(np.float32)
	Y = 2*X + 1
	return X, Y

def generate_datapoint():
	X = np.random.random((1, 10)).astype(np.float32)
	return X

def test_train_model(model, *args, **kwargs):
	model = model(*args, **kwargs)
	X, Y = generate_dataset()

	model.fit(X, Y)

	X = generate_datapoint()
	preds = model.predict(X)
	assert not np.any(np.isnan(preds))
