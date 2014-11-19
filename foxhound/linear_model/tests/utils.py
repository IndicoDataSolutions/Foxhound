import numpy as np
from numpy.random import RandomState


def generate_dataset():
	X = np.random.random((10, 10))
	Y = X.sum(axis=1)
	return X, Y

def generate_datapoint():
	X = np.random.random((1, 10))
	return X

def test_train_model(model, *args, **kwargs):
	model = model(*args, **kwargs)
	X, Y = generate_dataset()

	model.fit(X, Y)

	X = generate_datapoint()
	preds = model.predict(X)
	assert not np.any(np.isnan(preds))

def test_repeatable_model(model, *args, **kwargs):
	models = [model(rng=RandomState(0), *args, **kwargs) for i in range(2)]
	X, Y = generate_dataset()

	for model in models:
		model.fit(X, Y)

	X = generate_datapoint()
	results = [model.predict(X) for model in models]
	assert np.allclose(results[0], results[1])
