import numpy as np
from numpy.random import RandomState


def generate_dataset():
	X = np.random.random((10, 10))
	y = X.sum(axis=1)
	return X, y

def generate_datapoint():
	X = np.random.random((1, 10))
	return X

def test_train_model(model, *args, **kwargs):
	model = model(*args, **kwargs)
	X, y = generate_dataset()

	model.fit(X, y)

	X = generate_datapoint()
	preds = model.predict(X)
	assert not np.any(np.isnan(preds))

def test_repeatable_model(model, train=None, test=None, *args, **kwargs):
	models = [model(rng=RandomState(0), *args, **kwargs) for i in range(2)]
	X, y = train or generate_dataset()

	for model in models:
		model.fit(X, y)

	X = test or generate_datapoint()
	results = [model.predict(X) for model in models]
	print results
	assert np.allclose(results[0], results[1])
