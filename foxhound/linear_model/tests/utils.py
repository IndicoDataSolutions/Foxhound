import numpy as np

def test_train_model(model):
	X = np.random.random((100, 100))
	y = X.sum(axis=1)

	model.fit(X, y)

	X = np.linspace(0, 1, 100)
	model.predict(X)
