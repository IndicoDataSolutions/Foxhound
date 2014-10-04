import numpy as np

from foxhound.linear_model import LinearRegression

def test_linear_regression():
	X = np.random.random((100, 100))
	y = X.sum(axis=1)

	model = LinearRegression()
	model.fit(X, y)

	X = np.linspace(0, 1, 100)
	pred = model.predict(X)
