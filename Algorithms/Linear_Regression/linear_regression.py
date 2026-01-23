import numpy as np


class LinearRegression:
    def __init__(self, lr = 0.01, n_iterations = 100000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
        	y_pred  = np.dot(X, self.weights) + self.bias
        	dw = -2 * 1/n_samples * np.dot(X.T, y - y_pred)
        	db = -2 * 1/n_samples * np.sum(y - y_pred)

        	self.weights = self.weights - self.lr * dw
        	self.bias = self.bias - self.lr * db

    def predict(self, x):
    	return np.dot(x, self.weights) + self.bias






            