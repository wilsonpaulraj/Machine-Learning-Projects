import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from linear_regression import LinearRegression


X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

reg = LinearRegression(lr = 0.01, n_iterations = 1000)
reg.fit(X, y)



X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Predict y for those x values
y_line = reg.predict(X_line)

plt.scatter(X[:, 0], y, label="Training data")

# Regression line
plt.plot(X_line[:, 0], y_line, color="red", label="Regression line")

X_new = np.random.uniform(X.min(), X.max(), size=(10, 1))
y_pred_new = reg.predict(X_new)



plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

