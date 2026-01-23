from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
from knn import KNearestNeighbors
dataset = datasets.load_iris()
X,y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12308894)

clf = KNearestNeighbors()
clf.train(X_train, y_train)
predictions = clf.predict(X_test)

# Evaluation:
accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)

def plot_dataset():
    plt.figure()
    plt.scatter(X[:, 2], X[:, 3], c=y)
    plt.show()

# plot_dataset()
