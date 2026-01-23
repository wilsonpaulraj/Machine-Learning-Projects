import numpy as np

class KNearestNeighbors:
    def __init__(self, k = 3):
        self.k = k
    
    # calculate the distance between the points - using hammilton distance
    def calculate_distance(self, x1, x2):
        return np.sqrt(np.sum((x2-x1)**2))
    
    def top_k(self, x):
        distances = []

        X_train = self.X_train
        y_train = self.y_train

        for i in range(len(X_train)):
            distance = self.calculate_distance(x, X_train[i])
            y = y_train[i]
            distances.append([distance, y])
        
        distances.sort()
        # print(distances)

        freq = {}
        for i in range(self.k):
            dist, y = distances[i]
            freq[y] = freq.get(y, 0) + 1
        
        most_frequent = None

        # print(freq)

        for i in freq.keys():
            if (most_frequent == None):
                most_frequent = i
                continue

            if freq.get(i) > freq.get(most_frequent):
                most_frequent = i
        
        return most_frequent

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        predictions =  [self.top_k(x) for x in X]
        return predictions






