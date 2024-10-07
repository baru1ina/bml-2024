import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))



class logistic_regression:
    def __init__(self, input_size):
        super().__init__()

        self.W = np.random.randn(input_size + 1, )

    def get_weights(self):
        return self.W

    def fit(self, X, y, alpha=0.01, epochs=30):
        m = np.shape(X)[0]  # total number of samples
        n = np.shape(X)[1]  # total number of features
        X = np.concatenate((np.ones((m, 1)), X), axis=1)

        cost_history_list = []

        for current_iteration in range(epochs):
            y_estimated = sigmoid(X.dot(self.W))

            error = y_estimated - y

            gradient = (1 / m) * X.T.dot(error)

            self.W = self.W - alpha * gradient

    def predict(self, X):
        y = sigmoid(self.W[0] + X.dot(self.W[1:]))
        # print(self.W)
        y_pred = [1 if i > 0.5 else 0 for i in y]
        return y_pred
