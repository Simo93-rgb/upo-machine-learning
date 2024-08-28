import numpy as np


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-10):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.bias = None
        self.losses = []
        self.tolerance = tolerance

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        y = np.array(y).ravel()
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.theta) + self.bias
            h = self.sigmoid(linear_model)

            h = np.clip(h, 1e-10, 1 - 1e-10)

            loss = - (1 / n_samples) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
            self.losses.append(loss)
            if i % (self.n_iterations // 10) == 0:
                print(f'Iteration {i}, Loss: {loss}')

            dw = (1 / n_samples) * np.dot(X.T, (h - y))
            db = (1 / n_samples) * np.sum(h - y)

            self.theta -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i > 0 and abs(self.losses[-2] - loss) < self.tolerance:
                print(f"Convergence reached at iteration {i}")
                break

    def predict(self, X):
        linear_model = np.dot(X, self.theta) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
