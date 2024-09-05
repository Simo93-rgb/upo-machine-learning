import numpy as np


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.1, n_iterations=1000, tolerance=1e-10, regularization='ridge', lambda_=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_ = lambda_
        self.theta = None
        self.bias = None
        self.losses = []

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

            # Funzione di costo con regolarizzazione
            if self.regularization == 'ridge':
                regularization_term = (self.lambda_ / (2 * n_samples)) * np.sum(np.square(self.theta))
            elif self.regularization == 'lasso':
                regularization_term = (self.lambda_ / n_samples) * np.sum(np.abs(self.theta))
            else:
                regularization_term = 0

            loss = - (1 / n_samples) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + regularization_term
            self.losses.append(loss)
            if i % (self.n_iterations // 10) == 0:
                print(f'Iteration {i}, Loss: {loss}')

            dw = (1 / n_samples) * np.dot(X.T, (h - y))
            db = (1 / n_samples) * np.sum(h - y)

            # Aggiornamento con regolarizzazione
            if self.regularization == 'ridge':
                dw += (self.lambda_ / n_samples) * self.theta
            elif self.regularization == 'lasso':
                dw += (self.lambda_ / n_samples) * np.sign(self.theta)

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

    # Metodo per ottenere i parametri (necessario per scikit-learn)
    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'tolerance': self.tolerance,
            'regularization': self.regularization,
            'lambda_': self.lambda_
        }

    # Metodo per impostare i parametri (necessario per scikit-learn)
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
