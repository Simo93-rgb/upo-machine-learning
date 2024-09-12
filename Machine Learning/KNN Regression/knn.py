import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        """Salva i dati di addestramento."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, X_train):
        """Calcola la distanza euclidea tra un campione x1 e tutti i campioni di X_train."""
        return np.sqrt(np.sum((X_train - x1) ** 2, axis=1))

    def predict(self, X_test):
        """Predice i valori per i nuovi dati."""
        predictions = np.zeros(X_test.shape[0])

        for i, x in enumerate(X_test):
            distances = self._euclidean_distance(x, self.X_train)

            # Calcolo dei pesi come l'inverso del quadrato della distanza (aggiungo un epsilon per evitare divisioni per zero)
            epsilon = 1e-6
            weights = 1 / (distances ** 2 + epsilon)

            # Prendo i primi k vicini
            neighbor_indices = np.argsort(distances)[:self.k]

            # Prendo i valori corrispondenti a questi vicini
            neighbor_values = self.y_train[neighbor_indices]

            # Normalizzazione dei pesi
            neighbor_weights = weights[neighbor_indices]
            neighbor_weights_normalized = neighbor_weights / np.sum(neighbor_weights)

            # Calcolo della media pesata
            predictions[i] = np.sum(neighbor_values * neighbor_weights_normalized)

        return predictions