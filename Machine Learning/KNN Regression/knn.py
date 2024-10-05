from typing import Optional, Union
import numpy as np
import pandas as pd


class KNN:
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Addestra il modello KNN.

        Parameters:
        - X_train (np.ndarray): Il set di dati di training.
        - y_train (np.ndarray): I target di training.
        """
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Calcola la distanza euclidea tra x1 e tutte le righe di X_train.

        Parameters:
        - x1 (np.ndarray): Un singolo punto dati.
        - X_train (np.ndarray): Il set di dati di training.

        Returns:
        - np.ndarray: Un array di distanze.
        """
        return np.sqrt(np.sum((X_train - x1) ** 2, axis=1))


    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predice i valori per i nuovi dati basandosi sul KNN.

        Parameters:
        - X_test (np.ndarray | pd.DataFrame): I dati di test.

        Returns:
        - np.ndarray: Le predizioni.
        """
        # Se X_test è un DataFrame, convertilo in un array numpy
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        predictions = np.zeros(X_test.shape[0])

        # Converte self.y_train in un array numpy per accedere correttamente agli indici posizionali
        y_train_array = np.array(self.y_train)

        for i, x in enumerate(X_test):
            # Assicurati che ogni riga di X_test sia numerica
            if isinstance(x, np.ndarray) and x.dtype.kind in {'U', 'S'}:
                x = x.astype(float)

            # Calcolo delle distanze dal punto x a tutti i punti di training
            distances = self._euclidean_distance(x, self.X_train)

            # Identifica gli indici dei k vicini più vicini
            k_indices = np.argsort(distances)[:self.k]

            # Raccoglie i target dei k vicini (da y_train_array, che è ora un array numpy)
            k_nearest_targets = y_train_array[k_indices]

            # Predizione: la media dei target per la regressione
            predictions[i] = np.mean(k_nearest_targets)

        return predictions
