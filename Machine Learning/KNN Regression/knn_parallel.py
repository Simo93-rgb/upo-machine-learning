from typing import Optional, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class KNN_Parallel:
    def __init__(self, k: int = 3, n_jobs: int = -1) -> None:
        """
        Inizializza il modello KNN.

        Parameters:
        - n_neighbors(int): Numero di vicini da considerare.
        - n_jobs (int): Numero di lavori paralleli da eseguire. -1 per usare tutti i core.
        """
        self.n_neighbors = k
        self.n_jobs = n_jobs
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

    def _predict_single(self, x: np.ndarray) -> float:
        """
        Predice il valore per un singolo punto dati.

        Parameters:
        - x (np.ndarray): Un singolo punto dati.

        Returns:
        - float: La predizione per il punto dati.
        """
        X_train_array = np.array(self.X_train)
        y_train_array = np.array(self.y_train)

        # Calcolo delle distanze dal punto x a tutti i punti di training
        distances = self._euclidean_distance(x, X_train_array)

        # Identifica gli indici dei n_neighborsvicini più vicini
        k_indices = np.argsort(distances)[:self.n_neighbors]

        # Raccoglie i target dei n_neighborsvicini (da y_train_array, che è ora un array numpy)
        k_nearest_targets = y_train_array[k_indices]

        # Calcolo dei pesi usando l'inverso del quadrato della distanza (aggiungo epsilon per evitare divisioni per zero)
        epsilon = 1e-10
        k_nearest_distances = distances[k_indices]
        weights = np.where(k_nearest_distances < 1, 1 / (k_nearest_distances ** 2 + epsilon),
                           1 - (k_nearest_distances ** 2 + epsilon))

        # Normalizzazione dei pesi
        weights_normalized = weights / np.sum(weights)

        # Calcolo della predizione come media pesata
        return np.sum(k_nearest_targets * weights_normalized)

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predice i valori per i nuovi dati basandosi sul KNN con pesatura delle distanze.

        Parameters:
        - X_test (np.ndarray | pd.DataFrame): I dati di test.

        Returns:
        - np.ndarray: Le predizioni.
        """
        # Se X_test è un DataFrame, convertilo in un array numpy
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # Utilizza joblib per parallelizzare la predizione
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_single)(x) for x in X_test
        )

        return np.array(predictions)

    def set_params(self, n_neighbors: Optional[int] = None, n_jobs: Optional[int] = None) -> None:
        """
        Imposta i parametri del modello KNN.

        Parameters:
        - n_neighbors(Optional[int]): Il nuovo valore di n_neighbors(numero di vicini). Se None, mantiene il valore attuale.
        - n_jobs (Optional[int]): Il nuovo valore di n_jobs (numero di lavori paralleli). Se None, mantiene il valore attuale.
        """
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        if n_jobs is not None:
            self.n_jobs = n_jobs

