from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, r2_score, mean_absolute_error, root_mean_squared_error
import numpy as np
import pandas as pd
from typing import Dict, Union
from knn_parallel import KNN_Parallel

class KFoldValidation:
    def __init__(self, model, k_folds: int = 5) -> None:
        """
        Inizializza la classe di validazione incrociata K-Fold.

        Parameters:
        - model: Il modello da validare.
        - k_folds (int): Il numero di fold per la validazione incrociata.
        """
        self.model: KNN_Parallel = model
        self.k_folds = k_folds
        

    def validate_and_find_n_neighbors(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], 
                                    ) -> Dict[str, float]:
        """
        Esegue la k-fold cross-validation, trova il miglior valore di n_neighbors (k) in base alla RMSE,
        e restituisce le metriche per il k ottimale.

        Parameters:
        - X (np.ndarray | pd.DataFrame): Le features.
        - y (np.ndarray | pd.Series): I target.

        Returns:
        - Dict[str, float]: Un dizionario contenente le metriche medie calcolate per il k ottimale.
        """
        # Converti y in un array numpy se è una Serie Pandas
        y = np.array(y)

        # Inizializzazione dei risultati
        best_k = None
        best_rmse = float('inf')
        best_metrics = {}

        # Utilizzo di KFold di scikit-learn
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(X):
            # Selezione dei dati di training e test
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Itera sui valori di k (n_neighbors) con un range fisso (es. da 5 a 200 con step di 15)
            for k in range(5, 200, 15):

                # Aggiorna il valore di k nel modello
                self.model.set_params(n_neighbors=k)

                # Addestra il modello
                self.model.fit(X_train, y_train)

                # Predice sui dati di test
                y_pred = self.model.predict(X_test)

                # Calcola la RMSE per questo k
                rmse = root_mean_squared_error(y_test, y_pred)

                # Aggiorna il miglior k se la RMSE corrente è migliore (minima)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_k = k
            print(f'Fine giro k-fold')

        # Dopo aver trovato il miglior k, ricalcola le metriche per quel valore di k
        self.model.set_params(n_neighbors=best_k)

        # Inizializza le liste per memorizzare le metriche finali per il miglior k
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        mape_scores = []
        evs_scores = []

        # Secondo ciclo per la cross-validation usando il miglior k trovato
        for train_index, test_index in kf.split(X):
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Addestra il modello con il miglior k
            self.model.fit(X_train, y_train)

            # Predice sui dati di test
            y_pred = self.model.predict(X_test)

            # Calcolo delle metriche
            rmse_scores.append(root_mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
            mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))
            evs_scores.append(explained_variance_score(y_test, y_pred))

        # Restituisce le metriche medie per il miglior k
        best_metrics = {
            'RMSE': np.mean(rmse_scores),
            'MAE': np.mean(mae_scores),
            'R2': np.mean(r2_scores),
            'MAPE': np.mean(mape_scores),
            'EVS': np.mean(evs_scores),
            'Best_n_neighbors': best_k
        }

        return best_metrics

    
    def validate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[
        str, float]:
        """
        Esegue la k-fold cross-validation e restituisce la media delle metriche.

        Parameters:
        - X (np.ndarray | pd.DataFrame): Le features.
        - y (np.ndarray | pd.Series): I target.

        Returns:
        - Dict[str, float]: Un dizionario contenente le metriche medie calcolate.
        """
        # Converti y in un array numpy se è una Serie Pandas
        y = np.array(y)

        # Inizializzazione delle liste per memorizzare le metriche
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        mape_scores = []
        evs_scores = []

        # Utilizzo di KFold di scikit-learn
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            # Utilizzo di .iloc per selezionare le righe con indici posizionali
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            # Addestramento del modello
            self.model.fit(X_train, y_train)

            # Predizione sui dati di test
            y_pred = self.model.predict(X_test)

            # Calcolo delle metriche
            rmse_scores.append(root_mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
            mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))
            evs_scores.append(explained_variance_score(y_test, y_pred))

        # Restituzione delle metriche medie
        metrix = {
            'RMSE': np.mean(rmse_scores),
            'MAE': np.mean(mae_scores),
            'R2': np.mean(r2_scores),
            'MAPE': np.mean(mape_scores),
            'EVS': np.mean(evs_scores)
        }
        return metrix

