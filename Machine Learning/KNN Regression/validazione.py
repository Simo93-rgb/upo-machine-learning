from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from typing import Dict, Union
from knn_parallel import KNN_Parallel
import valutazione


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
        self.evaluate = valutazione.evaluate_model
        

    def validate_and_find_best_hyper_params(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
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
        best_minkowski = None
        best_mse = float('inf')
        best_mape = float('inf')

        best_metrics = {}

        # Utilizzo di KFold di scikit-learn
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        
        for minkowski in range(1,3):
            print(f'Sto eseguendo minkowski={minkowski}')
            self.model.set_params(minkowski=minkowski)
            # Itera sui valori di k (n_neighbors) con un range fisso (es. da 5 a 200 con step di 15)
            for k in range(2, 30):
                print(f'Sto eseguendo n_neighbourhood={k}')
                # Aggiorna il valore di k nel modello
                self.model.set_params(n_neighbors=k)
                for train_index, test_index in kf.split(X):
                    # Selezione dei dati di training e test
                    if isinstance(X, pd.DataFrame):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    else:
                        X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # Addestra il modello
                    self.model.fit(X_train, y_train)

                    # Predice sui dati di test
                    y_pred = self.model.predict(X_test)

                    # Calcola la RMSE per questo k
                    mse = valutazione.mean_squared_error(y_test, y_pred)
                    mape = valutazione.mean_absolute_percentage_error(y_test, y_pred)

                    # Aggiorna il miglior k se la RMSE corrente è migliore (minima)
                    # if mse < best_mse:
                    #     best_mse = mse
                    #     best_k = k
                    #     best_minkowski = minkowski
                    if mape < best_mape:
                        best_mape = mape
                        best_k = k
                        best_minkowski = minkowski
            print(f'Fine giro k-fold')

        # Dopo aver trovato il miglior k e miglior p di minkowski, ricalcola le metriche per quel valore di k e p
        print(f'Migliori iper parametri: k={best_k}, minkowski={best_minkowski}')
        self.model.set_params(n_neighbors=best_k, minkowski=best_minkowski)

        return self.validate(X, y)

    
    def validate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
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

        # Inizializzazione delle somme delle metriche
        metric_sums = {}

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

            # Calcolo delle metriche con evaluate
            metrics = self.evaluate(y_test, y_pred, print_metrix=False, savefile=False)

            # Aggiunta cumulativa delle metriche
            for metric_name, metric_value in metrics.items():
                if metric_name not in metric_sums:
                    metric_sums[metric_name] = 0
                metric_sums[metric_name] += metric_value

        # Calcolo delle metriche medie
        metric_averages = {metric_name: metric_value / self.k_folds for metric_name, metric_value in metric_sums.items()}
        
        return metric_averages


