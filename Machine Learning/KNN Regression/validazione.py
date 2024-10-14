from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, r2_score, mean_absolute_error, root_mean_squared_error
import numpy as np
import pandas as pd
from typing import Dict, Union

class KFoldValidation:
    def __init__(self, model, k_folds: int = 5) -> None:
        """
        Inizializza la classe di validazione incrociata K-Fold.

        Parameters:
        - model: Il modello da validare.
        - k_folds (int): Il numero di fold per la validazione incrociata.
        """
        self.model = model
        self.k_folds = k_folds

    def validate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], y_scaler=None) -> Dict[
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

            # Ripristina alla scala originale se lo scaler è presente
            if y_scaler is not None:
                y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

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
