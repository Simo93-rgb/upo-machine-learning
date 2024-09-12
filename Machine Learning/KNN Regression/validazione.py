import numpy as np
from valutazione import *


class KFoldValidation:
    def __init__(self, model, k_folds=5):
        self.model = model
        self.k_folds = k_folds

    def split(self, X):
        """Genera indici per k-fold cross-validation."""
        fold_size = len(X) // self.k_folds
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        folds = []
        for i in range(self.k_folds):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.delete(indices, np.arange(i * fold_size, (i + 1) * fold_size))
            folds.append((train_indices, test_indices))
        return folds

    def validate(self, X, y):
        """Esegue la k-fold cross-validation e restituisce la media delle metriche."""
        y = np.array(y)
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        mape_scores = []
        evs_scores = []

        folds = self.split(X)
        for train_indices, test_indices in folds:
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            # Calcolo delle metriche
            rmse_scores.append(root_mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score_metric(y_test, y_pred))
            mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))
            evs_scores.append(explained_variance(y_test, y_pred))

        # Restituzione delle metriche medie
        metrix = {
            'RMSE': np.mean(rmse_scores),
            'MAE': np.mean(mae_scores),
            'R2': np.mean(r2_scores),
            'MAPE': np.mean(mape_scores),
            'EVS': np.mean(evs_scores)
        }
        return metrix
