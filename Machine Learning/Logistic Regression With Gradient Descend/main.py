import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from ucimlrepo import fetch_ucirepo
from logistic_regression_with_gradient_descend import LogisticRegressionGD


def k_fold_cross_validation(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy


def leave_one_out_cross_validation(model, X, y):
    loo = LeaveOneOut()
    accuracies = []

    for train_index, val_index in loo.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        accuracy = accuracy_score(y_val, prediction)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy


if __name__ == "__main__":
    start_time = time.time()
    # iperparametri
    _lambda = 0.1
    learning_rate = 0.05
    n_iterations = 1000
    k = 5  # numero fold per la cross validation k-fold
    tolerance = 1e-6
    regularization = 'ridge'  # ['ridge', 'lasso', 'none']

    # Fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # Data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # Imputazione dei NaN con la media delle colonne
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Normalizzazione delle feature
    X_normalized = (X_imputed - np.mean(X_imputed, axis=0)) / np.std(X_imputed, axis=0)

    # Encoding delle classi
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split del dataset in training e validazione
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_encoded, test_size=0.3, random_state=42)

    # Inizializza e addestra il modello
    model = LogisticRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations, lambda_=_lambda,
                                 regularization=regularization)
    model.fit(X_train, y_train)

    # Prevedi i risultati sul set di validazione
    predictions = model.predict(X_val)

    # Modello con scikit-learn
    sk_model = LogisticRegression(max_iter=100)
    sk_model.fit(X_train, y_train)

    # Prevedi con scikit-learn sul set di validazione
    sk_predictions = sk_model.predict(X_val)

    # Calcolo dell'accuratezza per entrambi i modelli
    accuracy_custom = accuracy_score(y_val, predictions)
    accuracy_sk = accuracy_score(y_val, sk_predictions)

    # Esegui K-Fold Cross-Validation
    k_fold_accuracy = k_fold_cross_validation(
        LogisticRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations, lambda_=_lambda,
                             regularization=regularization), X_normalized,
        y_encoded, k=10)

    # Esegui Leave-One-Out Cross-Validation
    loo_accuracy = leave_one_out_cross_validation(
        LogisticRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations, lambda_=_lambda,
                             regularization=regularization), X_normalized,
        y_encoded)

       # Tempo di esecuzione
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
    # Stampa delle accuratezze
    print(f'Accuracy del modello implementato: {accuracy_custom}')
    print(f'Accuracy del modello Scikit-learn: {accuracy_sk}')
    print(f'K-Fold Cross-Validated Accuracy: {k_fold_accuracy}')
    print(f'Leave-One-Out Cross-Validated Accuracy: {loo_accuracy}')
