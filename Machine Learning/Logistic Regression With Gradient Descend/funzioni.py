import time
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from validazione import k_fold_cross_validation, leave_one_out_cross_validation, stratified_k_fold_cross_validation, \
    validation_test
from logistic_regression_with_gradient_descend import LogisticRegressionGD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from skopt import BayesSearchCV
from plot import plot_class_distribution, plot_corr_matrix, plot_roc_curve, \
    plot_metrics_comparison, plot_sigmoid, plot_confusion_matrix, plot_precision_recall, plot_regularization_effect
import os
import json


def carica_dati():
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    return X, y


def preprocessa_dati(X, y, class_balancer=""):
    # Imputazione dei NaN e normalizzazione
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_normalized = (X_imputed - X_imputed.mean(axis=0)) / X_imputed.std(axis=0)

    # Elimina le feature altamente correlate
    X_normalized, features_eliminate = elimina_feature_correlate(X_normalized)
    print(f"Features eliminate: {features_eliminate}")

    # Encoding delle classi
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).ravel()

    # Plotting delle classi prima del bilanciamento
    plot_class_distribution(y_encoded, file_name='class_distribution_pie_breast_cancer')

    # Gestione del bilanciamento delle classi
    if class_balancer:
        resampler = SMOTE(random_state=42) if class_balancer == "SMOTE" else RandomUnderSampler(random_state=42)
        X_normalized, y_encoded = resampler.fit_resample(X_normalized, y_encoded)
        plot_class_distribution(y_encoded, file_name=f'class_distribution_pie_breast_cancer_{class_balancer}')

    return X_normalized, features_eliminate, y_encoded


def elimina_feature_correlate(X, soglia=0.95):
    # Calcola la matrice di correlazione
    corr_matrix = np.corrcoef(X, rowvar=False)

    plot_corr_matrix(corr_matrix)

    # Trova le feature da eliminare (quelle altamente correlate)
    feature_da_eliminare = set()
    for i in range(len(corr_matrix)):
        for j in range(i):
            if abs(corr_matrix[i, j]) > soglia:
                feature_da_eliminare.add(i)

    # Elimina le feature dal dataset
    X_ridotto = np.delete(X, list(feature_da_eliminare), axis=1)

    return X_ridotto, feature_da_eliminare


def addestra_modelli(X_train, y_train, X_val, best_params, k=5):
    # Estrai i migliori iperparametri trovati con l'ottimizzazione bayesiana
    learning_rate = best_params['learning_rate']
    n_iterations = best_params['n_iterations']
    _lambda = best_params['lambda_']
    regularization = best_params['regularization']

    # Modello Logistic Regression implementato
    model = LogisticRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations, lambda_=_lambda,
                                 regularization=regularization)
    model.fit(X_train, y_train)

    # Predizione sul set di validazione
    # predictions = model.predict(X_val)

    # # Valutazione Cross-Validation
    # k_fold_accuracy = k_fold_cross_validation(model, X_train, y_train, k)
    # stratified_k_fold_accuracy = stratified_k_fold_cross_validation(model, X_train, y_train, n_splits=5)
    #
    # print(f"Accuratezza media con Stratified 5-Fold Cross-Validation: {stratified_k_fold_accuracy}")
    # print(f"Accuratezza con K-Fold {k} Cross-Validation: {k_fold_accuracy}")
    #
    # # Esegui Leave-One-Out Cross-Validation (solo se necessario)
    # loo_accuracy = leave_one_out_cross_validation(model, X_train, y_train)
    # print(f"Accuratezza con Leave-One-Out Cross-Validation: {loo_accuracy}")

    # Modello di scikit-learn Logistic Regression
    sk_model = LogisticRegression(max_iter=100)
    sk_model.fit(X_train, y_train)
    # sk_predictions = sk_model.predict(X_val)

    # return model, predictions, sk_model, sk_predictions
    return model, sk_model


def bayesian_optimization(X_train, y_train, file_path="best_parameters.json"):
    # Definisci lo spazio degli iperparametri da ottimizzare
    param_space = {
        'learning_rate': (1e-4, 1e-1, 'log-uniform'),
        'lambda_': (1e-4, 10e2, 'log-uniform'),
        'n_iterations': (100, 10000),
        'regularization': ['ridge', 'lasso', 'none']
    }

    def model_fit(learning_rate, lambda_, n_iterations, regularization):
        model = LogisticRegressionGD(learning_rate=learning_rate, lambda_=lambda_,
                                     n_iterations=n_iterations, regularization=regularization)
        return model

    bayes_search = BayesSearchCV(
        estimator=LogisticRegressionGD(),
        search_spaces=param_space,
        n_iter=150,
        cv=len(X_train),
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    bayes_search.fit(X_train, y_train)

    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    return best_params, best_score


def bayesian_optimization_wrapper(X_train, y_train, file_path="best_parameters.json"):
    # Controllo se esiste il file con i parametri salvati
    if os.path.exists(file_path):
        print(f"Caricamento dei parametri ottimali da {file_path}...")
        with open(file_path, 'r') as file:
            best_params = json.load(file)
        best_score = best_params.pop("accuracy", None)
    else:
        # Eseguire l'ottimizzazione bayesiana se il file non esiste
        print("Eseguendo l'ottimizzazione bayesiana...")
        best_params, best_score = bayesian_optimization(X_train, y_train)
        save_best_params(best_params, best_score, file_path)

    return best_params, best_score


def save_best_params(best_params, best_score, file_path="best_parameters.json"):
    best_params['accuracy'] = best_score
    with open(file_path, 'w') as file:
        json.dump(best_params, file)
    print(f"Parametri salvati in {file_path}.")