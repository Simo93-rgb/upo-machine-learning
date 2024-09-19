import time
from enum import Enum

import numpy as np
import pandas as pd
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
from plot import *
import os
import json



def carica_dati():
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    return X, y


def preprocessa_dati(X, y, class_balancer="", corr=0.95):
    # Elimina le feature altamente correlate
    X, features_eliminate = elimina_feature_correlate(X, soglia=corr)
    print(f"Features eliminate: {features_eliminate}")

    # Imputazione dei NaN e normalizzazione
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_normalized = (X_imputed - X_imputed.mean(axis=0)) / X_imputed.std(axis=0)

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
    """
    Elimina feature altamente correlate da un array numpy X basandosi sulla soglia fornita.

    Parametri:
    - X: array numpy bidimensionale, dove le colonne sono le feature.
    - soglia: valore soglia per la correlazione (default 0.95).

    Restituisce:
    - X_ridotto: array numpy con le feature eliminate.
    - feature_da_eliminare: set degli indici delle feature eliminate.
    """
    # Calcola la matrice di correlazione
    corr_matrix = np.corrcoef(X, rowvar=False)
    plot_corr_matrix(corr_matrix)
    # Calcola la varianza di ogni feature
    feature_variances = np.var(X, axis=0)

    num_features = corr_matrix.shape[0]
    feature_da_eliminare = set()

    for i in range(num_features):
        if i in feature_da_eliminare:
            continue  # Salta le feature già eliminate
        for j in range(i + 1, num_features):
            if j in feature_da_eliminare:
                continue  # Salta le feature già eliminate
            if abs(corr_matrix[i, j]) >= soglia:
                # Confronta le varianze per decidere quale eliminare
                if feature_variances[i] < feature_variances[j]:
                    feature_da_eliminare.add(i)
                    break  # Esci dal loop interno se la feature i è eliminata
                else:
                    feature_da_eliminare.add(j)

    # Elimina le feature dal dataset
    X_ridotto = np.delete(X, list(feature_da_eliminare), axis=1)

    return X_ridotto, feature_da_eliminare



def addestra_modelli(X_train, y_train, **best_params):
    # Modello Logistic Regression implementato
    model = LogisticRegressionGD()
    model.set_params(**best_params)
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


def bayesian_optimization(X_train, y_train):
    # Definisci lo spazio degli iperparametri da ottimizzare
    param_space = {
        'learning_rate': (1e-4, 1e-1, 'log-uniform'),
        'lambda_': (1e-4, 10e2, 'log-uniform'),
        'n_iterations': (100, 10000),
        'regularization': ['ridge', 'lasso', 'none']
    }


    bayes_search = BayesSearchCV(
        estimator=LogisticRegressionGD(),
        search_spaces=param_space,
        n_iter=150,
        cv=5,
        scoring='neg_log_loss',
        n_jobs=-1,
        random_state=42
    )

    bayes_search.fit(X_train, y_train)

    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    return best_params, best_score


def save_best_params(best_params, file_path="best_parameters.json"):
    with open(file_path, 'w') as file:
        json.dump(best_params, file)
    print(f"Parametri salvati in {file_path}.")


def load_best_params(X_train=None, y_train=None, file_path="Assets/best_parameters.json"):
    # Controllo se esiste il file con i parametri salvati
    if os.path.exists(file_path):
        print(f"Caricamento dei parametri ottimali da {file_path}...")
        with open(file_path, 'r') as file:
            best_params = json.load(file)
        best_score = best_params.pop("accuracy", None)  # Rimuovi 'accuracy' se presente
    elif not os.path.exists(file_path) and X_train is not None and y_train is not None:
        # Eseguire l'ottimizzazione bayesiana se il file non esiste
        print("Eseguendo l'ottimizzazione bayesiana...")
        best_params, best_score = bayesian_optimization(X_train, y_train)
        save_best_params(best_params, file_path)
    else:
        raise FileNotFoundError(f"Il file {file_path} non esiste. Assicurati di aver salvato gli iperparametri.")

    return best_params, best_score

def stampa_metriche_ordinate(metriche_modello1, metriche_modello2, file_path="Assets/"):
    # Creazione della lista delle metriche
    lista_metriche = [metriche_modello1, metriche_modello2]

    # Creazione del DataFrame escludendo 'conf_matrix'
    df_metriche = pd.DataFrame(lista_metriche).set_index('model_name').drop(columns=['conf_matrix'])

    # Ordinare le colonne se necessario
    df_metriche = df_metriche[sorted(df_metriche.columns)]

    # Stampare il DataFrame
    print(df_metriche)

    # Salvataggio su file
    # Controlla se la directory esiste, altrimenti la crea
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Salva il DataFrame in un file CSV
    csv_file = os.path.join(file_path, 'metriche_modelli.csv')
    df_metriche.to_csv(csv_file)

    for metriche in lista_metriche:
        print(f"Matrice di confusione per {metriche['model_name']}:")
        print(metriche['conf_matrix'])
        print()
