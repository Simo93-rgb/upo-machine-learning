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
    predictions = model.predict(X_val)

    # Valutazione Cross-Validation
    k_fold_accuracy = k_fold_cross_validation(model, X_train, y_train, k)
    stratified_k_fold_accuracy = stratified_k_fold_cross_validation(model, X_train, y_train, n_splits=5)

    print(f"Accuratezza media con Stratified 5-Fold Cross-Validation: {stratified_k_fold_accuracy}")
    print(f"Accuratezza con K-Fold {k} Cross-Validation: {k_fold_accuracy}")

    # Esegui Leave-One-Out Cross-Validation (solo se necessario)
    loo_accuracy = leave_one_out_cross_validation(model, X_train, y_train)
    print(f"Accuratezza con Leave-One-Out Cross-Validation: {loo_accuracy}")

    # Modello di scikit-learn Logistic Regression
    sk_model = LogisticRegression(max_iter=100)
    sk_model.fit(X_train, y_train)
    sk_predictions = sk_model.predict(X_val)

    return model, predictions, sk_model, sk_predictions


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
        estimator=model_fit(learning_rate=0.01, lambda_=0.1, n_iterations=1000, regularization='ridge'),
        search_spaces=param_space,
        n_iter=50,
        cv=5,
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
        best_score = None  # Se hai il best_score salvato puoi caricarlo anche qui
    else:
        # Eseguire l'ottimizzazione bayesiana se il file non esiste
        print("Eseguendo l'ottimizzazione bayesiana...")
        best_params, best_score = bayesian_optimization(X_train, y_train)
        save_best_params(best_params, file_path)

    return best_params, best_score


def save_best_params(best_params, file_path="best_parameters.json"):
    with open(file_path, 'w') as file:
        json.dump(best_params, file)
    print(f"Parametri salvati in {file_path}.")


if __name__ == "__main__":
    start_time = time.time()

    # Carica e pre-processa i dati
    X, y = carica_dati()
    X_normalized, features_eliminate, y_encoded = preprocessa_dati(X, y, class_balancer="")

    # Suddivisione in train (70%), validation (20%) e test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    # Eseguire l'ottimizzazione bayesiana sugli iperparametri
    best_params, best_score = bayesian_optimization_wrapper(X_train, y_train)

    print(f"Migliori iperparametri trovati: {best_params}")
    print(f"Accuracy del modello ottimizzato (validazione): {best_score}")

    # Misura il tempo di esecuzione del tuo modello LogisticRegressionGD
    start_model_time = time.time()

    model, predictions, sk_model, sk_predictions = addestra_modelli(
        X_train, y_train, X_val, best_params, k=5)

    end_model_time = time.time()
    print(f"\nTempo di esecuzione del modello Logistic Implementato: {end_model_time - start_model_time:.4f} secondi")

    # Valutazione finale sul Test Set
    print("\nValutazione finale sul Test Set:")

    test_predictions = model.predict(X_test)
    auc = validation_test(test_predictions, X_test, y_test, model, model_name="Modello Logistic Implementato")

    test_sk_predictions = sk_model.predict(X_test)
    sk_auc = validation_test(test_sk_predictions, X_test, y_test, sk_model, model_name="Modello Scikit-learn")

    print(f"\nTempo di esecuzione del modello Scikit-learn: {end_model_time - start_model_time:.4f} secondi")

    # Plottare la funzione sigmoidale
    plot_sigmoid()

    # Matrici di confusione
    plot_confusion_matrix(y_test, test_predictions, "Modello Logistic Implementato")
    plot_confusion_matrix(y_test, test_sk_predictions, "Modello Scikit-learn")

    # Curve ROC
    y_probs = model.sigmoid(np.dot(X_test, model.theta) + model.bias)
    plot_roc_curve(y_test, y_probs, "Modello Logistic Implementato")

    y_sk_probs = sk_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_sk_probs, "Modello Scikit-learn")

    plot_precision_recall(y_test, test_predictions, model_name="Modello Logistic Implementato")
    plot_precision_recall(y_test, test_sk_predictions, model_name="Modello Scikit-learn")
    # Confronto delle metriche
    metrics_dict = {
        "Modello Logistic Implementato": {
            "Precision": precision_score(y_test, test_predictions),
            "Recall": recall_score(y_test, test_predictions),
            "F1-Score": f1_score(y_test, test_predictions),
            "AUC": auc
        },
        "Modello Scikit-learn": {
            "Precision": precision_score(y_test, test_sk_predictions),
            "Recall": recall_score(y_test, test_sk_predictions),
            "F1-Score": f1_score(y_test, test_sk_predictions),
            "AUC": sk_auc
        }
    }

    plot_metrics_comparison(metrics_dict, ["Modello Logistic Implementato", "Modello Scikit-learn"])

    # Plottare l'effetto della regolarizzazione
    lambdas = np.logspace(-4, 2, 100)  # Lista di valori di lambda su scala logaritmica
    plot_regularization_effect(X_train, y_train, lambdas, regularization_type='ridge')
    plot_regularization_effect(X_train, y_train, lambdas, regularization_type='lasso')

    end_time = time.time()
    print(f"\nTempo di esecuzione totale: {end_time - start_time:.4f} secondi")
