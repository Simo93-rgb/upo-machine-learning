import time
import numpy as np
from validazione import k_fold_cross_validation, leave_one_out_cross_validation
from valutazione import evaluate_model, calculate_auc, plot_roc_curve, calculate_auc_sklearn, plot_roc_curve_sklearn
from logistic_regression_with_gradient_descend import LogisticRegressionGD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


def carica_dati():
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    return X, y


def preprocessa_dati(X, y):
    # Imputazione dei NaN e normalizzazione
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_normalized = (X_imputed - X_imputed.mean(axis=0)) / X_imputed.std(axis=0)
    (X_ridotto, features_eliminate) = elimina_feature_correlate(X_normalized)
    print(f"Features eliminate: {features_eliminate}")
    X_normalized = X_ridotto

    # Encoding delle classi
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).ravel()

    return X_normalized, y_encoded


def plot_corr_matrix(corr_matrix):
    # Visualizza la matrice di correlazione
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 10})
    plt.title("Matrice di Correlazione delle Feature")

    # Salva l'immagine come PNG
    plt.savefig('matrice_correlazione_breast_cancer.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


# elimina features correlate tramite matrice di correlazione
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


def addestra_modelli(X_train, y_train, X_val, best_params, k):
    # Estrai i migliori iperparametri trovati con l'ottimizzazione bayesiana
    learning_rate = best_params['learning_rate']
    n_iterations = best_params['n_iterations']
    _lambda = best_params['lambda_']
    regularization = best_params['regularization']

    # Modello Logistic Regression implementato
    model = LogisticRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations, lambda_=_lambda,
                                 regularization=regularization)
    model.fit(X_train, y_train)

    # Predizione con il modello implementato
    predictions = model.predict(X_val)

    # Cross-Validation con K-Fold (solo per stimare l'accuratezza)
    k_fold_accuracy = k_fold_cross_validation(
        LogisticRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations,
                             lambda_=_lambda, regularization=regularization), X_train, y_train, k=k)

    # Stratified K-Fold Cross-Validation con K=5
    stratified_kfold = StratifiedKFold(n_splits=5)

    # Esegui la cross-validation stratificata
    stratified_k_fold_scores = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring='accuracy')

    # Calcola la media delle performance
    mean_stratified_score = stratified_k_fold_scores.mean()

    print(f"Accuratezza media con Stratified 5-Fold Cross-Validation: {mean_stratified_score}")
    # Cross-Validation con LOO (solo per stimare l'accuratezza)
    loo_accuracy = leave_one_out_cross_validation(
        LogisticRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations,
                             lambda_=_lambda, regularization=regularization), X_train, y_train)

    # Stampa le accuratezze delle cross-validation
    print(f"Accuratezza con K-Fold {k} Cross-Validation: {k_fold_accuracy}")
    print(f"Accuratezza con Leave-One-Out Cross-Validation: {loo_accuracy}")

    # Modello Logistic Regression di scikit-learn
    sk_model = LogisticRegression(max_iter=100)
    sk_model.fit(X_train, y_train)
    sk_predictions = sk_model.predict(X_val)

    return model, predictions, sk_model, sk_predictions


# Funzione per eseguire la ricerca bayesiana degli iperparametri
def bayesian_optimization(X_train, y_train):
    # Definisci lo spazio degli iperparametri da ottimizzare
    param_space = {
        'learning_rate': (1e-4, 1e-1, 'log-uniform'),  # Intervallo per il learning rate (logaritmico)
        'lambda_': (1e-4, 1e1, 'log-uniform'),  # Parametro di regolarizzazione (lambda)
        'n_iterations': (100, 1000),  # Numero massimo di iterazioni
        'regularization': ['ridge', 'lasso', 'none']  # Tipo di regolarizzazione
    }

    # Wrapper per il modello LogisticRegressionGD per l'uso con BayesSearchCV
    def model_fit(learning_rate, lambda_, n_iterations, regularization):
        model = LogisticRegressionGD(learning_rate=learning_rate, lambda_=lambda_,
                                     n_iterations=n_iterations, regularization=regularization)
        return model

    # BayesSearchCV per ottimizzare il modello
    bayes_search = BayesSearchCV(
        estimator=model_fit(learning_rate=0.01, lambda_=0.1, n_iterations=1000, regularization='ridge'),
        search_spaces=param_space,
        n_iter=32,  # Numero di iterazioni dell'ottimizzazione
        cv=5,  # Cross-validation con 5 fold
        scoring='accuracy',  # Misura di performance da ottimizzare
        n_jobs=-1,  # Usare tutti i core
        random_state=42
    )

    # Fitting del modello con ottimizzazione bayesiana
    bayes_search.fit(X_train, y_train)

    # Migliori iperparametri trovati
    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_

    return best_params, best_score


def validation_test(predictions, X_val, y_val, model, model_name=""):
    if model_name == "Modello Scikit-learn":
        # Valutazione del modello Scikit-learn
        evaluate_model(predictions, y_val, model_name="Modello Scikit-learn")
        auc_sk = calculate_auc_sklearn(model, X_val, y_val)
        plot_roc_curve_sklearn(model, X_val, y_val, model_name="Modello Scikit-learn")
        return auc_sk
    else:
        evaluate_model(predictions, y_val, model_name=model_name)
        auc = calculate_auc(model, X_val, y_val)
        plot_roc_curve(model, X_val, y_val, model_name=model_name)
        return auc


if __name__ == "__main__":
    start_time = time.time()

    # Carica e pre-processa i dati
    X, y = carica_dati()
    X_normalized, y_encoded = preprocessa_dati(X, y)

    # Split train/validation/test
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Eseguire l'ottimizzazione bayesiana sugli iperparametri
    best_params, best_score = bayesian_optimization(X_train, y_train)

    # Stampa dei risultati migliori dell'ottimizzazione
    print(f"Migliori iperparametri trovati: {best_params}")
    print(f"Accuracy del modello ottimizzato (validazione): {best_score}")

    # Usa i migliori iperparametri per addestrare i modelli
    model, predictions, sk_model, sk_predictions = addestra_modelli(
        X_train,
        y_train,
        X_val,
        best_params,
        k=5
    )

    # Valutazione finale sul Test Set
    print("\nValutazione finale sul Test Set:")

    # Valutazione del modello Logistic Implementato sul Test Set
    test_predictions = model.predict(X_test)
    auc = validation_test(test_predictions, X_test, y_test, model, model_name="Modello Logistic Implementato")

    # Valutazione del modello Logistic Scikit-learn sul Test Set
    test_sk_predictions = sk_model.predict(X_test)
    sk_auc = validation_test(test_sk_predictions, X_test, y_test, sk_model, model_name="Modello Scikit-learn")

    # Tempo di esecuzione
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
