import time
import numpy as np
from validazione import k_fold_cross_validation, leave_one_out_cross_validation
from valutazione import evaluate_model, calculate_auc, plot_roc_curve, calculate_auc_sklearn, plot_roc_curve_sklearn
from logistic_regression_with_gradient_descend import LogisticRegressionGD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
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

    # Encoding delle classi
    label_encoder = LabelEncoder()
    y_encoded = (label_encoder.fit_transform(y))
    y_encoded= y_encoded.ravel()

    return X_normalized, y_encoded


def addestra_modelli(X_train, y_train, X_val, y_val, learning_rate, n_iterations, _lambda, regularization):
    # Modello Logistic Regression implementato
    model = LogisticRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations, lambda_=_lambda,
                                 regularization=regularization)
    model.fit(X_train, y_train)

    # Predizione con il modello implementato
    predictions = model.predict(X_val)

    # Modello scikit-learn
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

    # Definisci la funzione obiettivo per la cross-validation
    def objective(params):
        model = model_fit(**params)
        # Valutazione del modello usando cross-validation
        score = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))
        return score

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


def main():
    start_time = time.time()

    # Parametri iniziali
    _lambda = 0.1
    learning_rate = 0.01
    n_iterations = 100
    regularization = 'ridge'
    k = 5  # numero di fold per la cross-validation

    # Carica e pre-processa i dati
    X, y = carica_dati()
    X_normalized, y_encoded = preprocessa_dati(X, y)

    # Calcola la matrice di correlazione
    corr_matrix = np.corrcoef(X_normalized, rowvar=False)

    # Visualizza la matrice di correlazione
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matrice di Correlazione delle Feature")
    # plt.show()

    # Salva l'immagine come PNG
    plt.savefig('matrice_correlazione.png', format='png', dpi=600)
    plt.show()
    plt.close()

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_encoded, test_size=0.3, random_state=42)

    # Eseguire l'ottimizzazione bayesiana sugli iperparametri
    best_params, best_score = bayesian_optimization(X_train, y_train)

    # Stampa dei risultati migliori dell'ottimizzazione
    print(f"Migliori iperparametri trovati: {best_params}")
    print(f"Accuracy del modello ottimizzato: {best_score}")

    # Usa i migliori iperparametri per addestrare i modelli
    model, predictions, sk_model, sk_predictions = addestra_modelli(
        X_train, y_train, X_val, y_val,
        learning_rate=best_params['learning_rate'],
        n_iterations=best_params['n_iterations'],
        _lambda=best_params['lambda_'],
        regularization=best_params['regularization']
    )

    # Valutazione del modello Logistic Implementato
    evaluate_model(predictions, y_val, model_name="Modello Logistic Implementato")
    auc = calculate_auc(model, X_val, y_val)
    plot_roc_curve(model, X_val, y_val, model_name="Modello Logistic Implementato")

    # Valutazione del modello Scikit-learn
    evaluate_model(sk_predictions, y_val, model_name="Modello Scikit-learn")
    auc_sk = calculate_auc_sklearn(sk_model, X_val, y_val)
    plot_roc_curve_sklearn(sk_model, X_val, y_val, model_name="Modello Scikit-learn")

    # Cross-Validation
    k_fold_accuracy = k_fold_cross_validation(
        LogisticRegressionGD(learning_rate=best_params['learning_rate'], n_iterations=best_params['n_iterations'],
                             lambda_=_lambda, regularization=regularization), X_normalized, y_encoded, k=k)
    loo_accuracy = leave_one_out_cross_validation(
        LogisticRegressionGD(learning_rate=best_params['learning_rate'], n_iterations=best_params['n_iterations'],
                             lambda_=_lambda, regularization=regularization), X_normalized, y_encoded)

    print(f'K-Fold Cross-Validated Accuracy: {k_fold_accuracy}')
    print(f'Leave-One-Out Cross-Validated Accuracy: {loo_accuracy}')

    # Tempo di esecuzione
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
