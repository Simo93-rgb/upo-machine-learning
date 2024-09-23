import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, cross_val_score
from valutazione import evaluate_model, calculate_auc, calculate_auc
from logistic_regression_with_gradient_descend import LogisticRegressionGD
from ModelName import ModelName
import pandas as pd

def k_fold_cross_validation(X, y, model_enum, k=5) -> tuple[dict, dict]:
    model = LogisticRegressionGD(n_iterations=1000)
    sk_model = LogisticRegression(max_iter=1000)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    metrics_list = []
    sk_metrics_list = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        sk_model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        sk_predictions = sk_model.predict(X_val)

        # Valuta i modelli e accumula i risultati
        metrics = evaluate_model(model, X_val, predictions, y_val, model_enum.LOGISTIC_REGRESSION_GD.value)
        sk_metrics = evaluate_model(model, X_val, sk_predictions, y_val, model_enum.SCIKIT_LEARN.value)

        metrics_list.append(metrics)
        sk_metrics_list.append(sk_metrics)

    # Converti metrics_list e sk_metrics_list in DataFrame Pandas
    metrics_df = pd.DataFrame(metrics_list)
    sk_metrics_df = pd.DataFrame(sk_metrics_list)

    # Separa le metriche numeriche da quelle non numeriche
    numeric_metrics = ['precision', 'recall', 'fp_rate', 'f1_score', 'mcc', 'auc', 'prc_auc']
    # non_numeric_metrics = ['conf_matrix', 'model_name']
    non_numeric_metrics = ['model_name']

    # Calcola la media delle metriche numeriche con alta precisione
    mean_metrics = metrics_df[numeric_metrics].mean().apply(lambda x: round(x, 4)).to_dict()
    sk_mean_metrics = sk_metrics_df[numeric_metrics].mean().apply(lambda x: round(x, 4)).to_dict()

    # Ripristina i dizionari originali, mantenendo i campi non numerici invariati
    final_metrics_dict = {**mean_metrics, 'model_name': model_enum.LOGISTIC_REGRESSION_GD.value}
    final_sk_metrics_dict = {**sk_mean_metrics, 'model_name': model_enum.SCIKIT_LEARN.value}

    return final_metrics_dict, final_sk_metrics_dict




def leave_one_out_cross_validation(X, y):
    model = LogisticRegressionGD()
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


# Funzione per Stratified K-Fold Cross-Validation
def stratified_k_fold_cross_validation(model, X_train, y_train, n_splits=5):
    stratified_kfold = StratifiedKFold(n_splits=n_splits)
    stratified_scores = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    return stratified_scores.mean()
