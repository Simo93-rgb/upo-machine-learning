import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, cross_val_score
from valutazione import evaluate_model, calculate_auc, calculate_auc_sklearn
from logistic_regression_with_gradient_descend import LogisticRegressionGD

def k_fold_cross_validation(X, y, k=5):
    model = LogisticRegressionGD()
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


def validation_test(predictions, X_val, y_val, model, model_name=""):
    if model_name == "Modello Scikit-learn":
        evaluate_model(predictions, y_val, model_name="Modello Scikit-learn")
        auc_sk = calculate_auc_sklearn(model, X_val, y_val)
        return auc_sk
    else:
        evaluate_model(predictions, y_val, model_name=model_name)
        auc = calculate_auc(model, X_val, y_val)
        return auc
