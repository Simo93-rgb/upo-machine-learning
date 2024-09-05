import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Funzione per valutare precision, recall e F1
def evaluate_model(predictions, y_val, model_name=""):
    print(f"\nValutazione {model_name}")
    conf_matrix = confusion_matrix(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)

    print(f"Matrice di confusione:\n{conf_matrix}")
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')

# Funzione per calcolare l'AUC (modello personalizzato)
def calculate_auc(model, X_val, y_val):
    y_probs = model.sigmoid(np.dot(X_val, model.theta) + model.bias)
    auc = roc_auc_score(y_val, y_probs)
    print(f'AUC: {auc:.2f}')
    return auc

# Funzione per calcolare l'AUC (modello scikit-learn)
def calculate_auc_sklearn(model, X_val, y_val):
    y_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_probs)
    print(f'AUC Scikit-learn: {auc:.2f}')
    return auc

# Funzione per tracciare la curva ROC (modello personalizzato)
def plot_roc_curve(model, X_val, y_val, model_name=""):
    y_probs = model.sigmoid(np.dot(X_val, model.theta) + model.bias)
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    auc = roc_auc_score(y_val, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve {model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Funzione per tracciare la curva ROC (modello scikit-learn)
def plot_roc_curve_sklearn(model, X_val, y_val, model_name=""):
    y_probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    auc = roc_auc_score(y_val, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve {model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
