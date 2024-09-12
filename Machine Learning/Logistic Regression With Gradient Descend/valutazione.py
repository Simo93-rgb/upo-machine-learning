import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
    matthews_corrcoef



# Funzione per valutare precision, recall e F1
def evaluate_model(predictions, y_val, model_name=""):
    print(f"\nValutazione {model_name}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_val, predictions)
    print(f"Matrice di confusione:\n{conf_matrix}")

    # Precision, Recall, F1-Score
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)  # Identico a TP Rate
    f1 = f1_score(y_val, predictions)

    # False Positive Rate
    fp_rate = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[0][0])  # FP / (FP + TN)

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_val, predictions)

    # AUC (Area under ROC curve)
    auc = roc_auc_score(y_val, predictions)

    # Precision-Recall AUC
    precision_recall_curve_auc = roc_auc_score(y_val, predictions)

    # Stampa i risultati
    print(f'Precision: {precision}')
    print(f'Recall (TP Rate): {recall}')
    print(f'False Positive Rate: {fp_rate}')
    print(f'F1-Score: {f1}')
    print(f'MCC: {mcc}')
    print(f'AUC: {auc}')
    print(f'Precision-Recall AUC: {precision_recall_curve_auc}')

    return {
        'precision': precision,
        'recall': recall,
        'fp_rate': fp_rate,
        'f1_score': f1,
        'mcc': mcc,
        'auc': auc,
        'prc_auc': precision_recall_curve_auc
    }


def cohen_kappa(y_val, y_pred):
    # Calcola la matrice di confusione
    conf_matrix = confusion_matrix(y_val, y_pred)

    # Totale dei campioni
    n = np.sum(conf_matrix)

    # Calcola le probabilità marginali (per le righe e le colonne della matrice di confusione)
    p0 = np.sum(np.diag(conf_matrix)) / n  # Accord osservato (percentuale di corrispondenza)

    # Frequenze marginali (righe e colonne)
    pe_rows = np.sum(conf_matrix, axis=1) / n
    pe_cols = np.sum(conf_matrix, axis=0) / n

    # Calcolo dell'accordo atteso
    pe = np.sum(pe_rows * pe_cols)

    # Formula di Cohen's Kappa
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) != 0 else 1.0

    return kappa


# Funzione per calcolare l'AUC (modello personalizzato)
def calculate_auc(model, X_val, y_val):
    y_probs = model.sigmoid(np.dot(X_val, model.theta) + model.bias)
    auc = roc_auc_score(y_val, y_probs)
    print(f'AUC: {auc:.3f}')
    return auc


# Funzione per calcolare l'AUC (modello scikit-learn)
def calculate_auc_sklearn(model, X_val, y_val):
    y_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_probs)
    print(f'AUC Scikit-learn: {auc:.3f}')
    return auc



