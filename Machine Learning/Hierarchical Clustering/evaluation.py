# import os
# import pandas as pd
# from sklearn.metrics import silhouette_score
# import numpy as np
# from sklearn.metrics import confusion_matrix, silhouette_score, silhouette_samples
# from typing import List, Tuple
# import os
# import csv
#
# # Determina il percorso della cartella "Assets/results"
# current_dir = os.path.dirname(os.path.abspath(__file__))
# results_dir = os.path.join(current_dir, 'Assets', 'Results')
# plot_dir = os.path.join(current_dir, 'Assets', 'Plot')
# os.makedirs(results_dir, exist_ok=True)  # Crea la cartella se non esiste
# os.makedirs(plot_dir, exist_ok=True)  # Crea la cartella se non esiste
#
#
# def calculate_and_save_silhouette(X, labels, file_name="silhouette_score.csv"):
#     # Calcola lo score di silhouette
#     silhouette_avg = silhouette_score(X, labels)
#
#     # Salva in un file CSV
#     file_path = os.path.join(results_dir, file_name)
#     pd.DataFrame({"Silhouette Score": [silhouette_avg]}).to_csv(file_path, index=False)
#     print(f"Silhouette Score salvato in {file_path}")
#
#
# def purity_score(y_true, y_pred, results_dir='.', file_name='purity_score.csv'):
#     # Converti le etichette string in interi
#     unique_true = np.unique(y_true)
#     unique_pred = np.unique(y_pred)
#     true_to_int = {label: index for index, label in enumerate(unique_true)}
#     pred_to_int = {label: index for index, label in enumerate(unique_pred)}
#
#     y_true_int = np.array([true_to_int[label] for label in y_true])
#     y_pred_int = np.array([pred_to_int[label] for label in y_pred])
#
#     # Calcolo del purity score
#     contingency_matrix = np.zeros((len(unique_true), len(unique_pred)))
#     for i in range(len(y_true_int)):
#         contingency_matrix[y_true_int[i], y_pred_int[i]] += 1
#     purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
#
#     # Salva in un file CSV
#     file_path = os.path.join(results_dir, file_name)
#     pd.DataFrame({'Purity Score': [purity]}).to_csv(file_path, index=False)
#     print(f'Purity Score salvato in {file_path}')
#
#     return purity
#
#
# def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
#     """
#     Calcola la matrice di confusione.
#
#     Args:
#         y_true (np.ndarray): Etichette vere.
#         y_pred (np.ndarray): Etichette predette.
#
#     Returns:
#         np.ndarray: Matrice di confusione.
#     """
#     # Converti le etichette string in interi
#     unique_true = np.unique(y_true)
#     unique_pred = np.unique(y_pred)
#     true_to_int = {label: index for index, label in enumerate(unique_true)}
#     pred_to_int = {label: index for index, label in enumerate(unique_pred)}
#
#     y_true_int = np.array([true_to_int[label] for label in y_true])
#     y_pred_int = np.array([pred_to_int[label] for label in y_pred])
#     return confusion_matrix(y_true_int, y_pred_int)
#
#
# def calculate_purity(confusion_matrix: np.ndarray) -> float:
#     """
#     Calcola la purezza dei cluster basata sulla matrice di confusione.
#
#     Args:
#         confusion_matrix (np.ndarray): Matrice di confusione.
#
#     Returns:
#         float: Punteggio di purezza.
#     """
#     return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
#
#
# def calculate_f1_score(confusion_matrix: np.ndarray) -> float:
#     """
#     Calcola lo F1-score basato sulla matrice di confusione.
#
#     Args:
#         confusion_matrix (np.ndarray): Matrice di confusione.
#
#     Returns:
#         float: F1-score medio.
#     """
#     precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
#     recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
#     f1 = 2 * (precision * recall) / (precision + recall)
#     return np.mean(f1[~np.isnan(f1)])
#
#
# def calculate_false_positive_rate(confusion_matrix: np.ndarray) -> float:
#     """
#     Calcola il tasso di falsi positivi basato sulla matrice di confusione.
#
#     Args:
#         confusion_matrix (np.ndarray): Matrice di confusione.
#
#     Returns:
#         float: Tasso di falsi positivi medio.
#     """
#     fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
#     tn = confusion_matrix.sum() - (fp + confusion_matrix.sum(axis=1))
#     fpr = fp / (fp + tn)
#     return np.mean(fpr)
#
#
# def calculate_silhouette(X: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
#     """
#     Calcola il punteggio di silhouette e i valori di silhouette per ogni campione.
#
#     Args:
#         X (np.ndarray): Dati di input.
#         labels (np.ndarray): Etichette dei cluster.
#
#     Returns:
#         Tuple[float, np.ndarray]: Punteggio di silhouette medio e valori di silhouette per ogni campione.
#     """
#     silhouette_avg = silhouette_score(X, labels)
#     sample_silhouette_values = silhouette_samples(X, labels)
#     return silhouette_avg, sample_silhouette_values
#
#
# def find_optimal_clusters(X: np.ndarray, max_clusters: int, clustering_func, plot_dir:str) -> int:
#     """
#     Trova il numero ottimale di cluster usando il punteggio di silhouette.
#
#     Args:
#         X (np.ndarray): Dati di input.
#         max_clusters (int): Numero massimo di cluster da provare.
#         clustering_func: Funzione che esegue il clustering e restituisce le etichette.
#
#     Returns:
#         int: Numero ottimale di cluster.
#     """
#     silhouette_scores = []
#     for n_clusters in range(2, max_clusters + 1):
#         labels = clustering_func(n_clusters)
#         silhouette_avg, _ = calculate_silhouette(X, labels)
#         silhouette_scores.append(silhouette_avg)
#         from plot import save_silhouette_plot
#         save_silhouette_plot(X, labels, n_clusters, plot_dir)
#
#     return silhouette_scores.index(max(silhouette_scores)) + 2
#
#
# def save_evaluation_results(results: dict, file_name: str, output_dir: str):
#     """
#     Salva i risultati della valutazione in un file CSV.
#
#     Args:
#         results (dict): Dizionario contenente i risultati della valutazione.
#         file_name (str): Nome del file di output.
#         output_dir (str): Directory di output.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     file_path = os.path.join(output_dir, file_name)
#     with open(file_path, 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=results.keys())
#         writer.writeheader()
#         writer.writerow(results)
#     print(f"Risultati della valutazione salvati in {file_path}")
import numpy as np
from sklearn.metrics import confusion_matrix, silhouette_score, silhouette_samples, adjusted_rand_score
from typing import Tuple, Dict
import os
import csv


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcola la matrice di confusione.

    Args:
        y_true (np.ndarray): Etichette vere.
        y_pred (np.ndarray): Etichette predette.

    Returns:
        np.ndarray: Matrice di confusione.
    """
    return confusion_matrix(y_true, y_pred)


def calculate_purity(confusion_matrix: np.ndarray) -> float:
    """
    Calcola la purezza dei cluster basata sulla matrice di confusione.

    Args:
        confusion_matrix (np.ndarray): Matrice di confusione.

    Returns:
        float: Punteggio di purezza.
    """
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)


def calculate_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcola l'indice di Rand.

    Args:
        y_true (np.ndarray): Etichette vere.
        y_pred (np.ndarray): Etichette predette.

    Returns:
        float: Indice di Rand.
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(y_true, y_pred)


def calculate_precision_recall_f1(confusion_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcola precisione, richiamo e F1-score.

    Args:
        confusion_matrix (np.ndarray): Matrice di confusione.

    Returns:
        Tuple[float, float, float]: Precisione, richiamo e F1-score.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

        precision = np.nan_to_num(precision, 0)
        recall = np.nan_to_num(recall, 0)

        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1, 0)

    return np.mean(precision), np.mean(recall), np.mean(f1)


def calculate_tp_fp_tn_fn(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calcola TP, FP, TN, FN per clustering.

    Args:
        y_true (np.ndarray): Etichette vere.
        y_pred (np.ndarray): Etichette predette.

    Returns:
        Tuple[int, int, int, int]: TP, FP, TN, FN
    """
    tp = fp = tn = fn = 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j] and y_pred[i] == y_pred[j]:
                tp += 1
            elif y_true[i] != y_true[j] and y_pred[i] != y_pred[j]:
                tn += 1
            elif y_true[i] == y_true[j] and y_pred[i] != y_pred[j]:
                fn += 1
            else:
                fp += 1
    return tp, fp, tn, fn


def calculate_silhouette(X: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calcola il punteggio di silhouette e i valori di silhouette per ogni campione.

    Args:
        X (np.ndarray): Dati di input.
        labels (np.ndarray): Etichette dei cluster.

    Returns:
        Tuple[float, np.ndarray]: Punteggio di silhouette medio e valori di silhouette per ogni campione.
    """
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    return silhouette_avg, sample_silhouette_values


def find_optimal_clusters(X: np.ndarray, max_clusters: int, clustering_func, plot_dir: str) -> int:
    """
    Trova il numero ottimale di cluster usando il punteggio di silhouette.

    Args:
        X (np.ndarray): Dati di input.
        max_clusters (int): Numero massimo di cluster da provare.
        clustering_func: Funzione che esegue il clustering e restituisce le etichette.
        plot_dir (str): Directory di salvataggio del plot

    Returns:
        int: Numero ottimale di cluster.
    """
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        labels = clustering_func(n_clusters)
        silhouette_avg, _ = calculate_silhouette(X, labels)
        silhouette_scores.append(silhouette_avg)
        from plot import save_silhouette_plot
        save_silhouette_plot(X, labels, n_clusters, plot_dir)

    return silhouette_scores.index(max(silhouette_scores)) + 2


def save_evaluation_results(results: dict, file_name: str, output_dir: str):
    """
    Salva i risultati della valutazione in un file CSV.

    Args:
        results (dict): Dizionario contenente i risultati della valutazione.
        file_name (str): Nome del file di output.
        output_dir (str): Directory di output.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"Risultati della valutazione salvati in {file_path}")


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray, model_name: str = "") -> Dict[
    str, float]:
    # Converti le etichette string in interi
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    true_to_int = {label: index for index, label in enumerate(unique_true)}
    pred_to_int = {label: index for index, label in enumerate(unique_pred)}

    y_true_int = np.array([true_to_int[label] for label in y_true])
    y_pred_int = np.array([pred_to_int[label] for label in y_pred])

    # Calcola la matrice di confusione
    conf_matrix = confusion_matrix(y_true_int, y_pred_int)

    # Calcola la purezza
    purity = np.sum(np.amax(conf_matrix, axis=0)) / np.sum(conf_matrix)

    # Calcola precisione, richiamo e F1-score
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
        recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)

    # precision = np.nan_to_num(precision, 0)
    # recall = np.nan_to_num(recall, 0)
    # f1 = np.nan_to_num(f1, 0)

    # Calcola l'indice di Rand aggiustato
    ari = adjusted_rand_score(y_true_int, y_pred_int)

    # Calcola il coefficiente di silhouette
    silhouette_avg = silhouette_score(X, y_pred_int)

    # Calcola TP, FP, TN, FN
    tp, fp, tn, fn = calculate_tp_fp_tn_fn(y_true, y_pred)

    # Calcola FP rate e FN rate
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'purity': purity,
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1),
        'adjusted_rand_index': ari,
        'silhouette_score': silhouette_avg,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'model_name': model_name
    }
