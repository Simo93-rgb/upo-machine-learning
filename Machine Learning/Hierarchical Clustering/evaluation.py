import os
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import confusion_matrix, silhouette_score, silhouette_samples
from typing import List, Tuple
import os
import csv

# Determina il percorso della cartella "Assets/results"
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'Assets', 'Results')
plot_dir = os.path.join(current_dir, 'Assets', 'Plot')
os.makedirs(results_dir, exist_ok=True)  # Crea la cartella se non esiste
os.makedirs(plot_dir, exist_ok=True)  # Crea la cartella se non esiste


def calculate_and_save_silhouette(X, labels, file_name="silhouette_score.csv"):
    # Calcola lo score di silhouette
    silhouette_avg = silhouette_score(X, labels)

    # Salva in un file CSV
    file_path = os.path.join(results_dir, file_name)
    pd.DataFrame({"Silhouette Score": [silhouette_avg]}).to_csv(file_path, index=False)
    print(f"Silhouette Score salvato in {file_path}")


def purity_score(y_true, y_pred, results_dir='.', file_name='purity_score.csv'):
    # Converti le etichette string in interi
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    true_to_int = {label: index for index, label in enumerate(unique_true)}
    pred_to_int = {label: index for index, label in enumerate(unique_pred)}

    y_true_int = np.array([true_to_int[label] for label in y_true])
    y_pred_int = np.array([pred_to_int[label] for label in y_pred])

    # Calcolo del purity score
    contingency_matrix = np.zeros((len(unique_true), len(unique_pred)))
    for i in range(len(y_true_int)):
        contingency_matrix[y_true_int[i], y_pred_int[i]] += 1
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    # Salva in un file CSV
    file_path = os.path.join(results_dir, file_name)
    pd.DataFrame({'Purity Score': [purity]}).to_csv(file_path, index=False)
    print(f'Purity Score salvato in {file_path}')

    return purity


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcola la matrice di confusione.

    Args:
        y_true (np.ndarray): Etichette vere.
        y_pred (np.ndarray): Etichette predette.

    Returns:
        np.ndarray: Matrice di confusione.
    """
    # Converti le etichette string in interi
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    true_to_int = {label: index for index, label in enumerate(unique_true)}
    pred_to_int = {label: index for index, label in enumerate(unique_pred)}

    y_true_int = np.array([true_to_int[label] for label in y_true])
    y_pred_int = np.array([pred_to_int[label] for label in y_pred])
    return confusion_matrix(y_true_int, y_pred_int)


def calculate_purity(confusion_matrix: np.ndarray) -> float:
    """
    Calcola la purezza dei cluster basata sulla matrice di confusione.

    Args:
        confusion_matrix (np.ndarray): Matrice di confusione.

    Returns:
        float: Punteggio di purezza.
    """
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)


def calculate_f1_score(confusion_matrix: np.ndarray) -> float:
    """
    Calcola lo F1-score basato sulla matrice di confusione.

    Args:
        confusion_matrix (np.ndarray): Matrice di confusione.

    Returns:
        float: F1-score.
    """
    # Implementa il calcolo dell'F1-score
    pass


def calculate_false_positive_rate(confusion_matrix: np.ndarray) -> float:
    """
    Calcola il tasso di falsi positivi basato sulla matrice di confusione.

    Args:
        confusion_matrix (np.ndarray): Matrice di confusione.

    Returns:
        float: Tasso di falsi positivi.
    """
    # Implementa il calcolo del tasso di falsi positivi
    pass


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


def find_optimal_clusters(X: np.ndarray, max_clusters: int, clustering_func) -> int:
    """
    Trova il numero ottimale di cluster usando il punteggio di silhouette.

    Args:
        X (np.ndarray): Dati di input.
        max_clusters (int): Numero massimo di cluster da provare.
        clustering_func: Funzione che esegue il clustering e restituisce le etichette.

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
