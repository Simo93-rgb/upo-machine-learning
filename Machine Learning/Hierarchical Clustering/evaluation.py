import os
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np

# Determina il percorso della cartella "Assets/results"
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'Assets', 'results')
os.makedirs(results_dir, exist_ok=True)  # Crea la cartella se non esiste


def calculate_and_save_silhouette(X, labels, file_name="silhouette_score.csv"):
    # Calcola lo score di silhouette
    silhouette_avg = silhouette_score(X, labels)

    # Salva in un file CSV
    file_path = os.path.join(results_dir, file_name)
    pd.DataFrame({"Silhouette Score": [silhouette_avg]}).to_csv(file_path, index=False)
    print(f"Silhouette Score salvato in {file_path}")


def purity_score(y_true, y_pred, file_name="purity_score.csv"):
    # Calcolo del purity score
    contingency_matrix = np.zeros((np.unique(y_true).size, np.unique(y_pred).size))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    # Salva in un file CSV
    file_path = os.path.join(results_dir, file_name)
    pd.DataFrame({"Purity Score": [purity]}).to_csv(file_path, index=False)
    print(f"Purity Score salvato in {file_path}")

    return purity
