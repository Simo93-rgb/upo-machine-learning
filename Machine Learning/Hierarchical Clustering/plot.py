import numpy as np
from scipy.cluster.hierarchy import dendrogram
import os
import matplotlib.pyplot as plt
from typing import List, Callable
import os
# Determina il percorso della cartella "Assets/plot"


# def save_dendrogram(linkage_matrix, file_name="dendrogram.png"):
#     # Percorso completo per il file
#     file_path = os.path.join(plot_dir, file_name)
#     # Plot del dendrogramma
#     plt.figure(figsize=(10, 7))
#     dendrogram(linkage_matrix)
#     plt.title("Dendrogram")
#     plt.xlabel("Samples")
#     plt.ylabel("Distance")
#     plt.savefig(file_path)  # Salva l'immagine
#     plt.close()  # Chiude la figura per liberare memoria
#     print(f"Dendrogramma salvato in {file_path}")
#
#
# def save_silhouette_plot(X, labels, file_name="silhouette_plot.png"):
#     from sklearn.metrics import silhouette_samples
#     import numpy as np
#
#     file_path = os.path.join(plot_dir, file_name)
#
#     # Calcolo dei campioni della silhouette
#     silhouette_vals = silhouette_samples(X, labels)
#     y_ticks = []
#     y_lower, y_upper = 0, 0
#
#     # Plot delle silhouette
#     plt.figure(figsize=(10, 7))
#     for i, cluster in enumerate(np.unique(labels)):
#         cluster_silhouette_vals = silhouette_vals[labels == cluster]
#         cluster_silhouette_vals.sort()
#         y_upper += len(cluster_silhouette_vals)
#         plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
#         y_ticks.append((y_lower + y_upper) / 2)
#         y_lower += len(cluster_silhouette_vals)
#
#     plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")  # Linea verticale per il valore medio
#     plt.yticks(y_ticks, np.unique(labels))
#     plt.xlabel("Silhouette Coefficient")
#     plt.ylabel("Cluster")
#     plt.title("Silhouette Plot")
#     plt.savefig(file_path)  # Salva l'immagine
#     plt.close()
#     print(f"Plot della silhouette salvato in {file_path}")



# def save_plot(plot, file_name: str, plot_dir: str):
#     """
#     Salva il plot corrente nella directory specificata.
#
#     Args:
#         plot: Oggetto matplotlib.pyplot.
#         file_name (str): Nome del file di output.
#         plot_dir (str): Directory di output per i plot.
#     """
#     os.makedirs(plot_dir, exist_ok=True)
#     file_path = os.path.join(plot_dir, file_name)
#     plot.savefig(file_path)
#     plot.close()
#     print(f"Plot salvato in {file_path}")
#
# def save_dendrogram(linkage_matrix: np.ndarray, plot_dir: str):
#     """
#     Crea e salva il dendrogramma.
#
#     Args:
#         linkage_matrix (List[List[float]]): Matrice di linkage per il dendrogramma.
#         k (int): Numero di cluster.
#         plot_dir (str): Directory di output per i plot.
#     """
#     plt.figure(figsize=(10, 7))
#     dendrogram(linkage_matrix)
#     plt.title(f"Dendrogram")
#     plt.xlabel("Samples")
#     plt.ylabel("Distance")
#     save_plot(plt, f"dendrogram.png", plot_dir)
#
# def save_silhouette_plot(X: np.ndarray, labels: np.ndarray, k: int, plot_dir: str):
#     """
#     Crea e salva il plot della silhouette.
#
#     Args:
#         X (np.ndarray): Dati di input.
#         labels (np.ndarray): Etichette dei cluster.
#         k (int): Numero di cluster.
#         plot_dir (str): Directory di output per i plot.
#     """
#     from sklearn.metrics import silhouette_samples
#     silhouette_vals = silhouette_samples(X, labels)
#     y_ticks = []
#     y_lower, y_upper = 0, 0
#
#     plt.figure(figsize=(10, 7))
#     for i, cluster in enumerate(np.unique(labels)):
#         cluster_silhouette_vals = silhouette_vals[labels == cluster]
#         cluster_silhouette_vals.sort()
#         y_upper += len(cluster_silhouette_vals)
#         plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
#         y_ticks.append((y_lower + y_upper) / 2)
#         y_lower += len(cluster_silhouette_vals)
#
#     plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
#     plt.yticks(y_ticks, np.unique(labels))
#     plt.xlabel("Silhouette Coefficient")
#     plt.ylabel("Cluster")
#     plt.title(f"Silhouette Plot (k={k})")
#     save_plot(plt, f"silhouette_plot_k={k}.png", plot_dir)


import networkx as nx


def save_plot(plot, file_name: str, plot_dir: str):
    """
    Salva il plot corrente nella directory specificata.

    Args:
        plot: Oggetto matplotlib.pyplot.
        file_name (str): Nome del file di output.
        plot_dir (str): Directory di output per i plot.
    """
    os.makedirs(plot_dir, exist_ok=True)
    file_path = os.path.join(plot_dir, file_name)
    plot.savefig(file_path)
    plot.close()
    print(f"Plot salvato in {file_path}")


def save_dendrogram(linkage_matrix: np.ndarray, plot_dir: str):
    """
    Crea e salva il dendrogramma.

    Args:
        linkage_matrix (np.ndarray): Matrice di linkage per il dendrogramma.
        plot_dir (str): Directory di output per i plot.
    """
    from scipy.cluster.hierarchy import dendrogram
    plt.figure(figsize=(10, 7))
    dizionario = dendrogram(linkage_matrix)
    plt.title("Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    save_plot(plt, "dendrogram.png", plot_dir)
    return len(set(dizionario['color_list']))


def plot_dendrogram(linkage_matrix: np.ndarray, plot_dir: str, n_clusters: int = 3):
    """
    Crea e salva il dendrogramma con un numero specifico di cluster colorati.

    Args:
        linkage_matrix (np.ndarray): Matrice di linkage per il dendrogramma.
        plot_dir (str): Directory di output per i plot.
        n_clusters (int): Numero desiderato di cluster da visualizzare.
    """
    from scipy.cluster.hierarchy import dendrogram
    plt.figure(figsize=(10, 7))

    # Calcola la soglia di taglio per ottenere il numero desiderato di cluster
    threshold = linkage_matrix[-(n_clusters - 1), 2]

    dendrogram_dict = dendrogram(linkage_matrix, color_threshold=threshold)

    plt.title(f"Dendrogram with {n_clusters} clusters")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")

    # Aggiungi una linea orizzontale per indicare il taglio
    plt.axhline(y=float(threshold), color='r', linestyle='--')

    save_plot(plt, f"dendrogram_{n_clusters}_clusters.png", plot_dir)

    return len(set(dendrogram_dict['color_list']))

def save_silhouette_plot(X: np.ndarray, labels: np.ndarray, k: int, plot_dir: str):
    """
    Crea e salva il plot della silhouette.

    Args:
        X (np.ndarray): Dati di input.
        labels (np.ndarray): Etichette dei cluster.
        k (int): Numero di cluster.
        plot_dir (str): Directory di output per i plot.
    """
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, labels)
    y_ticks = []
    y_lower, y_upper = 0, 0

    plt.figure(figsize=(10, 7))
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
    plt.yticks(y_ticks, np.unique(labels))
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.title(f"Silhouette Plot (k={k})")
    save_plot(plt, f"silhouette_plot_k{k}.png", plot_dir)


def save_elbow_plot(X: np.ndarray, max_clusters: int, clustering_func: Callable, plot_dir: str):
    """
    Crea e salva il grafico del gomito.

    Args:
        X (np.ndarray): Dati di input.
        max_clusters (int): Numero massimo di cluster da provare.
        clustering_func (Callable): Funzione che esegue il clustering e restituisce le etichette.
        plot_dir (str): Directory di output per i plot.
    """
    from sklearn.metrics import silhouette_score

    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        labels = clustering_func(n_clusters)
        silhouette_scores.append(silhouette_score(X, labels))

    plt.figure(figsize=(10, 7))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel("Numero di cluster")
    plt.ylabel("Silhouette Score")
    plt.title("Metodo del gomito usando Silhouette Score")
    save_plot(plt, "elbow_plot.png", plot_dir)