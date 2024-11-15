import os
import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Union

from data import DataHandler
from hierarchical_clustering import HierarchicalClustering
from evaluation import *
from plot import save_dendrogram, save_silhouette_plot, save_elbow_plot


def setup_directories() -> Tuple[str, str, str]:
    """
    Configura e crea le directory necessarie per il progetto.

    Returns:
        Tuple[str, str, str]: Percorsi per dataset, risultati e plot.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'Assets', 'Dataset')
    output_dir = os.path.join(current_dir, 'Assets', 'Results')
    plot_dir = os.path.join(current_dir, 'Assets', 'Plot')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    return dataset_dir, output_dir, plot_dir


def load_and_preprocess_data(dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Carica e pre-processa il dataset.

    Args:
        dataset_path (str): Percorso del file del dataset.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Features e labels del dataset.
    """
    data_handler = DataHandler(dataset_path)
    data_handler.preprocess_data()
    X = data_handler.get_features()
    y = data_handler.get_labels().iloc[:, -1].values
    return X, y


def kmeans_pre_clustering(X: pd.DataFrame, n_clusters: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Esegue un pre-clustering utilizzando K-Means.

    Args:
        X (pd.DataFrame): Dati di input.
        n_clusters (int): Numero di cluster.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Centroidi e etichette dei cluster.
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)
    return kmeans.cluster_centers_, labels


def create_linkage_matrix(hc: HierarchicalClustering) -> np.ndarray:
    """
    Crea la matrice di linkage per il dendrogramma.

    Args:
        hc (HierarchicalClustering): Oggetto di clustering gerarchico.

    Returns:
        np.ndarray: Matrice di linkage.
    """
    linkage_matrix = []
    name_to_idx = {}
    current_idx = 0

    for a, b, dist in hc.get_cluster_history():
        if a not in name_to_idx:
            name_to_idx[a.name] = float(a.name)
            current_idx += 1
        if b not in name_to_idx:
            name_to_idx[b.name] = float(b.name)
            current_idx += 1

        linkage_matrix.append([
            name_to_idx[a.name],
            name_to_idx[b.name],
            dist,
            len(a.dataset_indices) + len(b.dataset_indices)
        ])

    return np.array(linkage_matrix)


def run_clustering(
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        linkage: str,
        distance: str,
        output_dir: str,
        plot_dir: str,
        max_clusters:int = 8,
        k_means_reduction:int=10) -> None:
    """
    Esegue il clustering gerarchico e salva i risultati.

    Args:
        X (pd.DataFrame): Dati di input.
        y (np.ndarray): Etichette vere.
        linkage (str): Metodo di linkage.
        distance (str): Metrica di distanza.
        output_dir (str): Directory per i risultati.
        plot_dir (str): Directory per i plot.
        max_clusters (int)
        k_means_reduction (int)
    """
    # Configurazione delle sottocartelle per i risultati
    sub_dir = f"{linkage}_{distance}"
    sub_output_dir = os.path.join(output_dir, sub_dir)
    sub_plot_dir = os.path.join(plot_dir, sub_dir)
    os.makedirs(sub_output_dir, exist_ok=True)
    os.makedirs(sub_plot_dir, exist_ok=True)

    # Inizializzazione e fit del modello
    hc = HierarchicalClustering(linkage=linkage, X=X, distance_metric=distance, pre_clustering_func=kmeans_pre_clustering,
                                n_clusters=k_means_reduction)
    print(f'Inizio fit per {linkage} linkage e distanza {distance}')
    hc.fit()
    print('Fine fit')
    # Creazione del dendrogramma
    linkage_matrix = create_linkage_matrix(hc)
    save_dendrogram(linkage_matrix, sub_plot_dir)

    # Trova il numero ottimale di cluster

    optimal_k = find_optimal_clusters(X, max_clusters, hc.predict, sub_plot_dir)
    print(f"Numero ottimale di cluster: {optimal_k}")
    # Creazione del grafico del gomito
    save_elbow_plot(X, max_clusters, hc.predict, sub_plot_dir)
    # Previsione e valutazione
    labels = hc.predict(optimal_k)

    # Calcola e salva le metriche di valutazione
    # Valuta il clustering
    evaluation_results = evaluate_clustering(y_true=y, y_pred=labels, X=X, model_name="Hierarchical Clustering")
    save_evaluation_results(evaluation_results, "evaluation_results.csv", sub_output_dir)

    # Salvataggio del plot della silhouette
    save_silhouette_plot(X, labels, optimal_k, sub_plot_dir)

    print(f"Risultati per {linkage} linkage e distanza {distance} salvati.")