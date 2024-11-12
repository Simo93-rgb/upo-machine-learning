import cupy as cp
from sklearn.metrics import pairwise_distances
from typing import Dict, Tuple, List, Union
from sklearn.metrics import pairwise_distances
import pandas as pd
import time
from joblib import Parallel, delayed

class Cluster:
    def __init__(self, indices: List[str], name: str):
        self.indices = indices
        self.name = name

    def __hash__(self):
        return hash(self.name)  # Usa l'id per l'hash, necessario per l'uso come chiave del dizionario

    def __eq__(self, other):
        return isinstance(other, Cluster) and self.name == other.name



class HierarchicalClustering:
    def __init__(self, X, linkage: str = 'single'):
        # Inizializza la matrice delle distanze come un dizionario di dizionari
        initial_distances = pairwise_distances(X, metric='euclidean')
        self.distance_matrix = {str(i): {str(j): cp.array(initial_distances[i, j])
                                         for j in range(X.shape[0]) if i != j}
                                for i in range(X.shape[0])}
        self.linkage = linkage
        self.clusters = {str(i): Cluster([str(i)], name=str(i)) for i in range(X.shape[0])}
        self.cluster_history = []
        self.dataset_dim = X.shape[0]

    def fit(self) -> None:
        i = 0
        start = time.time()
        while len(self.clusters) > 1:
            if i % 2 == 0:
                print(f'Analizzate {i} istanze in {time.time() - start:.4f}')

            # Trova i cluster più vicini
            closest_pair, min_dist = self._find_closest_clusters()

            # Unisci i cluster trovati
            self._merge_clusters(*closest_pair, distance=min_dist)
            i += 1

    def _find_closest_clusters(self) -> Tuple[Tuple[str, str], float]:
        """
        Trova i due cluster più vicini nella matrice delle distanze usando solo cluster attivi.
        """
        min_dist = cp.inf
        closest_pair = (None, None)

        active_keys = list(self.clusters.keys())
        for i, cluster_i in enumerate(active_keys):
            for cluster_j in active_keys[i + 1:]:
                dist = self.distance_matrix[cluster_i][cluster_j]
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (cluster_i, cluster_j)

        # Controlla che sia stata trovata una coppia valida
        if closest_pair == (None, None):
            raise ValueError("Nessun cluster trovato per l'unione.")

        return closest_pair, min_dist.item()

    def _merge_clusters(self, cluster1_idx: str, cluster2_idx: str, distance: float) -> None:
        """
        Unisce due cluster e aggiorna la matrice delle distanze mantenendo il dizionario dei cluster.
        """
        # Crea un nuovo cluster unendo gli indici dei cluster originali
        new_cluster_name = f"{cluster1_idx}_{cluster2_idx}"
        new_cluster = Cluster(self.clusters[cluster1_idx].indices + self.clusters[cluster2_idx].indices,
                              name=new_cluster_name)
        self.clusters[new_cluster_name] = new_cluster
        self.cluster_history.append((cluster1_idx, cluster2_idx, distance))

        # Aggiungi la nuova chiave alla matrice delle distanze
        self.distance_matrix[new_cluster_name] = {}

        # Calcola e aggiorna le distanze per il nuovo cluster
        for other in list(self.clusters.keys()):
            if other in (cluster1_idx, cluster2_idx, new_cluster_name):
                continue

            if self.linkage == 'single':
                new_dist = cp.minimum(self.distance_matrix[cluster1_idx][other],
                                      self.distance_matrix[cluster2_idx][other])
            elif self.linkage == 'complete':
                new_dist = cp.maximum(self.distance_matrix[cluster1_idx][other],
                                      self.distance_matrix[cluster2_idx][other])
            elif self.linkage == 'average':
                new_dist = (self.distance_matrix[cluster1_idx][other] +
                            self.distance_matrix[cluster2_idx][other]) / 2
            else:
                raise ValueError("Unsupported linkage method.")

            # Aggiorna la matrice delle distanze sia per il nuovo cluster sia per gli altri cluster
            self.distance_matrix[new_cluster_name][other] = new_dist
            self.distance_matrix[other][new_cluster_name] = new_dist

        # Rimuovi i cluster vecchi dalla matrice delle distanze
        del self.clusters[cluster1_idx]
        del self.clusters[cluster2_idx]
        del self.distance_matrix[cluster1_idx]
        del self.distance_matrix[cluster2_idx]
        for cluster in self.distance_matrix.values():
            cluster.pop(cluster1_idx, None)
            cluster.pop(cluster2_idx, None)
