from typing import Dict, Tuple, List, Union, Callable, Optional, Any
from sklearn.metrics import pairwise_distances
import pandas as pd
import time
import numpy as np


class Cluster:
    def __init__(self, indices: List[str], name: str):
        self.indices = indices
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Cluster) and self.name == other.name


class HierarchicalClustering:
    def __init__(self, X, linkage: str = 'single', pre_clustering_func: Optional[Callable] = None,
                 **pre_clustering_kwargs):
        self.X = X
        self.linkage = linkage
        self.pre_clustering_func = pre_clustering_func
        self.pre_clustering_kwargs = pre_clustering_kwargs
        self.distance_matrix = None
        self.clusters = None
        self.cluster_history = []
        self.dataset_dim = X.shape[0]

    def _apply_pre_clustering(self):
        try:
            result = self.pre_clustering_func(self.X, **self.pre_clustering_kwargs)
            if isinstance(result, tuple):
                self.X, labels = result
            else:
                self.X = result
            self.dataset_dim = self.X.shape[0]
            print(f"Pre-clustering applicato. Nuova dimensione del dataset: {self.dataset_dim}")
        except Exception as e:
            print(f"Errore durante il pre-clustering: {e}")
            print("Procedendo con il dataset originale.")

    def _compute_initial_distance_matrix(self):
        distance_matrix_df = pd.DataFrame(
            pairwise_distances(self.X, metric='euclidean'),
            index=[str(i) for i in range(self.dataset_dim)],
            columns=[str(i) for i in range(self.dataset_dim)]
        )
        self.distance_matrix = distance_matrix_df.to_dict()

    def fit(self) -> None:
        if self.pre_clustering_func:
            self._apply_pre_clustering()

        self._compute_initial_distance_matrix()
        self.clusters = {str(i): Cluster([str(i)], str(i)) for i in range(self.dataset_dim)}

        i = 0
        start = time.time()
        while len(self.clusters) > 1:
            if i % 2 == 0:
                print(f'Analizzate {i} istanze in {time.time() - start:.4f}')
            closest_pair, min_dist = self._find_closest_clusters()
            self._merge_clusters(*closest_pair, distance=min_dist)
            i += 1

    def _find_closest_clusters(self) -> Tuple[Tuple[str, str], float]:
        min_dist = float('inf')
        closest_pair = (None, None)
        for i, cluster_i in enumerate(self.clusters):
            for j, cluster_j in enumerate(list(self.clusters)[i + 1:]):
                dist = self.distance_matrix[cluster_i][cluster_j]
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (cluster_i, cluster_j)
        return closest_pair, min_dist

    def _merge_clusters(self, cluster1: str, cluster2: str, distance: float) -> None:
        indices = [self.clusters[cluster1].name, self.clusters[cluster2].name]
        new_cluster = Cluster(indices, name=f"{cluster1}_{cluster2}")
        self.clusters[new_cluster.name] = new_cluster
        self.cluster_history.append((cluster1, cluster2, distance))

        self.distance_matrix[new_cluster.name] = {}
        for cluster in list(self.clusters.keys()):
            new_distance = self._linkage_distance(self.clusters[new_cluster.name], self.clusters[cluster])
            self.distance_matrix[new_cluster.name][cluster] = new_distance
            self.distance_matrix[cluster][new_cluster.name] = new_distance

        del self.clusters[cluster1]
        del self.clusters[cluster2]
        del self.distance_matrix[cluster1]
        del self.distance_matrix[cluster2]
        for cluster in self.distance_matrix.values():
            cluster.pop(cluster1, None)
            cluster.pop(cluster2, None)

    def _linkage_distance(self, cluster1: Cluster, cluster2: Cluster) -> float:
        distances = [self.distance_matrix[str(i)][cluster2.name] for i in cluster1.indices]
        if self.linkage == 'single':
            return min(distances)
        elif self.linkage == 'complete':
            return max(distances)
        elif self.linkage == 'average':
            return sum(distances) / len(distances)
        else:
            raise ValueError("Metodo di linkage non supportato. Usa 'single', 'complete' o 'average'.")

    def get_cluster_history(self) -> List[Tuple[str, str, float]]:
        return self.cluster_history

    def predict(self, num_clusters: int) -> np.ndarray:
        """
        Prevede i cluster finali utilizzando self.cluster_history.
        """
        # Inizializza ogni punto dati come un proprio cluster (all'inizio ci sono tanti cluster quanti sono i punti)
        clusters = {str(i): [i] for i in range(self.dataset_dim)}

        # Applica le fusioni dalla storia fino a ottenere il numero desiderato di cluster
        for a, b, dist in self.cluster_history:
            if len(clusters) <= num_clusters:
                break

            # Unisci i due cluster 'a' e 'b'
            merged_cluster = clusters.pop(a) + clusters.pop(b)
            clusters[a + '_' + b] = merged_cluster

        # Inizializza l'array delle etichette
        labels = np.zeros(self.dataset_dim, dtype=int)

        # Assegna le etichette ai punti nei cluster finali
        for label_idx, (cluster_name, indices_list) in enumerate(clusters.items()):
            for i in indices_list:
                labels[i] = label_idx

        return labels
