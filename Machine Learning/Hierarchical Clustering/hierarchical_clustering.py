from typing import Dict, Tuple, List, Union, Callable, Optional, Any

from numpy.core.numeric import indices
from sklearn.metrics import pairwise_distances
import pandas as pd
import time
import numpy as np


class Cluster:
    def __init__(self, indices: List[str], name: str, previous_index: Tuple[int, int], dataset_indices: List[int]):
        """
        Costruttore dell'oggetto Cluster
        """
        self.indices = indices  # Una lista di stringhe che rappresenta gli indici originali da cui ha avuto origine il
        # cluster. Questi indici sono tipicamente utilizzati per identificare gli elementi all'interno di un dataset.
        self.name = name
        self.dataset_indices: List[int] = dataset_indices  # Indici del dataset originale
        self.previous_index: Tuple[int, int] = previous_index

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Cluster) and self.name == other.name

    def _append_index(self, i: int):
        self.dataset_indices.append(i)

    def set_dataset_index(self, _list: List[int]):
        self.dataset_indices = _list


class HierarchicalClustering:
    def __init__(self, X, linkage: str = 'single', distance_metric='euclidean', pre_clustering_func: Optional[Callable] = None,
                 **pre_clustering_kwargs):
        self.X = X
        self.linkage = linkage
        self.pre_clustering_func = pre_clustering_func
        self.pre_clustering_kwargs = pre_clustering_kwargs
        self.distance_matrix = None
        self.clusters = None
        self.cluster_history = []
        self.dataset_dim = X.shape[0]
        self.cluster_id = self.dataset_dim - 1
        self.distance_metric = distance_metric

    def _apply_pre_clustering(self):
        try:
            # Esegui la funzione di pre-clustering (ad esempio K-means)
            result = self.pre_clustering_func(self.X, **self.pre_clustering_kwargs)

            # Se il risultato è una tupla, ottieni sia il nuovo dataset ridotto che le etichette
            if isinstance(result, tuple):
                self.X, labels = result  # labels contiene le etichette dei cluster K-means

                # Inizializza i cluster basati sulle etichette di K-means
                self.clusters = {}
                for i, label in enumerate(labels):
                    if str(label) not in self.clusters:
                        # Crea un nuovo cluster per ogni nuova etichetta
                        self.clusters[str(label)] = (
                            Cluster(
                                indices=[str(label)],
                                name=str(label),
                                previous_index=(label, label),
                                dataset_indices=[i]
                            )
                        )
                    else:
                        # Aggiungi l'indice corrente al cluster esistente
                        # self.clusters[str(label)].indices.append(str(i))
                        self.clusters[str(label)].dataset_indices.append(i)


            # Se il risultato non è una tuple, significa che non ci sono etichette (solo dataset ridotto)
            else:
                self.X = result  # Assegna il dataset ridotto a self.X

                # Inizializza i cluster senza etichette (ogni punto è un proprio cluster)
                self.clusters = {str(i): Cluster([str(i)], str(i), i) for i in range(self.dataset_dim)}

            # Aggiorna la dimensione del dataset
            self.dataset_dim = self.X.shape[0]
            self.cluster_id = self.dataset_dim - 1

            print(f"Pre-clustering applicato. Nuova dimensione del dataset: {self.dataset_dim}")

        except Exception as e:
            print(f"Errore durante il pre-clustering: {e}")
            print("Procedendo con il dataset originale.")

    def _compute_initial_distance_matrix(self):
        distance_matrix_df = pd.DataFrame(
            pairwise_distances(self.X, metric=self.distance_metric),
            index=[str(i) for i in range(self.dataset_dim)],
            columns=[str(i) for i in range(self.dataset_dim)]
        )
        self.distance_matrix = distance_matrix_df.to_dict()

    def fit(self) -> None:
        if self.pre_clustering_func:
            self._apply_pre_clustering()

        # Calcola la matrice delle distanze iniziale (dopo il pre-clustering)
        self._compute_initial_distance_matrix()

        # Inizializza i cluster (se non sono già stati inizializzati dal pre-clustering)
        if not self.clusters:
            self.clusters = {str(i): Cluster(
                indices=[str(i)],
                name=str(i),
                dataset_indices=[i],
                previous_index=(i, i)
            ) for i in range(self.dataset_dim)}

        while len(self.clusters) > 1:
            # Trova la coppia di cluster più vicini e li unisce
            closest_pair, min_dist = self._find_closest_clusters()
            self._merge_clusters(*closest_pair, distance=min_dist)

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
        cluster1_obj = self.clusters[cluster1]
        cluster2_obj = self.clusters[cluster2]
        new_cluster = self._merge_cluster_objects(cluster1_obj, cluster2_obj)
        self.clusters[new_cluster.name] = new_cluster

        # Salvo gli oggetti cluster1 e cluster2
        self.cluster_history.append((self.clusters[cluster1], self.clusters[cluster2], distance))

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

    def _merge_cluster_objects(self, cluster1: Cluster, cluster2: Cluster) -> Cluster:
        """
        Unisce due oggetti Cluster in un nuovo Cluster.
        """
        # Unisci gli indici e i dataset_indices
        merged_indices = list(set(cluster1.indices + cluster2.indices))
        merged_local_indices = (int(cluster1.name), int(cluster2.name))

        # Crea un nuovo nome per il cluster unito
        self.cluster_id += 1
        new_cluster_name = str(self.cluster_id)
        dataset_indices = [cluster1.dataset_indices, cluster2.dataset_indices]
        flat_dataset_indices = [item for sublist in dataset_indices for item in sublist]
        # Crea un nuovo oggetto Cluster con gli indici uniti
        new_cluster = Cluster(
            indices=merged_indices,
            name=new_cluster_name,
            dataset_indices=flat_dataset_indices,
            previous_index=merged_local_indices
        )
        return new_cluster

    def _linkage_distance(self, cluster1: Cluster, cluster2: Cluster) -> float:
        distances = [self.distance_matrix[str(i)][cluster2.name] for i in cluster1.previous_index]

        if self.linkage == 'single':
            return min(distances)

        elif self.linkage == 'complete':
            return max(distances)

        elif self.linkage == 'average':
            return sum(distances) / len(distances)
        elif self.linkage == 'centroid':
            # Calcola il centroide di ciascun cluster
            centroid1 = np.mean([self.X[int(i)] for i in cluster1.indices], axis=0)
            centroid2 = np.mean([self.X[int(i)] for i in cluster2.indices], axis=0)

            # Calcola la distanza euclidea tra i due centroidi
            return np.linalg.norm(centroid1 - centroid2)

        else:
            raise ValueError("Metodo di linkage non supportato. Usa 'single', 'complete' o 'average'.")

    def get_cluster_history(self) -> List[Tuple[Cluster, Cluster, float]]:
        return self.cluster_history

    def _predict(self, num_clusters: int) -> np.ndarray:
        """
        Prevede i cluster finali utilizzando `self.cluster_history`.
        """

        clusters_finali = {str(i): Cluster([str(i)], str(i), i) for i in range(self.dataset_dim)}
        # clusters_finali = {str(i): [i] for i in range(self.dataset_dim)}

        # Applica le fusioni dalla storia fino a ottenere il numero desiderato di cluster
        for a, b, dist in self.cluster_history:
            if len(clusters_finali) <= num_clusters:
                break

            # merged_cluster_indices = clusters_finali.pop(a.name) + clusters_finali.pop(b.name)
            merged_clusters = self._merge_cluster_objects(a, b)
            clusters_finali[merged_clusters.name] = merged_clusters
            del clusters_finali[a.name]
            del clusters_finali[b.name]

        final_cluster = list(self.clusters.keys())[0]
        labels_finali = np.zeros(len(self.clusters[final_cluster].dataset_indices), dtype=int)

        for label_idx, (nome_cluster_finale, indici_lista) in enumerate(clusters_finali.items()):
            for indice in indici_lista.dataset_indices:
                labels_finali[indice] = label_idx

        return labels_finali

    def predict(self, num_clusters: int) -> np.ndarray:
        """
        Prevede i cluster finali utilizzando `self.cluster_history`.
        Ricostruisce i cluster andando a ritroso nella storia delle fusioni.
        """
        clusters: List[Cluster] = []
        # Applica le fusioni dalla storia fino a ottenere il numero desiderato di cluster
        for i, x in enumerate(self.cluster_history[-(num_clusters - 1):][::-1]):
            clusters.append(x[0])
            if i == num_clusters - 2:
                clusters.append(x[1])

        # Inizializza l'array delle etichette
        final_cluster = list(self.clusters.keys())[0]
        labels = np.zeros(len(self.clusters[final_cluster].dataset_indices), dtype=int)
        # labels = np.zeros(self.dataset_dim, dtype=int)

        # Assegna le etichette ai punti nei cluster finali
        for cluster in clusters:
            for index in cluster.dataset_indices:
                labels[index] = cluster.name

        return labels
