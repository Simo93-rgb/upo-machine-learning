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
        """
        Inizializza il clustering gerarchico con una matrice delle distanze e il tipo di linkage desiderato.

        :param : Dataset.
        :param linkage: Metodo di linkage ('single', 'complete', o 'average').
        """
        self.distance_matrix = None
        self._compute_initial_distance_matrix(X)
        self.linkage = linkage
        self.clusters = {str(i): Cluster([str(i)], str(i)) for i in range(X.shape[0])}
        self.cluster_history = []
        self.dataset_dim = X.shape[0]

    def _compute_initial_distance_matrix(self, X):
        """
        Calcola la matrice delle distanze iniziale tra tutti i punti e
        la salva in self.distance_matrix: Dict[str, Dict[str, float]]
        """
        # Step 1: Calcola la matrice delle distanze utilizzando ad esempio una distanza euclidea
        distance_matrix_df = pd.DataFrame(
            pairwise_distances(X, metric='euclidean'),
            index=[str(i) for i in range(len(X))],
            columns=[str(i) for i in range(len(X))]
        )

        # Step 2: Converti la matrice delle distanze in un dizionario nidificato
        distance_matrix_dict = distance_matrix_df.to_dict()
        self.distance_matrix = distance_matrix_dict

    def fit(self) -> None:
        """
        Esegue il clustering gerarchico, registrando ogni unione di cluster.
        """
        i = 0
        start = time.time()
        while len(self.clusters) > 1:
            if i % 2 == 0:
                print(f'Analizzate {i} istanze in {time.time() - start:.4f}')
            # Trova i cluster più vicini
            closest_pair, min_dist = self._find_closest_clusters()

            # Unisci i cluster
            self._merge_clusters(*closest_pair, distance=min_dist)
            i += 1

    def _find_closest_clusters(self) -> Tuple[Tuple[str, str], float]:
        """
        Trova i due cluster più vicini nella matrice delle distanze.

        :return: Una tupla con la coppia di nomi dei cluster più vicini e la loro distanza minima.
        """
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
        """
        Unisce due cluster, aggiorna la matrice delle distanze e registra la storia delle unioni.

        :param cluster1: Nome del primo cluster.
        :param cluster2: Nome del secondo cluster.
        :param distance: Distanza tra i due cluster uniti.
        """
        indices = [self.clusters[cluster1].name, self.clusters[cluster2].name]
        new_cluster = Cluster(indices, name=f"{cluster1}_{cluster2}")
        self.clusters[new_cluster.name] = new_cluster
        self.cluster_history.append((cluster1, cluster2, distance))

        # Inizializza la chiave per il nuovo cluster in self.distance_matrix
        self.distance_matrix[new_cluster.name] = {}

        # Aggiorna la matrice delle distanze per il nuovo cluster
        for cluster in list(self.clusters.keys()):
            # if cluster != new_cluster:
            new_distance = self._linkage_distance(self.clusters[new_cluster.name], self.clusters[cluster])
            self.distance_matrix[new_cluster.name][cluster] = new_distance
            self.distance_matrix[cluster][new_cluster.name] = new_distance

        # Elimina i cluster vecchi dalla matrice e dal dizionario dei cluster
        del self.clusters[cluster1]
        del self.clusters[cluster2]
        del self.distance_matrix[cluster1]
        del self.distance_matrix[cluster2]
        for cluster in self.distance_matrix.values():
            cluster.pop(cluster1, None)
            cluster.pop(cluster2, None)

    def _linkage_distance(self, cluster1: Cluster, cluster2: Cluster) -> float:
        """
        Calcola la distanza tra due cluster in base al metodo di linkage.

        :param cluster1: Primo cluster.
        :param cluster2: Secondo cluster.
        :return: La distanza calcolata tra i due cluster.
        """
        # distances = [self.distance_matrix[str(i)][str(j)]
        #              for i in cluster1.indices for j in cluster2.indices]
        # distances = [self.distance_matrix[cluster1.name][cluster2.name]]
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
        """
        Restituisce la storia delle unioni di cluster.

        :return: Una lista di tuple con i cluster uniti e la distanza di unione.
        """
        return self.cluster_history

    def predict(self, num_clusters: int) -> Dict[str, Cluster]:
        """
        Assegna i punti ai cluster finali.

        :param num_clusters: Numero desiderato di cluster finali.
        :return: Lista di oggetti Cluster a cui ogni punto appartiene.
        """
        # Continua a unire cluster fino a ottenere il numero desiderato
        while len(self.clusters) > num_clusters:
            closest_pair, min_dist = self._find_closest_clusters()
            self._merge_clusters(*closest_pair, distance=min_dist)

        # Ora ogni punto è assegnato al suo cluster finale
        final_clusters = list(self.clusters.values())

        # Mappa i dati ai loro cluster finali
        labels = {}
        for cluster in final_clusters:
            for i in cluster.indices:
                labels[i] = cluster  # Assegna l'oggetto Cluster al dato corrispondente

        return labels
