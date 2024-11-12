import numpy as np
import pandas as pd


class HierarchicalClustering:
    def __init__(self, distance_metric, linkage='single'):
        self.distance_metric = distance_metric # Funzione di calcolo della distanza
        self.distance_matrix = None
        self.clusters = None
        self.linkage_matrix = None
        self.linkage = linkage

    def fit(self, X):
        """
        Esegue il clustering gerarchico agglomerativo sui dati in input X.
        Parametri:
        - X (np.ndarray): matrice di input con le caratteristiche (features) degli oggetti.
        """
        print("Eseguendo il clustering gerarchico agglomerativo sui dati in input.")

        # Step 1: crea la matrice delle distanze iniziale
        self._compute_initial_distance_matrix(X)
        print("Matrice delle distanze iniziale creata.")

        # Step 2: inizializza i cluster
        self._initialize_clusters(X.shape[0])

        # Step 3: esegui il clustering gerarchico
        self._perform_hierarchical_clustering()

        print("Clustering gerarchico completato.")

    def _compute_initial_distance_matrix(self, X):
        """
        Calcola la matrice delle distanze iniziale tra tutti i punti.
        """
        n = X.shape[0]
        self.distance_matrix = pd.DataFrame(index=range(n), columns=range(n))

        for i in range(n):
            for j in range(i + 1, n):
                self.distance_matrix.iloc[i, j] = self.distance_metric(X[i], X[j])
                self.distance_matrix.iloc[j, i] = self.distance_matrix.iloc[i, j]

    def _initialize_clusters(self, n):
        """
        Inizializza i cluster, dove ogni oggetto è un cluster a sé stante.
        """
        self.clusters = [Cluster([i]) for i in range(n)]

    # def _perform_hierarchical_clustering(self):
    #     """
    #     Esegue il clustering gerarchico agglomerativo.
    #     """
    #     n = len(self.clusters)
    #     self.linkage_matrix = []
    #
    #     while n > 1:
    #         print(f"Numero di cluster rimasti: {n}")
    #
    #         # Trova i due cluster più vicini e fai il merge
    #         closest_i, closest_j, min_dist = self._find_closest_clusters()
    #         print(f"Cluster più vicini trovati: {closest_i}, {closest_j} (distanza: {min_dist:.2f})")
    #
    #         # Crea il nuovo cluster mergato
    #         new_cluster = Cluster(self.clusters[closest_i].indices + self.clusters[closest_j].indices)
    #
    #         # Salva il merge per la linkage matrix
    #         self.linkage_matrix.append(
    #             [self.clusters[closest_i][0], self.clusters[closest_j][0], min_dist, len(new_cluster)])
    #
    #         # Aggiorna la matrice delle distanze
    #         self._update_distance_matrix(closest_i, closest_j, new_cluster)
    #
    #         # Rimuovi i cluster mergiati e aggiungi il nuovo cluster
    #         self.clusters.pop(max(closest_i, closest_j))
    #         self.clusters.pop(min(closest_i, closest_j))
    #         self.clusters.append(new_cluster)
    #
    #         n = len(self.clusters)
    #
    #     self.linkage_matrix = np.array(self.linkage_matrix)
    def _perform_hierarchical_clustering(self):
        n = len(self.clusters)
        self.linkage_matrix = []
        cluster_to_index = {i: i for i in range(n)}  # Mappa di cluster e loro indici nella matrice

        while n > 1:
            print(f"Numero di cluster rimasti: {n}")

            # Trova i due cluster più vicini e fai il merge
            closest_i, closest_j, min_dist = self._find_closest_clusters(cluster_to_index)
            print(f"Cluster più vicini trovati: {closest_i}, {closest_j} (distanza: {min_dist:.2f})")

            # Crea il nuovo cluster
            new_cluster = Cluster(self.clusters[closest_i].indices + self.clusters[closest_j].indices)

            # Salva il merge nella linkage matrix
            self.linkage_matrix.append(
                [closest_i, closest_j, min_dist, len(new_cluster)])

            # Aggiorna la matrice delle distanze
            self._update_distance_matrix(closest_i, closest_j, new_cluster, cluster_to_index)

            # Aggiorna la mappa cluster_to_index
            new_index = max(cluster_to_index.values()) + 1
            cluster_to_index[new_index] = new_index
            cluster_to_index.pop(closest_i)
            cluster_to_index.pop(closest_j)

            # Aggiorna i cluster
            self.clusters.pop(max(closest_i, closest_j))
            self.clusters.pop(min(closest_i, closest_j))
            self.clusters.append(new_cluster)

            n = len(self.clusters)

        self.linkage_matrix = np.array(self.linkage_matrix)

    def _update_distance_matrix(self, idx1, idx2, new_cluster, cluster_to_index):
        """
        Aggiorna la matrice delle distanze dopo la fusione di due cluster.
        """
        n = len(self.clusters)
        new_index = max(cluster_to_index.values()) + 1

        # Espandi la matrice delle distanze se necessario
        if new_index >= self.distance_matrix.shape[0]:
            rows = cols = new_index + 1
            self.distance_matrix = self.distance_matrix.reindex(index=range(rows), columns=range(cols), fill_value=0.0)

        # Aggiorna le distanze per il nuovo cluster
        for i in range(n):
            dist = self._linkage_distance(self.clusters[i], new_cluster)
            self.distance_matrix.iloc[i, new_index] = dist
            self.distance_matrix.iloc[new_index, i] = dist

        # Rimuovi le righe e le colonne dei cluster mergiati
        self.distance_matrix = self.distance_matrix.drop([idx1, idx2], axis=0)
        self.distance_matrix = self.distance_matrix.drop([idx1, idx2], axis=1)

    def _find_closest_clusters(self, cluster_to_index):
        min_dist = float('inf')
        closest_i = None
        closest_j = None

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                dist = self._linkage_distance(self.clusters[i], self.clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_i = i
                    closest_j = j

        return cluster_to_index[closest_i], cluster_to_index[closest_j], min_dist

    # def _update_distance_matrix(self, idx1, idx2, new_cluster):
    #     """
    #     Aggiorna la matrice delle distanze dopo un merge di due cluster.
    #     """
    #     n = len(self.clusters)
    #
    #     # Aggiungi una nuova riga e colonna per il nuovo cluster
    #     self.distance_matrix[n] = 0
    #     self.distance_matrix.loc[n] = 0
    #
    #     # Calcola le nuove distanze per il nuovo cluster
    #     for i in range(n):
    #         self.distance_matrix.iloc[i, n] = self._linkage_distance(self.clusters[i], new_cluster)
    #         self.distance_matrix.iloc[n, i] = self.distance_matrix.iloc[i, n]
    #
    #     # Rimuovi le righe e le colonne corrispondenti ai cluster mergiati
    #     self.distance_matrix = self.distance_matrix.drop([idx1, idx2], axis=0)
    #     self.distance_matrix = self.distance_matrix.drop([idx1, idx2], axis=1)

    def _linkage_distance(self, cluster1, cluster2):
        """
        Calcola la distanza di linkage tra due cluster.
        """
        if self.linkage == "single":
            return np.min([self.distance_matrix.iloc[i, j] for i in cluster1.indices for j in cluster2.indices])
        elif self.linkage == "complete":
            return np.max([self.distance_matrix.iloc[i, j] for i in cluster1.indices for j in cluster2.indices])
        elif self.linkage == "average":
            return np.mean([self.distance_matrix.iloc[i, j] for i in cluster1.indices for j in cluster2.indices])
        elif self.linkage == "centroid":
            centroid1 = np.mean([self.distance_matrix.iloc[i] for i in cluster1.indices], axis=0)
            centroid2 = np.mean([self.distance_matrix.iloc[j] for j in cluster2.indices], axis=0)
            return np.linalg.norm(centroid1 - centroid2)
        elif self.linkage == "ward":
            # Ward's method: incremento della varianza
            all_points = [self.distance_matrix.iloc[i] for i in cluster1.indices + cluster2.indices]
            centroid = np.mean(all_points, axis=0)
            return np.sum([np.linalg.norm(self.distance_matrix.iloc[i] - centroid) ** 2 for i in
                           cluster1.indices + cluster2.indices])
        else:
            raise ValueError("Metodo di linkage non riconosciuto")

    def predict(self, n_clusters=1):
        """
        Divide i dati in n_clusters cluster.

        Parametri:
        - n_clusters (int): il numero di cluster desiderato.

        Restituisce:
        - np.ndarray: le assegnazioni dei dati ai cluster.
        """
        print(f"Divido i dati in {n_clusters} cluster.")
        # Crea una lista di cluster iniziali
        clusters = [[i] for i in range(len(self.linkage_matrix) + 1)]

        # Unisci i cluster fino a raggiungere il numero desiderato
        while len(clusters) > n_clusters:
            min_dist = float('inf')
            merge_indices = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._linkage_distance(clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_indices = (i, j)

            # Esegui il merge dei due cluster più vicini
            new_cluster = clusters[merge_indices[0]] + clusters[merge_indices[1]]
            clusters.pop(max(merge_indices))
            clusters.pop(min(merge_indices))
            clusters.append(new_cluster)

        # Assegna i dati ai cluster finali
        labels = np.zeros(len(self.linkage_matrix) + 1, dtype=int)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = i

        return labels

class Cluster:
    def __init__(self, indices):
        self.indices = indices

    def __repr__(self):
        return f"Cluster({self.indices})"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.indices[index]



