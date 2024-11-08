import os
import numpy as np


class HierarchicalClustering:
    def __init__(self, linkage="single"):
        """
        Inizializza l'algoritmo di clustering gerarchico.

        Parametri:
        - linkage (str): la strategia di linkage da utilizzare. Può essere:
          - "single": distanza minima (single link)
          - "complete": distanza massima (complete link)
          - "average": distanza media (average link)
          - "centroid": distanza tra i centroidi
          - "ward": metodo di Ward
        """
        print(f"Inizializzando l'algoritmo di clustering gerarchico con linkage '{linkage}'.")
        self.linkage = linkage
        self.linkage_matrix = []

    def fit(self, X):
        """
        Esegue il clustering gerarchico agglomerativo sui dati in input X.

        Parametri:
        - X (np.ndarray): matrice di input con le caratteristiche (features) degli oggetti.
        """
        print("Eseguendo il clustering gerarchico agglomerativo sui dati in input.")

        # Step 1: crea la matrice delle distanze iniziale
        distance_matrix = self._compute_initial_distances(X)
        print("Matrice delle distanze iniziale creata.")
        # Step 2: ogni oggetto è un cluster
        clusters = [[i] for i in range(X.shape[0])]

        # Lista per tracciare i merge per il dendrogramma
        linkage_matrix = []

        while len(clusters) > 1:
            print(f"Numero di cluster rimasti: {len(clusters)}")
            # Step 3.1: trova i due cluster più vicini e fai il merge
            i, j, min_dist = self._find_closest_clusters(distance_matrix, clusters)
            print(f"Cluster più vicini trovati: {i}, {j} (distanza: {min_dist:.2f})")

            new_cluster = clusters[i] + clusters[j]

            # Salva il merge per la linkage matrix
            linkage_matrix.append([clusters[i][0], clusters[j][0], min_dist, len(new_cluster)])

            # Rimuovi i cluster mergiati e aggiungi il nuovo cluster
            clusters.pop(max(i, j))
            clusters.pop(min(i, j))
            clusters.append(new_cluster)

            # Step 3.2: aggiorna la matrice delle distanze
            distance_matrix = self._update_distances(distance_matrix, clusters, i, j)
            print("Matrice delle distanze aggiornata.")

            # Verifica la dimensione della matrice per evitare errori
            print(f"Dimensione matrice delle distanze: {distance_matrix.shape}")

        # Salva la linkage matrix per il dendrogramma
        self.linkage_matrix = np.array(linkage_matrix)
        print("Clustering gerarchico completato.")

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
                    dist = self._linkage_distance(self.linkage_matrix, clusters[i], clusters[j])
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

    def _compute_initial_distances(self, X):
        """
        Calcola la matrice di distanze iniziale tra le features.

        Parametri:
        - X (np.ndarray): matrice delle caratteristiche degli oggetti.

        Ritorna:
        - np.ndarray: matrice delle distanze iniziale.
        """
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance_matrix[i, j] = self._euclidean_distance(X[i], X[j])
                distance_matrix[j, i] = distance_matrix[i, j]  # matrice simmetrica

        return distance_matrix

    def _euclidean_distance(self, x, y):
        """
        Calcola la distanza euclidea tra due vettori.

        Parametri:
        - x, y (np.ndarray): i due vettori tra cui calcolare la distanza.

        Ritorna:
        - float: la distanza euclidea.
        """
        return np.sqrt(np.sum((x - y) ** 2))

    def _find_closest_clusters(self, distance_matrix, clusters):
        """
        Trova i due cluster più vicini.

        Parametri:
        - distance_matrix (np.ndarray): matrice delle distanze corrente.
        - clusters (list): lista dei cluster attuali.

        Ritorna:
        - tuple: indici dei due cluster più vicini e la distanza tra di essi.
        """
        min_dist = float("inf")
        closest_pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                print(f"Cluster1: {clusters[i]}, Cluster2: {clusters[j]}")
                dist = self._linkage_distance(distance_matrix, clusters[i], clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (i, j)

        return closest_pair[0], closest_pair[1], min_dist

    def _linkage_distance(self, distance_matrix, cluster1, cluster2):
        """
        Calcola la distanza tra due cluster in base alla strategia di linkage selezionata.

        Parametri:
        - distance_matrix (np.ndarray): matrice delle distanze corrente.
        - cluster1, cluster2 (list): i due cluster da confrontare.

        Ritorna:
        - float: la distanza tra i due cluster.
        """
        if self.linkage == "single":
            return np.min([distance_matrix[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == "complete":
            return np.max([distance_matrix[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == "average":
            return np.mean([distance_matrix[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == "centroid":
            centroid1 = np.mean([distance_matrix[i] for i in cluster1], axis=0)
            centroid2 = np.mean([distance_matrix[j] for j in cluster2], axis=0)
            return np.linalg.norm(centroid1 - centroid2)
        elif self.linkage == "ward":
            # Ward's method: incremento della varianza
            all_points = [distance_matrix[i] for i in cluster1 + cluster2]
            centroid = np.mean(all_points, axis=0)
            return np.sum([np.linalg.norm(distance_matrix[i] - centroid) ** 2 for i in cluster1 + cluster2])
        else:
            raise ValueError("Metodo di linkage non riconosciuto")

    def _update_distances(self, distance_matrix, clusters, idx1, idx2):
        """
        Aggiorna la matrice delle distanze dopo un merge.

        Parametri:
        - distance_matrix (np.ndarray): matrice delle distanze corrente.
        - clusters (list): lista aggiornata dei cluster.
        - idx1, idx2 (int): indici dei cluster mergiati.

        Ritorna:
        - np.ndarray: la nuova matrice delle distanze.
        """
        n = len(clusters)
        new_distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                new_distance_matrix[i, j] = self._linkage_distance(distance_matrix, clusters[i], clusters[j])
                new_distance_matrix[j, i] = new_distance_matrix[i, j]

        return new_distance_matrix

    def get_linkage_matrix(self):
        """
        Ritorna la linkage matrix per il plot del dendrogramma.

        Ritorna:
        - np.ndarray: la linkage matrix.
        """
        return self.linkage_matrix
