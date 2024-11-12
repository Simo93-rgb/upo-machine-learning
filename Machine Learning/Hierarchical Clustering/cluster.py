import numpy as np
class Cluster:
    def __init__(self, indices):
        self.indices = indices

    def __repr__(self):
        return f"Cluster({self.indices})"

    def __len__(self):
        return len(self.indices)


def _find_closest_clusters(self, distance_matrix, clusters):
    min_dist = float('inf')
    closest_i = None
    closest_j = None

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = self._linkage_distance(distance_matrix, clusters[i], clusters[j])
            if dist < min_dist:
                min_dist = dist
                closest_i = i
                closest_j = j

    return closest_i, closest_j, min_dist


def _update_distances(self, distance_matrix, clusters, idx1, idx2):
    """
    Aggiorna la matrice delle distanze dopo un merge di due cluster.
    """
    n = len(clusters)
    new_distance_matrix = distance_matrix.copy()

    # Crea il nuovo cluster mergato
    new_cluster = Cluster(clusters[idx1].indices + clusters[idx2].indices)

    # Rimuovi le righe e le colonne corrispondenti agli indici dei cluster mergati
    new_distance_matrix = np.delete(new_distance_matrix, [idx1, idx2], axis=0)
    new_distance_matrix = np.delete(new_distance_matrix, [idx1, idx2], axis=1)

    # Aggiungi una nuova riga e colonna per il nuovo cluster mergato
    for i in range(n - 2):
        new_distance_matrix[i, n - 2] = self._linkage_distance(distance_matrix, clusters[i], new_cluster)
        new_distance_matrix[n - 2, i] = new_distance_matrix[i, n - 2]

    return new_distance_matrix, new_cluster