import os

import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame

from data import DataHandler
from hierarchical_clustering_prove import HierarchicalClustering
from evaluation import calculate_and_save_silhouette, purity_score
from plot import save_dendrogram, save_silhouette_plot
import gc

# Imposta il flag per un comportamento più aggressivo del garbage collector
# gc.set_debug(gc.DEBUG_COLLECTABLE)
# Configurazione dei percorsi
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'Assets', 'Dataset')
output_dir = os.path.join(current_dir, 'Assets', 'Results')
os.makedirs(output_dir, exist_ok=True)
dataset_name = 'Frogs_MFCCs_reduced'

# Inizializza il gestore dati e carica il dataset
data_handler = DataHandler(f'{dataset_dir}/{dataset_name}.csv')
data_handler.preprocess_data()
X:DataFrame = data_handler.get_features()
y = data_handler.get_labels().iloc[:, -1].values  # Usa la colonna delle specie per la valutazione
print(f'Caricamento Dataset {dataset_name}')



# K-Means di scikitlearn
def kmeans_pre_clustering(X, n_clusters=10):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)
    return kmeans.cluster_centers_, labels


# Inizializza il modello di clustering gerarchico
hc = HierarchicalClustering(linkage='single', X=X, pre_clustering_func=kmeans_pre_clustering, n_clusters=10 )
print(f'Inizio fit')
hc.fit()
print('Fine fit')
# Usa cluster_history per il dendrogramma
print('Creazione dendogramma')
dendrogram_path = os.path.join(output_dir, "dendrogram.png")
# Crea una lista di tuple con indici numerici per i cluster uniti
linkage_matrix = []
current_idx = 0  # Indice per i cluster

# Crea un dizionario per mappare i nomi ai numeri di cluster
name_to_idx = {}

for a, b, dist in hc.cluster_history:
    if a not in name_to_idx:
        name_to_idx[a] = current_idx
        current_idx += 1
    if b not in name_to_idx:
        name_to_idx[b] = current_idx
        current_idx += 1

    # Aggiungi la fusione al linkage_matrix
    linkage_matrix.append([name_to_idx[a], name_to_idx[b], dist, 2])  # "2" è il numero di elementi nei cluster uniti

# Converti la lista in un array numpy
linkage_matrix = np.array(linkage_matrix)
save_dendrogram(linkage_matrix, file_name=dendrogram_path)

# Previsione e valutazione
num_clusters = 4  # Numero di cluster desiderato
cluster_objects = hc.predict(num_clusters)

# Ora cluster_objects è una lista di oggetti Cluster
# Per ottenere solo i nomi dei cluster, per esempio:
cluster_names = [cluster for cluster in cluster_objects]

# Calcola e salva il punteggio di silhouette
silhouette_path = os.path.join(output_dir, "silhouette_score.csv")
calculate_and_save_silhouette(X, cluster_names, file_name=silhouette_path)
#
# # Calcola e salva la purezza dei cluster
# purity_path = os.path.join(output_dir, "purity_score.csv")
# purity = purity_score(y, cluster_names, file_name=purity_path)
#
# # Plot e salva il grafico della silhouette
# silhouette_plot_path = os.path.join(output_dir, "silhouette_plot.png")
# save_silhouette_plot(X, cluster_names, file_name=silhouette_plot_path)
#
# print("Progetto completato e risultati salvati.")
# #Disabilita il flag per ripristinare il comportamento predefinito del garbage collector
# #gc.set_debug(0)