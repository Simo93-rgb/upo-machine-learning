import os
from data import DataHandler
from hierarchical_clustering import HierarchicalClustering
from evaluation import calculate_and_save_silhouette, purity_score
from plot import save_dendrogram, save_silhouette_plot

# Determina il percorso della cartella "Assets/plot"
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'Assets', 'Dataset')

# Inizializza il gestore dati e carica il dataset
data_handler = DataHandler(f'{dataset_dir}/Frogs_MFCCs_reduced.csv')
data_handler.preprocess_data()
X = data_handler.get_features()
y = data_handler.get_labels().iloc[:, -1].values  # Usa la colonna delle specie per la valutazione

# Inizializza il modello di clustering gerarchico
hc = HierarchicalClustering(linkage='single')
hc.fit(X)

# Plot e salva il dendrogramma
save_dendrogram(hc.linkage_matrix, file_name="dendrogram.png")

# Previsione e valutazione
num_clusters = 10  # Numero di cluster desiderato
labels = hc.predict(num_clusters)

# Valutazione e salvataggio delle metriche
calculate_and_save_silhouette(X, labels, file_name="silhouette_score.csv")
purity = purity_score(y, labels, file_name="purity_score.csv")

# Plot e salva il grafico della silhouette
save_silhouette_plot(X, labels, file_name="silhouette_plot.png")

print("Progetto completato e risultati salvati.")