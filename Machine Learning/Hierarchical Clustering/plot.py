import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Determina il percorso della cartella "Assets/plot"
current_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(current_dir, 'Assets', 'plot')
os.makedirs(plot_dir, exist_ok=True)  # Crea la cartella se non esiste


def save_dendrogram(linkage_matrix, file_name="dendrogram.png"):
    # Percorso completo per il file
    file_path = os.path.join(plot_dir, file_name)
    # Plot del dendrogramma
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title("Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.savefig(file_path)  # Salva l'immagine
    plt.close()  # Chiude la figura per liberare memoria
    print(f"Dendrogramma salvato in {file_path}")


def save_silhouette_plot(X, labels, file_name="silhouette_plot.png"):
    from sklearn.metrics import silhouette_samples
    import numpy as np

    file_path = os.path.join(plot_dir, file_name)

    # Calcolo dei campioni della silhouette
    silhouette_vals = silhouette_samples(X, labels)
    y_ticks = []
    y_lower, y_upper = 0, 0

    # Plot delle silhouette
    plt.figure(figsize=(10, 7))
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")  # Linea verticale per il valore medio
    plt.yticks(y_ticks, np.unique(labels))
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.title("Silhouette Plot")
    plt.savefig(file_path)  # Salva l'immagine
    plt.close()
    print(f"Plot della silhouette salvato in {file_path}")
