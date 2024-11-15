from funzioni import *


def single_run(max_clusters: int = 8, k_means_reduction: int = 10, optimal_k=-1):
    # Setup iniziale
    dataset_dir, output_dir, plot_dir = setup_directories()
    dataset_name = 'Frogs_MFCCs'
    dataset_path = os.path.join(dataset_dir, f'{dataset_name}.csv')

    # Caricamento e pre-processing dei dati
    X, y = load_and_preprocess_data(dataset_path)
    print(f'Dataset {dataset_name} caricato e pre-processato')

    # Definizione dei metodi di linkage e delle metriche di distanza da utilizzare
    linkage_methods = ['single', 'complete', 'average', 'centroid']
    distance_metrics = ['euclidean', 'manhattan']

    run_clustering(X,
                   y,
                   linkage_methods[1],
                   distance_metrics[1],
                   output_dir,
                   plot_dir,
                   max_clusters=max_clusters,
                   k_means_reduction=k_means_reduction,
                   optimal_k=optimal_k)

    print("Progetto completato e tutti i risultati salvati.")


def multi_run(max_clusters: int = 8, k_means_reduction: int = 10, optimal_k=-1):
    # Setup iniziale
    dataset_dir, output_dir, plot_dir = setup_directories()
    dataset_name = 'Frogs_MFCCs'
    dataset_path = os.path.join(dataset_dir, f'{dataset_name}.csv')

    # Caricamento e pre-processing dei dati
    X, y = load_and_preprocess_data(dataset_path)
    print(f'Dataset {dataset_name} caricato e pre-processato')

    # Definizione dei metodi di linkage e delle metriche di distanza da utilizzare
    linkage_methods = ['single', 'complete', 'average', 'centroid']
    distance_metrics = ['euclidean', 'manhattan']

    # Esecuzione del clustering per ogni combinazione di linkage e distanza
    for linkage in linkage_methods:
        for distance in distance_metrics:
            run_clustering(X,
                           y,
                           linkage,
                           distance,
                           output_dir,
                           plot_dir,
                           max_clusters=max_clusters,
                           k_means_reduction=k_means_reduction,
                           optimal_k=optimal_k)

    print("Progetto completato e tutti i risultati salvati.")


if __name__ == "__main__":
    single_run(8, 10, 4)

#
# # Imposta il flag per un comportamento più aggressivo del garbage collector
# # gc.set_debug(gc.DEBUG_COLLECTABLE)
# # Configurazione dei percorsi
# current_dir = os.path.dirname(os.path.abspath(__file__))
# dataset_dir = os.path.join(current_dir, 'Assets', 'Dataset')
# output_dir = os.path.join(current_dir, 'Assets', 'Results')
# plot_dir = os.path.join(current_dir, 'Assets', 'Plot')
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(plot_dir, exist_ok=True)
# dataset_name = 'Frogs_MFCCs'
#
# # Inizializza il gestore dati e carica il dataset
# data_handler = DataHandler(f'{dataset_dir}/{dataset_name}.csv')
# data_handler.preprocess_data()
# X: pd.DataFrame = data_handler.get_features()
# y = data_handler.get_labels().iloc[:, -1].values  # Usa la colonna delle specie per la valutazione
# print(f'Caricamento Dataset {dataset_name}')
# n_clusters = 15
#
# # K-Means di scikitlearn
# def kmeans_pre_clustering(X, n_clusters=n_clusters):
#     from sklearn.cluster import KMeans
#     kmeans = KMeans(n_clusters=n_clusters)
#     labels = kmeans.fit_predict(X)
#     return kmeans.cluster_centers_, labels
#
#
# # Inizializza il modello di clustering gerarchico
# hc = HierarchicalClustering(linkage='complete', X=X, pre_clustering_func=kmeans_pre_clustering, n_clusters=n_clusters)
# print(f'Inizio fit')
# hc.fit()
# print('Fine fit')
# # Usa cluster_history per il dendrogramma
#
# # Crea una lista di tuple con indici numerici per i cluster uniti
# linkage_matrix = []
# current_idx = 0  # Indice per i cluster
#
# # Crea un dizionario per mappare i nomi ai numeri di cluster
# name_to_idx = {}
#
# for a, b, dist in hc.get_cluster_history():
#     if a not in name_to_idx:
#         name_to_idx[a.name] = float(a.name)
#         current_idx += 1
#     if b not in name_to_idx:
#         name_to_idx[b.name] = float(b.name)
#         current_idx += 1
#
#     # Aggiungi la fusione al linkage_matrix
#     linkage_matrix.append([name_to_idx[a.name], name_to_idx[b.name], dist, len(a.dataset_indices) + len(
#         b.dataset_indices)])  # "2" è il numero di elementi nei cluster uniti
#
# # Converti la lista in un array numpy
# linkage_matrix = np.array(linkage_matrix)
# # Previsione e valutazione
# # Trova il numero ottimale di cluster
# max_clusters = 8  # Puoi modificare questo valore
# optimal_k = find_optimal_clusters(X, max_clusters, hc.predict)
#
# print(f"Numero ottimale di cluster: {optimal_k}")
# labels = hc.predict(optimal_k)
#
# # Calcola la matrice di confusione
# conf_matrix = calculate_confusion_matrix(y, labels)
#
# # Calcola e salva varie metriche di valutazione
# evaluation_results = {
#     "purity": calculate_purity(conf_matrix),
#     "f1_score": calculate_f1_score(conf_matrix),
#     "false_positive_rate": calculate_false_positive_rate(conf_matrix),
#     "silhouette_score": calculate_silhouette(X, labels)[0]
# }
#
# save_evaluation_results(evaluation_results, "evaluation_results.csv", output_dir)
#
# # Salva i plot
# save_dendrogram(linkage_matrix, plot_dir)
# save_silhouette_plot(X, labels, optimal_k, plot_dir)
#
# print("Progetto completato e risultati salvati.")
#
# # # Calcola e salva il punteggio di silhouette
# # silhouette_path = os.path.join(output_dir, "silhouette_score.csv")
# # calculate_and_save_silhouette(X, labels, file_name=silhouette_path)
# #
# # # Calcola e salva la purezza dei cluster
# # purity_path = os.path.join(output_dir, "purity_score.csv")
# # purity = purity_score(y, labels, file_name=purity_path)
# #
# # # Plot e salva il grafico della silhouette
# # silhouette_plot_path = os.path.join(output_dir, "silhouette_plot.png")
# # save_silhouette_plot(X, labels, file_name=silhouette_plot_path)
# #
# # print("Progetto completato e risultati salvati.")
# # #Disabilita il flag per ripristinare il comportamento predefinito del garbage collector
# # #gc.set_debug(0)
