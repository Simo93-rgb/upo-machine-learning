import pandas as pd
import os

# Directory base
base_dir = 'Assets/Results'

# Lista per salvare tutti i risultati
results = []

# Scansione di tutte le cartelle
for dirpath, dirnames, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirpath, filename)
            try:
                # Lettura del CSV
                df = pd.read_csv(file_path)

                # Aggiungiamo il percorso della cartella come informazione
                df['folder'] = os.path.basename(dirpath)
                results.append(df)

            except Exception as e:
                print(f'Errore nella lettura di {filename}: {e}')

# Combiniamo tutti i risultati in un unico DataFrame
if results:
    all_results = pd.concat(results, ignore_index=True)
    top = 20
    # Ordiniamo per silhouette_score in ordine decrescente e prendiamo i primi 3
    top_3 = all_results.sort_values('silhouette_score', ascending=False).head(top)

    # Selezioniamo solo le colonne rilevanti per la visualizzazione
    columns_to_show = ['folder', 'model_name', 'silhouette_score', 'silhouette_clusters', 'dendogram_clusters']
    path = os.path.join(base_dir,'beast_results.json' )
    #top_3.to_html(path, index=False, border=0)
    top_3.to_json(path, orient='records', indent=4)
    print(f"\nI {top} migliori risultati:")
    print(top_3[columns_to_show])
else:
    print("Nessun risultato trovato")