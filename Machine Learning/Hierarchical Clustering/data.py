import os
from typing import List

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from ucimlrepo import fetch_ucirepo


class DataHandler:
    def __init__(self, file_path):
        self.data: DataFrame = pd.read_csv(file_path)


    def preprocess_data(self, soglia: float = 0.95, drop=None):
        # Esegui operazioni di pulizia, normalizzazione o riduzione delle caratteristiche, se necessario
        # Ad esempio: normalizzazione tra -1 e 1 per le caratteristiche MFCC
        if drop is None:
            drop = ['Species', 'Genus', 'RecordID']
        self.data = self.data.drop(columns=drop)
        self.data.iloc[:, :-1] = (self.data.iloc[:, :-1] - self.data.iloc[:, :-1].min()) / (
                self.data.iloc[:, :-1].max() - self.data.iloc[:, :-1].min())
        if soglia <=1:
            self.elimina_feature_correlate(soglia)


    def get_features(self):
        # Ritorna solo le colonne delle feature per il clustering
        return self.data.iloc[:, :-1].values  # Esclude le etichette

    def get_labels(self):
        # Ritorna le etichette per valutazione e visualizzazione
        return self.data.iloc[:, -1:]  # Colonne per Famiglia, Genere, e Specie

    def elimina_feature_correlate(self, soglia=0.95):
        """
        Elimina feature altamente correlate da un array numpy X basandosi sulla soglia fornita.

        Parametri:
        - X: array numpy bidimensionale, dove le colonne sono le feature.
        - soglia: valore soglia per la correlazione (default 0.95).

        Restituisce:
        - X_ridotto: array numpy con le feature eliminate.
        - feature_da_eliminare: set degli indici delle feature eliminate.
        """
        X = pd.DataFrame(self.data)
        family = X['Family'].copy()  # Salva la colonna Family
        X = X.drop(columns=['Family'])

        # Calcola la matrice di correlazione
        corr_matrix = np.corrcoef(X, rowvar=False)
        # Calcola la varianza di ogni feature
        feature_variances = np.var(X, axis=0)

        num_features = corr_matrix.shape[0]
        feature_da_eliminare = set()

        for i in range(num_features):
            if i in feature_da_eliminare:
                continue  # Salta le feature già eliminate
            for j in range(i + 1, num_features):
                if j in feature_da_eliminare:
                    continue  # Salta le feature già eliminate
                if abs(corr_matrix[i, j]) >= soglia:
                    # Confronta le varianze per decidere quale eliminare
                    if feature_variances[i] < feature_variances[j]:
                        feature_da_eliminare.add(i)
                        break  # Esci dal loop interno se la feature i è eliminata
                    else:
                        feature_da_eliminare.add(j)

        # Elimina le feature dal dataset
        X_ridotto = np.delete(X, list(feature_da_eliminare), axis=1)

        # Crea un nuovo dataframe con le colonne rimanenti e reinserisce Family
        colonne_rimaste = X.columns.difference([X.columns[i] for i in feature_da_eliminare])
        self.data = pd.DataFrame(X_ridotto, columns=colonne_rimaste)
        self.data['Family'] = family  # Aggiunge Family come ultima colonna
        print(f'Sono state eliminate {len(feature_da_eliminare)} features:')
        [print(f'{feature}') for feature in feature_da_eliminare]


if __name__ == '__main__':
    # Determina il percorso della cartella "Assets/plot"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'Assets', 'Dataset')

    # Inizializza il gestore dati e carica il dataset
    data_handler = DataHandler(f'{dataset_dir}/Frogs_MFCCs_reduced.csv')
    data_handler.data = data_handler.data.drop(columns=['Genus', 'Species', 'RecordID'])

    # fetch dataset
    hepatitis = fetch_ucirepo(id=46)

    # data (as pandas dataframes)
    X = hepatitis.data.features
    y = hepatitis.data.targets

    # Concatenare X e y in un unico DataFrame
    data = pd.concat([X, y], axis=1)

    # Salvare il DataFrame concatenato in un file CSV
    data.to_csv(f'{dataset_dir}/hepatitis.csv', index=False)
