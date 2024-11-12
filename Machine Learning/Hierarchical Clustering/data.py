import os

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


class DataHandler:
    def __init__(self, file_path):
        self.data:DataFrame = pd.read_csv(file_path)
        self.data = self.data.drop(columns=['Species', 'Genus', 'MFCCs_1', 'RecordID'])


    def preprocess_data(self):
        # Esegui operazioni di pulizia, normalizzazione o riduzione delle caratteristiche, se necessario
        # Ad esempio: normalizzazione tra -1 e 1 per le caratteristiche MFCC
        self.data.iloc[:, :-1] = (self.data.iloc[:, :-1] - self.data.iloc[:, :-1].min()) / (
                    self.data.iloc[:, :-1].max() - self.data.iloc[:, :-1].min())


    def get_features(self):
        # Ritorna solo le colonne delle feature per il clustering
        return self.data.iloc[:, :-1].values  # Esclude le etichette

    def get_labels(self):
        # Ritorna le etichette per valutazione e visualizzazione
        return self.data.iloc[:, -1:]  # Colonne per Famiglia, Genere, e Specie



if __name__ == '__main__':
    # Determina il percorso della cartella "Assets/plot"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'Assets', 'Dataset')

    # Inizializza il gestore dati e carica il dataset
    data_handler = DataHandler(f'{dataset_dir}/Frogs_MFCCs_reduced.csv')
    data_handler.data = data_handler.data.drop(columns=['Genus','Species','RecordID'])


    # fetch dataset
    hepatitis = fetch_ucirepo(id=46)

    # data (as pandas dataframes)
    X = hepatitis.data.features
    y = hepatitis.data.targets

    # Concatenare X e y in un unico DataFrame
    data = pd.concat([X, y], axis=1)

    # Salvare il DataFrame concatenato in un file CSV
    data.to_csv(f'{dataset_dir}/hepatitis.csv', index=False)
