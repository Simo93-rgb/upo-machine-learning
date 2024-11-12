import os

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, file_path):
        self.data:DataFrame = pd.read_csv(file_path)
        self.data = self.data.drop(columns=['Species', 'Genus', 'MFCCs_1', 'RecordID'])


    def preprocess_data(self):
        # Esegui operazioni di pulizia, normalizzazione o riduzione delle caratteristiche, se necessario
        # Ad esempio: normalizzazione tra -1 e 1 per le caratteristiche MFCC
        self.data.iloc[:, :-1] = (self.data.iloc[:, :-1] - self.data.iloc[:, :-1].min()) / (
                    self.data.iloc[:, :-1].max() - self.data.iloc[:, :-1].min())
        # Esegui lo split dei dati
        self.data, X_test = train_test_split(self.data, test_size=0.8, random_state=42)

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
