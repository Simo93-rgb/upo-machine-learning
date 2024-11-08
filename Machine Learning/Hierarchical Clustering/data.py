import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data = self.data.drop(columns=['Species', 'RecordID', 'Genus'])

    def preprocess_data(self):
        # Esegui operazioni di pulizia, normalizzazione o riduzione delle caratteristiche, se necessario
        # Ad esempio: normalizzazione tra -1 e 1 per le caratteristiche MFCC
        self.data.iloc[:, :-3] = (self.data.iloc[:, :-3] - self.data.iloc[:, :-3].min()) / (
                    self.data.iloc[:, :-3].max() - self.data.iloc[:, :-3].min())

    def get_features(self):
        # Ritorna solo le colonne delle feature per il clustering
        return self.data.iloc[:, :-1].values  # Esclude le etichette

    def get_labels(self):
        # Ritorna le etichette per valutazione e visualizzazione
        return self.data.iloc[:, -1:]  # Colonne per Famiglia, Genere, e Specie
