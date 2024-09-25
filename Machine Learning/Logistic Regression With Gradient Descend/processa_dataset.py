# Carica e pre-processa i dati
from funzioni import *

X, y = carica_dati()
X_normalized, features_eliminate, y_encoded = preprocessa_dati(X, y, class_balancer="", corr=1.01, save_dataset=True)
