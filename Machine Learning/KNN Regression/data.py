import os
from typing import Optional, Tuple 
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def fetch_data(assets_dir: str = "") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carica o scarica il dataset e lo restituisce.

    Parameters:
    - assets_dir (str): Directory degli asset.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: I dati X (features) e y (target).
    """
    if not os.path.exists(f'{assets_dir}/CCPP.csv'):
        combined_cycle_power_plant = fetch_ucirepo(id=17)
        X = combined_cycle_power_plant.data.features
        y = combined_cycle_power_plant.data.targets
        
        # Crea un DataFrame unendo X e y
        df = pd.DataFrame(X, columns=X.columns)
        df['target'] = y
        # Salva in CSV
        csv_file = os.path.join(assets_dir, f'CCPP.csv')
        df.to_csv(csv_file, index=False)
        print(f"Dataset salvato in {csv_file}")
    else:
        df = pd.read_csv(f'{assets_dir}/CCPP.csv')
        X = df.drop(columns='target')
        y = df['target']

        # Converti X e y in valori numerici se ci sono stringhe
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

    return X, y


def edit_dataset(X: pd.DataFrame, y: pd.Series, X_standardization: bool = True, y_standardization: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Standardizza il dataset e lo suddivide in training e test.

    Parameters:
    - X (pd.DataFrame): Le features.
    - y (pd.Series): Il target.
    - X_standardization (bool): Se standardizzare o meno X.
    - y_standardization (bool): Se standardizzare o meno y.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], Optional[StandardScaler]]:
      X_train, X_test, y_train, y_test, e gli scaler (se usati).
    """
    x_scaler, y_scaler = None, None  # Inizializzo gli scaler a None

    if X_standardization:
        # Standardizzazione delle feature con z-score normalization
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X)

    if y_standardization:
        # Normalizzazione del target y
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Suddivisione in train (80%) e test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, x_scaler, y_scaler
