import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score


def mean_squared_error(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score_metric(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return r2_score(y_true, y_pred)


def explained_variance(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return explained_variance_score(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    epsilon = 1e-10  # Per evitare la divisione per zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def evaluate_model(y_true, y_pred, message=""):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    if message:
        message = "su " + message

    print(f"Mean Squared Error (MSE) {message}:", mean_squared_error(y_true, y_pred))
    print(f"Root Mean Squared Error (RMSE) {message}:", root_mean_squared_error(y_true, y_pred))
    print(f"Mean Absolute Error (MAE) {message}:", mean_absolute_error(y_true, y_pred))
    print(f"R² Score {message}:", r2_score_metric(y_true, y_pred))
    print(f"Explained Variance Score {message}:", explained_variance(y_true, y_pred))
    print(f"Mean Absolute Percentage Error (MAPE) {message}:", mean_absolute_percentage_error(y_true, y_pred))


def validate_predictions(y_true, y_pred):
    # Verifica che y_true e y_pred siano array NumPy, liste o Serie/DataFrame Pandas
    assert isinstance(y_true, (list, np.ndarray, pd.Series,
                               pd.DataFrame)), f"Errore: y_true deve essere una lista, un array NumPy, una Serie o un DataFrame Pandas, ma è {type(y_true)}"
    assert isinstance(y_pred, (list, np.ndarray, pd.Series,
                               pd.DataFrame)), f"Errore: y_pred deve essere una lista, un array NumPy, una Serie o un DataFrame Pandas, ma è {type(y_pred)}"

    # Se y_true o y_pred sono DataFrame, estrai la prima colonna
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.iloc[:, 0].values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.iloc[:, 0].values

    # Se y_true o y_pred sono Serie Pandas, converti in NumPy array
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Converti in array NumPy se sono liste
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    # Verifica che abbiano la stessa lunghezza
    assert len(y_true) == len(
        y_pred), f"Errore: y_true e y_pred devono avere la stessa lunghezza. Lunghezza di y_true: {len(y_true)}, Lunghezza di y_pred: {len(y_pred)}"

    # Verifica che i tipi di dato siano consistenti
    assert y_true.dtype == y_pred.dtype, f"Errore: y_true e y_pred devono essere dello stesso tipo. Tipo di y_true: {y_true.dtype}, Tipo di y_pred: {y_pred.dtype}"

    # Verifica che y_true e y_pred contengano solo valori numerici
    assert np.issubdtype(y_true.dtype, np.number), "Errore: y_true deve contenere solo valori numerici."
    assert np.issubdtype(y_pred.dtype, np.number), "Errore: y_pred deve contenere solo valori numerici."

    # Verifica che non ci siano NaN nei dati
    assert not np.isnan(y_true).any(), "Errore: y_true contiene valori NaN."
    assert not np.isnan(y_pred).any(), "Errore: y_pred contiene valori NaN."

    # Verifica che non ci siano valori infiniti
    assert np.isfinite(y_true).all(), "Errore: y_true contiene valori infiniti."
    assert np.isfinite(y_pred).all(), "Errore: y_pred contiene valori infiniti."

    # Se tutti gli assert passano, ritorna gli array validati
    # print("Tutti i controlli sono stati superati. y_true e y_pred sono validi.")
    return y_true, y_pred
