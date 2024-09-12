from ucimlrepo import fetch_ucirepo

from knn import KNN
from valutazione import *
from validazione import KFoldValidation
from plot import plot_predictions, plot_residuals, plot_corr_matrix

# Caricamento del dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    # Fetch del dataset Combined Cycle Power Plant
    combined_cycle_power_plant = fetch_ucirepo(id=294)

    # data (as pandas dataframes)
    X = combined_cycle_power_plant.data.features
    y = combined_cycle_power_plant.data.targets

    # Standardizzazione delle feature con z-score normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalizzazione del target y
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Suddivisione in train (60%), validation (20%) e test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    # Creazione del modello KNN
    knn = KNN(k=5)

    # Validazione incrociata (k-fold) sul set di training/validation
    k_fold_validator = KFoldValidation(model=knn, k_folds=5)
    metrix = k_fold_validator.validate(X_train, y_train)
    [print(f"{chiave} su cross validation (k-fold): {valore}\n") for chiave, valore in metrix.items()]

    # Addestramento finale sul training set completo
    knn.fit(X_train, y_train)

    # Predizioni su validation set (per confermare la bontà del modello)
    y_val_pred = knn.predict(X_val)
    y_val_pred_rescaled = scaler.inverse_transform(y_val_pred.reshape(-1, 1)).ravel()  # Riportare alla scala originale
    evaluate_model(y_true=scaler.inverse_transform(y_val.reshape(-1, 1)).ravel(), y_pred=y_val_pred_rescaled, message="Validation Set")

    # Predizioni su test set
    y_test_pred = knn.predict(X_test)
    y_test_pred_rescaled = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()  # Riportare alla scala originale
    evaluate_model(y_true=scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), y_pred=y_test_pred_rescaled, message="Test Set")

    # Visualizzazioni
    plot_predictions(scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), y_test_pred_rescaled, model_name="KNN")
    plot_residuals(scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), y_test_pred_rescaled, model_name="KNN")