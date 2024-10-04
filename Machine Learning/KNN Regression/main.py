from sklearn.discriminant_analysis import StandardScaler
from knn import KNN
from valutazione import *
from validazione import KFoldValidation
from plot import plot_predictions, plot_residuals, plot_learning_curve
from data import fetch_data, edit_dataset
import os

# Percorso del file main.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Percorso alla cartella "Assets" nella directory "KNN Regression"
assets_dir = os.path.join(current_dir, 'Assets')


if __name__ == "__main__":

    # Fetch del dataset Combined Cycle Power Plant
    X, y = fetch_data(assets_dir)

    # Manipolazione dataset con opzioni di standardizzazione per X e y
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = edit_dataset(X, y, X_standardization=True, y_standardization=True)

    # Creazione del modello KNN
    knn = KNN(k=50)

    # Plot della curva di apprendimento
    plot_learning_curve(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, model=knn, assets_dir=assets_dir)

    # Validazione incrociata (k-fold) sul set di trai∟ning/validation
    k_fold_validator = KFoldValidation(model=knn, k_folds=10)
    metrix = k_fold_validator.validate(X_train, y_train)
    [print(f"{chiave} su cross validation (k-fold): {valore}\n") for chiave, valore in metrix.items()]

    # Addestramento finale sul training set completo
    knn.fit(X_train, y_train)

    # Predizioni su test set
    y_test_pred = knn.predict(X_test)

    # Se è stata applicata la standardizzazione su y, esegui l'inverso della trasformazione
    if y_scaler:
        y_test_pred_rescaled = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()  # Riportare alla scala originale
        y_test_rescaled = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_test_pred_rescaled = y_test_pred
        y_test_rescaled = y_test

    # Valutazione del modello
    evaluate_model(y_true=y_test_rescaled, y_pred=y_test_pred_rescaled, message="Test Set")

    # Visualizzazioni
    plot_predictions(y_test_rescaled, y_test_pred_rescaled, model_name="KNN", assets_dir=assets_dir)
    plot_residuals(y_test_rescaled, y_test_pred_rescaled, model_name="KNN", assets_dir=assets_dir)
