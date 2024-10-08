import argparse

from scipy.special.cython_special import eval_sh_legendre
from sklearn.discriminant_analysis import StandardScaler
from knn_parallel import KNN_Parallel
from valutazione import *
from validazione import KFoldValidation
from plot import plot_predictions, plot_residuals, plot_learning_curve
from data import fetch_data, edit_dataset
import os

# Percorso del file main.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Percorso alla cartella "Assets" nella directory "KNN Regression"
assets_dir = os.path.join(current_dir, 'Assets')


# Funzione per convertire stringhe in booleani
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # Parser degli argomenti posizionali
    parser = argparse.ArgumentParser(description="KNN with optional standardization and k value.")
    parser.add_argument('X_standardization', type=str2bool, nargs='?', default=True, help='Enable/Disable standardization of X (default: True)')
    parser.add_argument('y_standardization', type=str2bool, nargs='?', default=True, help='Enable/Disable standardization of y (default: True)')
    parser.add_argument('k', type=int, nargs='?', default=50, help='Value of k for KNN (default: 50)')
    parser.add_argument('test_size', type=float, nargs='?', default=0.2, help='Value of k for KNN (default: 0.2)')

    args = parser.parse_args()

    # Utilizzo degli argomenti passati
    X_standardization = args.X_standardization
    y_standardization = args.y_standardization
    k = args.k
    test_size=args.test_size
    # Fetch del dataset Combined Cycle Power Plant
    X, y = fetch_data(assets_dir)

    # Manipolazione dataset con opzioni di standardizzazione per X e y
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = edit_dataset(
        X,
        y,
        X_standardization=X_standardization,
        y_standardization=y_standardization,
        test_size=test_size
    )

    # Creazione del modello KNN
    knn = KNN_Parallel(k=k)



    # Validazione incrociata (k-fold) sul set di trai∟ning/validation
    k_fold_validator = KFoldValidation(model=knn, k_folds=10)
    metrix = k_fold_validator.validate(X_train, y_train, y_scaler)
    print('######## K-FOLD ########')
    [print(f"{chiave} su cross validation (k-fold): {valore}") for chiave, valore in metrix.items()]

    # Addestramento finale sul training set completo
    knn.fit(X_train, y_train)

    # Predizioni su test set
    y_test_pred = knn.predict(X_test)

    # Se è stata applicata la standardizzazione su y, esegui l'inverso della trasformazione
    if y_scaler:
        y_test_pred_rescaled = y_scaler.inverse_transform(
        y_test_pred.reshape(-1, 1)).ravel()  # Riportare alla scala originale
        y_test_rescaled = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_train_rescaled = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    else:
        y_test_pred_rescaled = y_test_pred
        y_test_rescaled = y_test
        y_train_rescaled = y_train

    # Se è stata applicata la standardizzazione su X, esegui l'inverso della trasformazione
    if x_scaler:
        X_train_rescaled = x_scaler.inverse_transform(X_train)
        X_test_rescaled = x_scaler.inverse_transform(X_test)
    else:
        X_train_rescaled = X_train
        X_test_rescaled = X_test

    # Valutazione del modello
    print('######## Evaluating my model ########')
    evaluate_model(y_true=y_test_rescaled, y_pred=y_test_pred_rescaled, message="Test Set")

    # Visualizzazioni
    # Plot della curva di apprendimento
    plot_learning_curve(
        X_train=X_train_rescaled,
        y_train=y_train_rescaled,  # Uso di y_train riscalato
        X_test=X_test_rescaled,  # Uso di X_test riscalato
        y_test=y_test_rescaled,  # Uso di y_test riscalato
        model=knn,
        assets_dir=assets_dir,
        file_name=f'/not_std/learning_curve_X_st_{X_standardization}_y_st_{y_standardization}_k_{k}_test_size_{test_size}'
    )

    plot_predictions(y_test_rescaled, y_test_pred_rescaled, model_name="KNN", assets_dir=assets_dir)
    plot_residuals(y_test_rescaled, y_test_pred_rescaled, model_name="KNN", assets_dir=assets_dir)
