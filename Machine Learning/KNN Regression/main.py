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
    parser.add_argument('--X_standardization', '-X', type=str2bool, nargs='?', default=True, help='Enable/Disable standardization of X (default: True)')
    parser.add_argument('--n_neighboors', '-n', type=int, nargs='?', default=22, help='Value of k for KNN (default: 22)')
    parser.add_argument('--test_size', '-t', type=float, nargs='?', default=0.02, help='Value of k for KNN (default: 0.2)')

    args = parser.parse_args()

    # Utilizzo degli argomenti passati
    X_standardization = args.X_standardization
    n = args.n_neighboors
    test_size=args.test_size

    print(f'Args:\nX_standardization = {X_standardization}\nKNN(k={n})\ntest_size = {test_size}')

    # Fetch del dataset Combined Cycle Power Plant
    X, y = fetch_data(assets_dir)

    # Manipolazione dataset con opzioni di standardizzazione per X e y
    X_train, X_test, y_train, y_test, x_scaler = edit_dataset(
        X,
        y,
        X_standardization=X_standardization,
        test_size=test_size
    )

    # Creazione del modello KNN
    knn = KNN_Parallel(k=n)



    # Validazione incrociata (k-fold) sul set di trai∟ning/validation
    k_fold_validator = KFoldValidation(model=knn, k_folds=10)
    # metrix = k_fold_validator.validate_and_find_n_neighbors(X_train, y_train)
    metrix = k_fold_validator.validate(X_train, y_train)
    print('######## K-FOLD ########')
    [print(f"{chiave} su cross validation (k-fold): {valore}") for chiave, valore in metrix.items()]

    # Addestramento finale sul training set completo
    knn.fit(X_train, y_train)

    # Predizioni su test set
    y_test_pred = knn.predict(X_test)


    # Se è stata applicata la standardizzazione su X, esegui l'inverso della trasformazione
    if x_scaler:
        X_train_rescaled = x_scaler.inverse_transform(X_train)
        X_test_rescaled = x_scaler.inverse_transform(X_test)
    else:
        X_train_rescaled = X_train
        X_test_rescaled = X_test

    # Valutazione del modello
    print('######## Evaluating my model ########')
    evaluate_model(y_true=y_test, y_pred=y_test_pred, message="Test Set")

    # Visualizzazioni
    # Plot della curva di apprendimento


    plot_predictions(y_test, y_test_pred, model_name="KNN", assets_dir=assets_dir)
    plot_residuals(y_test, y_test_pred, model_name="KNN", assets_dir=assets_dir)
