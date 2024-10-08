import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error
from valutazione import validate_predictions
import seaborn as sns
from data import fetch_data, edit_dataset
from knn import KNN
from knn_parallel import  KNN_Parallel
import os
import argparse

# Percorso del file main.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Percorso alla cartella "Assets" nella directory "KNN Regression"
assets_dir = os.path.join(current_dir, 'Assets')

def plot_predictions(y_true, y_pred, model_name="", assets_dir=""):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    x_dim, y_dim = [16, 9]
    plt.figure(figsize=(x_dim, y_dim))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2)
    plt.title(f'Confronto Valori Predetti e Reali - {model_name}', fontsize=24)
    plt.xlabel('Valori Reali', fontsize=18)
    plt.ylabel('Valori Predetti', fontsize=18)
    plt.savefig(f'{assets_dir}/predictions_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_residuals(y_true, y_pred, model_name="", assets_dir=""):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    x_dim, y_dim = [16, 9]
    residuals = y_true - y_pred
    plt.figure(figsize=(x_dim, y_dim))
    plt.hist(residuals, bins=60, color='orange')
    plt.title(f'Distribuzione dei Residui - {model_name}', fontsize=24)
    plt.xlabel('Errore di Predizione', fontsize=18)
    plt.ylabel('Frequenza', fontsize=18)
    plt.savefig(f'{assets_dir}/residuals_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_corr_matrix(X, assets_dir=""):
    x_dim, y_dim = [20, 14]
    corr_matrix = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matrice di Correlazione delle Feature", fontsize=24)
    plt.savefig(f'{assets_dir}/correlation_matrix.png', format='png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_learning_curve(model, X_train, y_train, X_test, y_test, assets_dir="", file_name='learning_curve'):
    print('Plotting Learning Curve')
    """ Plotta la curva di apprendimento per un dato modello KNN. """

    train_errors = []
    val_errors = []

    # Dividiamo il dataset di training in dimensioni crescenti
    training_sizes = np.linspace(0.1, 1.0, 10)  # dal 10% al 100% del training set

    for size in training_sizes:
        print(f'Training on {round(size*100, None)}% of dataset')
        # Determina il numero di campioni da utilizzare
        current_size = int(size * len(X_train))


        # Prendi un sottoinsieme di X_train e y_train
        X_train_subset = X_train[:current_size]
        y_train_subset = y_train[:current_size]

        # Allena il modello sul sottoinsieme
        model.fit(X_train_subset, y_train_subset)

        # Calcola l'errore sul training set
        y_train_pred = model.predict(X_train_subset)
        train_errors.append(root_mean_squared_error(y_train_subset, y_train_pred))  # RMSE

        # Calcola l'errore sul validation set
        y_val_pred = model.predict(X_test)
        val_errors.append(root_mean_squared_error(y_test, y_val_pred))  # RMSE

    # Plot della learning curve
    plt.figure(figsize=(16, 9))
    plt.plot(training_sizes * len(X_train), train_errors, label='Errore Training Set', marker='o')
    plt.plot(training_sizes * len(X_train), val_errors, label='Errore Validation Set', marker='o')
    plt.xlabel('Dimensione Training Set')
    plt.ylabel('RMSE')
    plt.title('Curva di Apprendimento - KNN')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{assets_dir}/{file_name}.png', format='png', dpi=600, bbox_inches='tight')

    # plt.show()



if __name__ == '__main__':
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

    # Carica i dati e prepara il dataset
    X, y = fetch_data(assets_dir)
    X_train, X_test, y_train, y_test, _, _ = edit_dataset(X, y, X_standardization=X_standardization, y_standardization=y_standardization, test_size=test_size)

    # Crea il modello KNN
    knn = KNN_Parallel(k=k)
    print(f'Parallel Plotting:\nX_standardization = {X_standardization}\ny_standardization = {y_standardization}\nKNN(k={k})\ntest_size = {test_size}')

    # Plot della learning curve
    plot_learning_curve(
        knn, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        assets_dir, 
        file_name=f'/std/learning_curve_X_st_{X_standardization}_y_st_{y_standardization}_k_{k}_test_size_{test_size}'
    )
