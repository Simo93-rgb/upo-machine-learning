import matplotlib.pyplot as plt
import numpy as np
from valutazione import validate_predictions
import seaborn as sns


def plot_predictions(y_true, y_pred, model_name=""):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    x_dim, y_dim = [16, 12]
    plt.figure(figsize=(x_dim, y_dim))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2)
    plt.title(f'Confronto Valori Predetti e Reali - {model_name}', fontsize=24)
    plt.xlabel('Valori Reali', fontsize=18)
    plt.ylabel('Valori Predetti', fontsize=18)
    plt.savefig(f'predictions_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true, y_pred, model_name=""):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    x_dim, y_dim = [16, 12]
    residuals = y_true - y_pred
    plt.figure(figsize=(x_dim, y_dim))
    plt.hist(residuals, bins=60, color='orange')
    plt.title(f'Distribuzione dei Residui - {model_name}', fontsize=24)
    plt.xlabel('Errore di Predizione', fontsize=18)
    plt.ylabel('Frequenza', fontsize=18)
    plt.savefig(f'residuals_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_corr_matrix(X):
    x_dim, y_dim = [20, 16]
    corr_matrix = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matrice di Correlazione delle Feature", fontsize=24)
    plt.savefig('correlation_matrix.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


