import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from logistic_regression_with_gradient_descend import LogisticRegressionGD


def plot_sigmoid():
    # Crea un array di valori x
    x = np.linspace(-10, 10, 100)

    # Funzione sigmoidale
    sigmoid = 1 / (1 + np.exp(-x))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, sigmoid, color='blue')
    plt.title("Funzione Sigmoidale")
    plt.xlabel('z')
    plt.ylabel('Sigmoid(z)')
    plt.grid(True)
    plt.savefig('sigmoid_function.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_corr_matrix(X_normalized, features_eliminate=None):
    # Calcola la matrice di correlazione
    corr_matrix = np.corrcoef(X_normalized, rowvar=False)

    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 10})
    plt.title("Matrice di Correlazione delle Feature")

    # Evidenzia le feature eliminate
    if features_eliminate:
        for feature in features_eliminate:
            plt.gca().add_patch(plt.Rectangle((feature, feature), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.savefig('correlation_matrix_breast_cancer.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_class_distribution(y):
    # Conta il numero di campioni per ciascuna classe
    unique, counts = np.unique(y, return_counts=True)

    # Crea il grafico a barre
    plt.figure(figsize=(8, 6))
    plt.bar(unique, counts, color=['blue', 'orange'])
    plt.title('Distribuzione delle Classi')
    plt.xlabel('Classe')
    plt.ylabel('Numero di campioni')
    plt.xticks(unique, ['B (Benigno)', 'M (Maligno)'])
    plt.savefig('class_distribution_breast_cancer.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matrice di Confusione - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f'confusion_matrix_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


# Funzione per tracciare la curva ROC (modello personalizzato)
def plot_roc_curve(y_true, y_probs, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Curva ROC - {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(f'ROC_curve_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


# Funzione per tracciare la curva ROC (modello scikit-learn)
def plot_roc_curve_sklearn(model, X_val, y_val, model_name=""):
    y_probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    auc = roc_auc_score(y_val, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve {model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'ROC_curve_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison(metrics_dict, model_names):
    # Modifica questo dizionario con i risultati ottenuti dalle valutazioni
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(len(metrics))

    for i, model_name in enumerate(model_names):
        results = [metrics_dict[model_name][metric] for metric in metrics]
        ax.bar(index + i * bar_width, results, bar_width, label=model_name)

    ax.set_xlabel('Metriche')
    ax.set_title('Confronto delle Metriche tra i Modelli')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.savefig('metrics_comparison.png', format='png', dpi=600, bbox_inches='tight')

    plt.show()


def plot_regularization_effect(X_train, y_train, lambdas, regularization_type='ridge'):
    coefficients = []

    for _lambda in lambdas:
        model = LogisticRegressionGD(learning_rate=0.01, n_iterations=1000, lambda_=_lambda,
                                     regularization=regularization_type)
        model.fit(X_train, y_train)
        coefficients.append(model.theta)

    coefficients = np.array(coefficients)

    plt.figure(figsize=(10, 6))
    for i in range(coefficients.shape[1]):
        plt.plot(lambdas, coefficients[:, i], label=f'Feature {i}')

    plt.xscale('log')
    plt.title(f"Effetto della Regolarizzazione {regularization_type.capitalize()} sui Coefficienti")
    plt.xlabel("Lambda")
    plt.ylabel("Coefficients")
    plt.legend(loc='best')
    plt.savefig('regularization_effect.png', format='png', dpi=600, bbox_inches='tight')

    plt.show()
