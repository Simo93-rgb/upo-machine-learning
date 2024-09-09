import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from logistic_regression_with_gradient_descend import LogisticRegressionGD


def plot_sigmoid():
    x_dim, y_dim = [16, 12]
    # Crea un array di valori x
    x = np.linspace(-10, 10, 100)

    # Funzione sigmoidale
    sigmoid = 1 / (1 + np.exp(-x))

    # Plot
    plt.figure(figsize=(x_dim, y_dim))
    plt.plot(x, sigmoid, color='blue')
    plt.title("Funzione Sigmoidale", fontsize=24)
    plt.xlabel('z', fontsize=18)
    plt.ylabel('Sigmoid(z)', fontsize=18)
    plt.grid(True)
    plt.savefig('sigmoid_function.png', format='png', dpi=1200, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_corr_matrix(X_normalized, features_eliminate=None):
    x_dim, y_dim = [20, 16]
    # Calcola la matrice di correlazione
    corr_matrix = np.corrcoef(X_normalized, rowvar=False)

    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12})  # Font migliorato
    plt.title("Matrice di Correlazione delle Feature", fontsize=24)

    # Evidenzia le feature eliminate
    if features_eliminate:
        for feature in features_eliminate:
            plt.gca().add_patch(plt.Rectangle((feature, feature), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.savefig('correlation_matrix_breast_cancer.png', format='png', dpi=1200, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_class_distribution(y, file_name=""):
    x_dim, y_dim = [16, 12]
    # Conta il numero di campioni per ciascuna classe
    unique, counts = np.unique(y, return_counts=True)

    # Calcola le percentuali
    total = np.sum(counts)
    percentages = [(count / total) * 100 for count in counts]

    # Etichette con la percentuale
    labels = [f'B (Benigno) - {percentages[0]:.1f}%', f'M (Maligno) - {percentages[1]:.1f}%']

    # Crea un grafico a torta
    plt.figure(figsize=(x_dim, y_dim))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'orange'], startangle=90,
            textprops={'fontsize': 18})  # Font migliorato
    plt.title('Distribuzione delle Classi', fontsize=24)
    plt.axis('equal')  # Assicura che il grafico sia disegnato come un cerchio
    if file_name:
        plt.savefig(f'{file_name}.png', format='png', dpi=1200, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    x_dim, y_dim = [16, 12]
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16})  # Font migliorato
    plt.title(f"Matrice di Confusione - {model_name}", fontsize=24)
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("True", fontsize=18)
    plt.savefig(f'confusion_matrix_{model_name}.png', format='png', dpi=1200, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_roc_curve(y_true, y_probs, model_name):
    x_dim, y_dim = [16, 12]
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(x_dim, y_dim))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f"Curva ROC - {model_name}", fontsize=24)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig(f'ROC_curve_{model_name}.png', format='png', dpi=1200, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_roc_curve_sklearn(model, X_val, y_val, model_name=""):
    x_dim, y_dim = [16, 12]
    y_probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    auc = roc_auc_score(y_val, y_probs)

    plt.figure(figsize=(x_dim, y_dim))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve {model_name} (AUC = {auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'Curva ROC - {model_name}', fontsize=24)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig(f'ROC_curve_{model_name}.png', format='png', dpi=1200, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_metrics_comparison(metrics_dict, model_names):
    x_dim, y_dim = [16, 12]
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']

    fig, ax = plt.subplots(figsize=(x_dim, y_dim))

    bar_width = 0.35
    index = np.arange(len(metrics))

    for i, model_name in enumerate(model_names):
        results = [metrics_dict[model_name][metric] for metric in metrics]
        ax.bar(index + i * bar_width, results, bar_width, label=model_name)

    ax.set_xlabel('Metriche', fontsize=18)
    ax.set_title('Confronto delle Metriche tra i Modelli', fontsize=24)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics, fontsize=16)
    ax.legend(fontsize=16)
    plt.savefig('metrics_comparison.png', format='png', dpi=1200, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_regularization_effect(X_train, y_train, lambdas, regularization_type='ridge'):
    x_dim, y_dim = [20, 12]
    coefficients = []

    for _lambda in lambdas:
        model = LogisticRegressionGD(learning_rate=0.01, n_iterations=1000, lambda_=_lambda,
                                     regularization=regularization_type)
        model.fit(X_train, y_train)
        coefficients.append(model.theta)

    coefficients = np.array(coefficients)

    plt.figure(figsize=(x_dim, y_dim))
    for i in range(coefficients.shape[1]):
        plt.plot(lambdas, coefficients[:, i], label=f'Feature {i}', lw=2)

    plt.xscale('log')
    plt.title(f"Effetto della Regolarizzazione {regularization_type.capitalize()} sui Coefficienti", fontsize=24)
    plt.xlabel("Lambda", fontsize=18)
    plt.ylabel("Coefficients", fontsize=18)
    plt.legend(loc='best', fontsize=16)
    plt.savefig('regularization_effect.png', format='png', dpi=1200, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()
