import numpy
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc, precision_score, \
    f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

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
    plt.savefig('Assets/sigmoid_function.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_corr_matrix(corr_matrix: numpy.ndarray, features_eliminate=None):
    x_dim, y_dim = [20, 16]

    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12})  # Font migliorato
    plt.title("Matrice di Correlazione delle Feature", fontsize=24)

    # Evidenzia le feature eliminate
    if features_eliminate:
        for feature in features_eliminate:
            plt.gca().add_patch(plt.Rectangle((feature, feature), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.savefig('Assets/correlation_matrix_breast_cancer.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_class_distribution(y, file_name=""):
    x_dim, y_dim = [16, 12]

    # Conta il numero di campioni per ciascuna classe
    unique, counts = np.unique(y, return_counts=True)

    # Calcola le percentuali
    total = np.sum(counts)
    percentages = [(count / total) * 100 for count in counts]

    # Etichette con la percentuale e il numero di elementi
    labels = [f'B (Benigno) - {counts[0]} ({percentages[0]:.1f}%)',
              f'M (Maligno) - {counts[1]} ({percentages[1]:.1f}%)']

    # Crea un grafico a torta
    plt.figure(figsize=(x_dim, y_dim))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'orange'], startangle=90,
            textprops={'fontsize': 27})  # Font ingrandito del 50% (18 * 1.5 = 27)

    plt.title('Distribuzione delle Classi', fontsize=36)  # Font titolo ingrandito del 50% (24 * 1.5 = 36)
    plt.axis('equal')  # Assicura che il grafico sia disegnato come un cerchio

    # Salva l'immagine, se specificato
    if file_name:
        plt.savefig(f'Assets/{file_name}.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata

    # Mostra il grafico
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    x_dim, y_dim = [16, 12]
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16})  # Font migliorato
    plt.title(f"Matrice di Confusione - {model_name}", fontsize=24)
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("True", fontsize=18)
    plt.savefig(f'Assets/confusion_matrix_{model_name}.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
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
    plt.savefig(f'Assets/ROC_curve_{model_name}.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
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
    plt.savefig(f'Assets/ROC_curve_{model_name}.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_metrics_comparison(metrics_dict, model_names):
    x_dim, y_dim = [16, 12]
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']

    fig, ax = plt.subplots(figsize=(x_dim, y_dim))

    bar_width = 0.35
    index = np.arange(len(metrics))

    # Trova il minimo e il massimo delle metriche per adattare i limiti dell'asse y
    all_results = [metrics_dict[model_name][metric] for model_name in model_names for metric in metrics]
    min_metric = min(all_results)
    max_metric = max(all_results)

    # Definisci i limiti dell'asse y per fare zoom solo sulla parte alta delle barre
    zoom_factor = 0.1  # Puoi regolare questo fattore per aumentare/diminuire lo zoom
    y_min = min_metric - zoom_factor * (max_metric - min_metric)
    y_max = max_metric + zoom_factor * (max_metric - min_metric)

    for i, model_name in enumerate(model_names):
        results = [metrics_dict[model_name][metric] for metric in metrics]
        ax.bar(index + i * bar_width, results, bar_width, label=model_name)

    ax.set_xlabel('Metriche', fontsize=18)
    ax.set_title('Confronto delle Metriche tra i Modelli', fontsize=24)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics, fontsize=16)
    ax.legend(fontsize=16)

    # Imposta i limiti dell'asse y per il "zoom"
    ax.set_ylim([y_min, y_max])

    plt.savefig('Assets/metrics_comparison.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()



def plot_regularization_effect(X_train, y_train, feature_names, lambdas, regularization_type='ridge'):
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
        plt.plot(lambdas, coefficients[:, i], label=f'{feature_names[i]}', lw=2)

    plt.xscale('log')
    plt.title(f"Effetto della Regolarizzazione {regularization_type.capitalize()} sui Coefficienti", fontsize=24)
    plt.xlabel("Lambda", fontsize=18)
    plt.ylabel("Coefficients", fontsize=18)
    plt.legend(loc='best', fontsize=16)
    plt.savefig(f'Assets/regularization_effect_{regularization_type}.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_precision_recall(y_true, y_scores, model_name=""):
    # Calcola precisione, recall e soglie
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Calcola l'AUC (Area Under the Curve) per la curva Precision-Recall
    auc_score = auc(recalls, precisions)

    # Crea il plot
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, label=f'AUC = {auc_score:.2f}', linewidth=2)

    # Impostazioni del grafico
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('Curva Precision-Recall', fontsize=20)
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True)

    # Salva il grafico, se richiesto
    if model_name:
        plt.savefig(f'Assets/precision_vs_recall_{model_name}.png', format='png', dpi=600, bbox_inches='tight')

    # Mostra il grafico
    plt.show()


def plot_learning_curve_with_kfold(model, X, y, cv=5, model_name=""):
    fig_size = (16, 12)
    # Frazioni del training set da usare (es. dal 10% al 100%)
    train_sizes = np.linspace(0.1, 1.0, 10)

    # Calcola learning curve con K-Fold CV
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='neg_log_loss', n_jobs=-1
    )

    # Calcola la media e deviazione standard delle performance
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot della learning curve
    plt.figure(figsize=fig_size)
    plt.plot(train_sizes, -train_mean, label="Train Loss", color="blue")
    plt.fill_between(train_sizes, -train_mean - train_std, -train_mean + train_std, alpha=0.1, color="blue")
    plt.plot(train_sizes, -val_mean, label="Validation Loss", color="orange")
    plt.fill_between(train_sizes, -val_mean - val_std, -val_mean + val_std, alpha=0.1, color="orange")

    plt.title(f"Learning Curve {model_name}", fontsize=24)
    plt.xlabel("Training Set Size", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.legend(loc="best", fontsize=16)
    plt.grid(True)
    if model_name:
        plt.savefig(f'Assets/learnin_curve_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_learning_curve_with_loss(estimator, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
                                  scoring='neg_log_loss', model_name=""):
    """
    Funzione per plottare la curva di apprendimento con la loss sui dati di training e validation.

    Parametri:
    - estimator: il modello (es. LogisticRegression)
    - X_train: dataset di training (feature)
    - y_train: dataset di training (target)
    - cv: numero di fold per la cross-validation
    - train_sizes: frazioni di training set da usare per calcolare il punteggio
    - scoring: metrica di valutazione ('neg_log_loss' per la loss)

    Output:
    - Grafico delle curve di apprendimento per il training e il validation set con la loss.
    """
    plt.figure(figsize=(10, 6))

    # Calcola le curve di apprendimento
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X_train, y_train, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=-1
    )

    # Convertiamo i punteggi negativi delle loss in valori positivi (perchè scikit-learn usa score, non loss)
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    # Plottiamo i risultati
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training loss")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Validation loss")

    # Plottiamo anche le aree di deviazione standard
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="r",
                     alpha=0.2)
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, color="g", alpha=0.2)

    plt.title(f"Curva di Apprendimento (Loss) {model_name}")
    plt.xlabel("Numero di campioni di training")
    plt.ylabel("Loss (Log-Loss)")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def plot_results(X_test, y_test, model, sk_model, test_predictions, test_sk_predictions, auc, sk_auc, model_enum):

    # Curve ROC
    y_probs = model.sigmoid(np.dot(X_test, model.theta) + model.bias)
    plot_roc_curve(y_test, y_probs, f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}")

    y_sk_probs = sk_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_sk_probs, f"Modello {model_enum.SCIKIT_LEARN.value}")

    plot_precision_recall(y_test, test_predictions, model_name=f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}")
    plot_precision_recall(y_test, test_sk_predictions, model_name=f"Modello {model_enum.SCIKIT_LEARN.value}")

    # Confronto delle metriche
    metrics_dict = {
        f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}": {
            "Precision": precision_score(y_test, test_predictions),
            "Recall": recall_score(y_test, test_predictions),
            "F1-Score": f1_score(y_test, test_predictions),
            "AUC": auc
        },
        f"Modello {model_enum.SCIKIT_LEARN.value}": {
            "Precision": precision_score(y_test, test_sk_predictions),
            "Recall": recall_score(y_test, test_sk_predictions),
            "F1-Score": f1_score(y_test, test_sk_predictions),
            "AUC": sk_auc
        }
    }

    plot_metrics_comparison(metrics_dict, [f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}", f"Modello {model_enum.SCIKIT_LEARN.value}"])

def plot_graphs(X_train, y_train, y_test, test_predictions, test_sk_predictions, model_enum, remaining_feature_names):
    # Plottare la funzione sigmoidale
    plot_sigmoid()

    # Matrici di confusione
    plot_confusion_matrix(y_test, test_predictions, f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}")
    plot_confusion_matrix(y_test, test_sk_predictions, f"Modello {model_enum.SCIKIT_LEARN.value}")

    # Plottare l'effetto della regolarizzazione
    lambdas = np.logspace(-4, 2, 100)  # Lista di valori di lambda su scala logaritmica
    plot_regularization_effect(X_train, feature_names=remaining_feature_names, y_train=y_train, lambdas=lambdas,
                               regularization_type='ridge')
    plot_regularization_effect(X_train, feature_names=remaining_feature_names, y_train=y_train, lambdas=lambdas,
                               regularization_type='lasso')
