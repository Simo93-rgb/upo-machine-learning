from funzioni import *
from enum import Enum
import pandas as pd


class ModelName(Enum):
    LOGISTIC_REGRESSION_GD = "LogisticRegressionGD"
    SCIKIT_LEARN = "Scikit-learn"


if __name__ == "__main__":
    start_time = time.time()
    plotting = True
    # Carica e pre-processa i dati
    X, y = carica_dati()
    X_normalized, features_eliminate, y_encoded = preprocessa_dati(X, y, class_balancer="SMOTE", corr=0.9)

    # Ottieni i nomi delle feature
    all_feature_names = X.columns
    remaining_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if
                               i not in features_eliminate]
    print(remaining_feature_names)

    # Suddivisione in train (80%), test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.8, random_state=42)

    # Caricamento iperparametri
    best_params, best_score = load_best_params(X_train, y_train)
    print(f"Migliori iperparametri trovati: {best_params}")

    # Cross-validation
    k = 10
    k_fold_accuracy = k_fold_cross_validation(X_train, y_train, k=k)
    print(f"Accuratezza con k-fold (k={k}) Cross-Validation: {k_fold_accuracy}")

    model = LogisticRegressionGD(
        learning_rate=best_params["learning_rate"],
        lambda_=best_params["lambda_"],
        n_iterations=best_params["n_iterations"],
        regularization=best_params["regularization"]
    )
    plot_learning_curve_with_kfold(model, X_normalized, y_encoded, cv=k,
                                   model_name=ModelName.LOGISTIC_REGRESSION_GD.value)

    # Addestramento del modello
    start_model_time = time.time()
    model, sk_model = addestra_modelli(X_train, y_train, **best_params)
    end_model_time = time.time()
    print(
        f"\nTempo di esecuzione del modello {ModelName.LOGISTIC_REGRESSION_GD.value}: {end_model_time - start_model_time:.4f} secondi")

    # Valutazione finale
    print("\nValutazione finale sul Test Set:")
    test_predictions = model.predict(X_test)
    scores = validation_test(test_predictions, X_test, y_test, model,
                             model_name=f"Modello {ModelName.LOGISTIC_REGRESSION_GD.value}")

    test_sk_predictions = sk_model.predict(X_test)
    sk_scores = validation_test(test_sk_predictions, X_test, y_test, sk_model,
                                model_name=f"Modello {ModelName.SCIKIT_LEARN.value}")

    print(
        f"\nTempo di esecuzione del modello {ModelName.SCIKIT_LEARN.value}: {end_model_time - start_model_time:.4f} secondi")

    stampa_metriche_ordinate(scores, sk_scores)

    # Plotting
    if plotting:
        plot_graphs(X_train, y_train, y_test, test_predictions, test_sk_predictions, ModelName, remaining_feature_names)
        plot_results(X_test, y_test, model, sk_model, test_predictions, test_sk_predictions, scores["auc"],
                     sk_scores["auc"], ModelName)

    end_time = time.time()
    print(f"\nTempo di esecuzione totale: {end_time - start_time:.4f} secondi")
