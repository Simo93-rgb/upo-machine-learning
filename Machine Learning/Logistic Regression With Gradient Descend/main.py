from funzioni import *
from enum import Enum
import pandas as pd
from valutazione import *
from ModelName import ModelName

if __name__ == "__main__":
    start_time = time.time()
    plotting = True
    # Carica e pre-processa i dati
    X, y = carica_dati()
    X_normalized, features_eliminate, y_encoded = preprocessa_dati(X, y, class_balancer="", corr=0.95)

    # Ottieni i nomi delle feature
    all_feature_names = X.columns
    remaining_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if
                               i not in features_eliminate]
    print(remaining_feature_names)

    # Suddivisione in train (80%), test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.8, random_state=42)

    # Caricamento iperparametri
    best_params, best_score = load_best_params(X_train, y_train,
                                               scorer=make_scorer(greater_is_better=True, score_func=f1_score))

    print(f"Migliori iperparametri trovati: {best_params}")

    # Cross-validation
    k = 10
    k_fold_metrics, k_fold_sk_metrics = k_fold_cross_validation(X_train, y_train, ModelName, k=k)
    stampa_metriche_ordinate(k_fold_metrics, k_fold_sk_metrics, file_name="k_fold_metriche_modelli_parametri_base")

    plot_learning_curve_with_kfold(
        model=LogisticRegressionGD(n_iterations=1000),
        X=X_normalized,
        y=y_encoded,
        cv=k,
        model_name=ModelName.LOGISTIC_REGRESSION_GD.value
    )
    plot_learning_curve_with_kfold(
        model=LogisticRegression(max_iter=1000),
        X=X_normalized,
        y=y_encoded,
        cv=k,
        model_name=ModelName.SCIKIT_LEARN.value
    )

    # Addestramento del modello
    start_model_time = time.time()
    model, sk_model = addestra_modelli(X_train, y_train, **best_params)
    end_model_time = time.time()
    print(
        f"\nTempo di esecuzione del modello {ModelName.LOGISTIC_REGRESSION_GD.value}: {end_model_time - start_model_time:.4f} secondi")

    # Valutazione finale
    print("\nValutazione finale sul Test Set:")
    test_predictions = model.predict(X_test)
    scores = evaluate_model(
        predictions=test_predictions,
        X_val=X_test,
        y_val=y_test,
        model=model,
        model_name=f"Modello {ModelName.LOGISTIC_REGRESSION_GD.value}"
    )

    test_sk_predictions = sk_model.predict(X_test)
    sk_scores = evaluate_model(
        predictions=test_sk_predictions,
        X_val=X_test,
        y_val=y_test,
        model=sk_model,
        model_name=f"Modello {ModelName.SCIKIT_LEARN.value}"
    )

    print(
        f"\nTempo di esecuzione del modello {ModelName.SCIKIT_LEARN.value}: {end_model_time - start_model_time:.4f} secondi"
    )

    stampa_metriche_ordinate(scores, sk_scores, save_to_file=True)

    # Plotting
    if plotting:
        plot_graphs(X_train, y_train, y_test, test_predictions, test_sk_predictions, ModelName, remaining_feature_names)
        plot_results(X_test, y_test, model, sk_model, test_predictions, test_sk_predictions, scores["auc"],
                     sk_scores["auc"], ModelName)


    end_time = time.time()
    print(f"\nTempo di esecuzione totale: {end_time - start_time:.4f} secondi")
