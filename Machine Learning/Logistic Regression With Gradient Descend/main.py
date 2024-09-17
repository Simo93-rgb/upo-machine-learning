from funzioni import *

if __name__ == "__main__":
    start_time = time.time()

    # Carica e pre-processa i dati
    X, y = carica_dati()
    X_normalized, features_eliminate, y_encoded = preprocessa_dati(X, y, class_balancer="Undersampling")

    # Ottieni i nomi delle feature
    all_feature_names = X.columns

    # Elimina i nomi delle feature corrispondenti agli indici delle feature eliminate
    remaining_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if
                               i not in features_eliminate]

    print(remaining_feature_names)  # Questo sarà l'elenco delle feature rimaste

    # Suddivisione in train (80%), test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.8, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    # caricamento da file degli iperparametri
    best_params, best_score = load_best_params()

    print(f"Migliori iperparametri trovati: {best_params}")
    print(f"Accuracy del modello con iperparametri trovati tramite baesyan optimization: {best_score}")

    # Esegui Leave-One-Out Cross-Validation
    loo_accuracy = leave_one_out_cross_validation(X_train, y_train)
    print(f"Accuratezza con Leave-One-Out Cross-Validation: {loo_accuracy}")

    # Misura il tempo di esecuzione del tuo modello LogisticRegressionGD
    start_model_time = time.time()

    # model, predictions, sk_model, sk_predictions = addestra_modelli(
    #     X_train, y_train, X_val, best_params, k=5)
    model, sk_model = addestra_modelli(X_train, y_train, **best_params)

    end_model_time = time.time()
    print(f"\nTempo di esecuzione del modello Logistic Implementato: {end_model_time - start_model_time:.4f} secondi")

    # Valutazione finale sul Test Set
    print("\nValutazione finale sul Test Set:")

    test_predictions = model.predict(X_test)
    auc = validation_test(test_predictions, X_test, y_test, model, model_name="Modello Logistic Implementato")

    test_sk_predictions = sk_model.predict(X_test)
    sk_auc = validation_test(test_sk_predictions, X_test, y_test, sk_model, model_name="Modello Scikit-learn")

    print(f"\nTempo di esecuzione del modello Scikit-learn: {end_model_time - start_model_time:.4f} secondi")
    plot_learning_curve(model, X_train, y_train, cv=5, scoring="accuracy", model_name="Modello Logistic Implementato")
    plot_learning_curve(sk_model, X_train, y_train, cv=5, scoring="accuracy", model_name="Modello Scikit-learn")
    # Plottare la funzione sigmoidale
    plot_sigmoid()

    # Matrici di confusione
    plot_confusion_matrix(y_test, test_predictions, "Modello Logistic Implementato")
    plot_confusion_matrix(y_test, test_sk_predictions, "Modello Scikit-learn")

    # Curve ROC
    y_probs = model.sigmoid(np.dot(X_test, model.theta) + model.bias)
    plot_roc_curve(y_test, y_probs, "Modello Logistic Implementato")

    y_sk_probs = sk_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_sk_probs, "Modello Scikit-learn")

    plot_precision_recall(y_test, test_predictions, model_name="Modello Logistic Implementato")
    plot_precision_recall(y_test, test_sk_predictions, model_name="Modello Scikit-learn")
    # Confronto delle metriche
    metrics_dict = {
        "Modello Logistic Implementato": {
            "Precision": precision_score(y_test, test_predictions),
            "Recall": recall_score(y_test, test_predictions),
            "F1-Score": f1_score(y_test, test_predictions),
            "AUC": auc
        },
        "Modello Scikit-learn": {
            "Precision": precision_score(y_test, test_sk_predictions),
            "Recall": recall_score(y_test, test_sk_predictions),
            "F1-Score": f1_score(y_test, test_sk_predictions),
            "AUC": sk_auc
        }
    }

    plot_metrics_comparison(metrics_dict, ["Modello Logistic Implementato", "Modello Scikit-learn"])

    # Plottare l'effetto della regolarizzazione
    lambdas = np.logspace(-4, 2, 100)  # Lista di valori di lambda su scala logaritmica
    plot_regularization_effect(X_train, feature_names=remaining_feature_names, y_train=y_train, lambdas=lambdas, regularization_type='ridge')
    plot_regularization_effect(X_train, feature_names=remaining_feature_names, y_train=y_train, lambdas=lambdas, regularization_type='lasso')

    end_time = time.time()
    print(f"\nTempo di esecuzione totale: {end_time - start_time:.4f} secondi")
