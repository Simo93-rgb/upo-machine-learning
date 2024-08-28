import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from logistic_regression_with_gradient_descend import LogisticRegressionGD

if __name__ == "__main__":
    # Fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # Data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # Imputazione dei NaN con la media delle colonne
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Normalizzazione delle feature
    X_normalized = (X_imputed - np.mean(X_imputed, axis=0)) / np.std(X_imputed, axis=0)

    # Encoding delle classi
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split del dataset in training e validazione
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_encoded, test_size=0.3, random_state=42)

    # Inizializza e addestra il modello
    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    # Prevedi i risultati sul set di validazione
    predictions = model.predict(X_val)

    # Modello con scikit-learn
    sk_model = LogisticRegression(max_iter=1000)
    sk_model.fit(X_train, y_train)

    # Prevedi con scikit-learn sul set di validazione
    sk_predictions = sk_model.predict(X_val)

    # Calcolo dell'accuratezza per entrambi i modelli
    accuracy_custom = accuracy_score(y_val, predictions)
    accuracy_sk = accuracy_score(y_val, sk_predictions)

    print(f'Accuracy del modello implementato: {accuracy_custom}')
    print(f'Accuracy del modello Scikit-learn: {accuracy_sk}')
