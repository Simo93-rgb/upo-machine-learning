import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# Unire caratteristiche e target in un unico DataFrame
iris_df = pd.concat([X, y], axis=1)

# Salvare in un file CSV
iris_df.to_csv("iris_dataset.csv", index=False)

print("Dataset salvato come 'iris_dataset.csv'")