from collections import Counter
from random import shuffle

import numpy as np
from imblearn.under_sampling import RandomUnderSampler

# Dataset di prova
X_test = np.random.rand(100, 5)  # 100 righe, 5 feature casuali
y_test = np.array([0] * 80 + [1] * 20)  # Classe 0 = 80 campioni, Classe 1 = 20 campioni

# Applica il RandomUnderSampler
resampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = resampler.fit_resample(X_test, y_test)

print(f"Distribuzione originale: {Counter(y_test)}")
print(f"Distribuzione dopo undersampling: {Counter(y_resampled)}")
print(y_resampled)
shuffle(y_resampled)
print(y_resampled)
print(f"Distribuzione dopo undersampling e shuffle: {Counter(y_resampled)}")
