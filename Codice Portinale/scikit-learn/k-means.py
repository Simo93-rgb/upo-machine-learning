# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:43:45 2020

@author: Luigi Portinale
"""

print(__doc__)

# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, centers=2, random_state=random_state)

# correct number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("k-means clusters=2")

X, y = make_blobs(n_samples=n_samples, random_state=random_state) #default centers=3

# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

# correct number of clusters
X, y = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("k-means clusters=3")

# correct number of clusters
X, y = make_blobs(n_samples=n_samples, centers=3) # random state different than previous, means different data 
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("k-means clusters=3")