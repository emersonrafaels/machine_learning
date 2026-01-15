"""
Exemplo 07 - Inicialização dos Centroides: k-means++

Descrição:
Este exemplo demonstra o uso do método de inicialização "k-means++" no algoritmo KMeans.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Geração de dados sintéticos
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Aplicação do KMeans com inicialização k-means++
kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
kmeans.fit(X)

# Previsões e centros dos clusters
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Visualização dos clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.6)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    s=200,
    alpha=0.75,
    marker="X",
    label="Centroids",
)
plt.title("KMeans - Inicialização k-means++")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
