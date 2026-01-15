"""
Exemplo 09 - Inicialização dos Centroides: Custom

Descrição:
Este exemplo demonstra como usar uma inicialização personalizada para os centroides no algoritmo KMeans.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Geração de dados sintéticos
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Definição de centroides iniciais personalizados
custom_centers = np.array([[0, 0], [5, 5], [-5, -5], [3, -3]])

# Aplicação do KMeans com inicialização customizada
kmeans = KMeans(n_clusters=4, init=custom_centers, n_init=1, random_state=42)
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
plt.title("KMeans - Inicialização Customizada")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
