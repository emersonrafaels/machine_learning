"""
Exemplo 03 - Avaliação de clusters (Silhouette Score)

Descrição:
Este exemplo demonstra como usar o Silhouette Score para avaliar a qualidade dos clusters gerados pelo KMeans.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Geração de dados sintéticos
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Aplicação do KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Cálculo do Silhouette Score
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.2f}")

# Visualização dos clusters
centers = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title('KMeans - Clusters e Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()