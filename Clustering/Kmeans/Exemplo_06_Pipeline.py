"""
Exemplo 06 - Otimização e pipeline

Descrição:
Este exemplo demonstra como criar um pipeline de Machine Learning com pré-processamento e KMeans.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Geração de dados sintéticos
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Criação do pipeline
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=4, random_state=42))]
)

# Treinamento do pipeline
pipeline.fit(X)
labels = pipeline["kmeans"].labels_

# Avaliação com Silhouette Score
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.2f}")

# Visualização dos clusters
centers = pipeline["kmeans"].cluster_centers_
import matplotlib.pyplot as plt

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
plt.title("Pipeline - KMeans com Pré-processamento")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
