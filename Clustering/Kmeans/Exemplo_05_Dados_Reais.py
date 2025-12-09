"""
Exemplo 04 - Aplicação em dados reais

Descrição:
Este exemplo aplica o algoritmo KMeans no dataset Iris, com pré-processamento básico.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Carregamento do dataset Iris
data = load_iris()
X = data.data
y = data.target

# Pré-processamento: Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicação do KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Visualização dos clusters (usando as duas primeiras features)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('KMeans - Clusters no Dataset Iris')
plt.xlabel('Feature 1 (normalizada)')
plt.ylabel('Feature 2 (normalizada)')
plt.show()