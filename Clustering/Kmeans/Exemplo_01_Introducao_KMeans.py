"""
Exemplo 01 - Introdução ao Algoritmo KMeans

Este script demonstra a aplicação básica do algoritmo K-Means utilizando dados
sintéticos gerados pela função make_blobs. Ele ilustra o processo completo:

1. Geração de dados com clusters naturais.
2. Treinamento do modelo KMeans.
3. Extração dos centróides e rótulos dos clusters.
4. Visualização dos agrupamentos e centróides finais.

Requisitos:
- numpy
- matplotlib
- scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# ---------------------------------------------------------------------------
# Função 1: Geração de dados
# ---------------------------------------------------------------------------
def gerar_dados(n_samples=300, centers=4, cluster_std=0.6, random_state=42):
    """
    Gera dados sintéticos utilizando make_blobs.

    Args:
        n_samples (int): Número de pontos a serem gerados.
        centers (int): Quantidade de clusters naturais.
        cluster_std (float): Dispersão dos pontos em torno dos centróides.
        random_state (int): Semente de reprodutibilidade.

    Returns:
        tuple: (X, y) onde:
            X (ndarray): matriz com coordenadas dos pontos.
            y (ndarray): rótulos verdadeiros dos blobs (não usados pelo KMeans).
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X, y


# ---------------------------------------------------------------------------
# Função 2: Treinar o modelo KMeans
# ---------------------------------------------------------------------------
def treinar_kmeans(X, n_clusters=4, random_state=42):
    """
    Ajusta o modelo KMeans aos dados fornecidos.

    Args:
        X (ndarray): Dados de entrada.
        n_clusters (int): Número de clusters desejados.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (kmeans, labels, centers) onde:
            kmeans (KMeans): modelo treinado.
            labels (ndarray): rótulos previstos para cada ponto.
            centers (ndarray): coordenadas dos centróides finais.
    """
    modelo = KMeans(n_clusters=n_clusters, random_state=random_state)
    modelo.fit(X)

    labels = modelo.labels_
    centers = modelo.cluster_centers_

    return modelo, labels, centers


# ---------------------------------------------------------------------------
# Função 3: Plotar clusters e centróides
# ---------------------------------------------------------------------------
def plotar_clusters(X, labels, centers):
    """
    Plota os pontos clusterizados e os centróides calculados.

    Args:
        X (ndarray): Dados 2D.
        labels (ndarray): Rótulos dos clusters atribuídos pelo modelo.
        centers (ndarray): Coordenadas dos centróides.

    Returns:
        None
    """
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

    plt.title("KMeans - Clusters e Centroids")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Função principal (pipeline)
# ---------------------------------------------------------------------------
def executar_kmeans():
    """
    Executa o pipeline completo:
    - Gera dados sintéticos
    - Treina o modelo KMeans
    - Plota clusters e centróides
    """
    X, y = gerar_dados()
    modelo, labels, centers = treinar_kmeans(X)
    plotar_clusters(X, labels, centers)


# Execução direta ------------------------------------------------------------
if __name__ == "__main__":
    executar_kmeans()
