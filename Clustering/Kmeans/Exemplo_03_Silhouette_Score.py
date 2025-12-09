"""
Exemplo 03 - Avaliação de clusters (Silhouette Score)

Descrição:
Este exemplo demonstra como usar o Silhouette Score para avaliar a qualidade
dos clusters gerados pelo algoritmo KMeans.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Função 1: Geração de dados sintéticos
# ---------------------------------------------------------------------------
def gerar_dados(n_samples=300, centers=4, cluster_std=0.6, random_state=42):
    """
    Gera dados sintéticos utilizando make_blobs.

    Args:
        n_samples (int): Quantidade de pontos a serem gerados.
        centers (int): Número de agrupamentos naturais.
        cluster_std (float): Dispersão dos pontos ao redor dos centróides.
        random_state (int): Semente para reprodutibilidade dos resultados.

    Returns:
        ndarray: Matriz X contendo os pontos gerados.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X


# ---------------------------------------------------------------------------
# Função 2: Treinar o modelo KMeans
# ---------------------------------------------------------------------------
def treinar_kmeans(X, n_clusters=4, random_state=42):
    """
    Treina o algoritmo KMeans sobre os dados fornecidos.

    Args:
        X (ndarray): Conjunto de dados.
        n_clusters (int): Número de clusters desejado.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (kmeans, labels, centers)
            - kmeans (KMeans): modelo treinado
            - labels (ndarray): rótulos atribuídos a cada ponto
            - centers (ndarray): coordenadas dos centróides
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return kmeans, labels, centers


# ---------------------------------------------------------------------------
# Função 3: Calcular Silhouette Score
# ---------------------------------------------------------------------------
def calcular_silhouette(X, labels):
    """
    Calcula o Silhouette Score para avaliar a qualidade dos clusters.

    Args:
        X (ndarray): Dados utilizados no clustering.
        labels (ndarray): Rótulos atribuídos pelo modelo.

    Returns:
        float: valor do Silhouette Score.
    """
    return silhouette_score(X, labels)


# ---------------------------------------------------------------------------
# Função 4: Plotar os clusters resultantes
# ---------------------------------------------------------------------------
def plotar_clusters(X, labels, centers):
    """
    Plota os clusters formados pelo KMeans e seus centróides.

    Args:
        X (ndarray): Dados bidimensionais.
        labels (ndarray): Rótulos atribuídos pelo modelo.
        centers (ndarray): Coordenadas dos centróides.

    Returns:
        None
    """
    plt.scatter(
        X[:, 0], X[:, 1],
        c=labels,
        cmap='viridis',
        s=50,
        alpha=0.6
    )

    plt.scatter(
        centers[:, 0], centers[:, 1],
        c='red',
        s=200,
        alpha=0.75,
        marker='X',
        label='Centroids'
    )

    plt.title('KMeans - Clusters e Centroides')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Função principal do pipeline
# ---------------------------------------------------------------------------
def executar_avaliacao_silhouette():
    """
    Executa o pipeline completo:
    1. Geração dos dados
    2. Treinamento do KMeans
    3. Cálculo do Silhouette Score
    4. Plot dos clusters
    """
    X = gerar_dados()
    kmeans, labels, centers = treinar_kmeans(X)

    sil_score = calcular_silhouette(X, labels)
    print(f"Silhouette Score: {sil_score:.2f}")

    plotar_clusters(X, labels, centers)


# Execução direta ------------------------------------------------------------
if __name__ == "__main__":
    executar_avaliacao_silhouette()