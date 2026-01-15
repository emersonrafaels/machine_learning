"""
Exemplo 04 - Escolha automática do número de clusters (Silhouette Score)

Descrição:
Este exemplo demonstra como escolher automaticamente o número de clusters (k)
para o KMeans utilizando o Silhouette Score. O pipeline é:

1. Geração de dados sintéticos.
2. Avaliação de vários valores de k usando Silhouette Score.
3. Seleção do k com melhor score.
4. Treinamento final do KMeans com k ideal.
5. Visualização dos clusters resultantes.
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
        random_state (int): Semente para reprodutibilidade.

    Returns:
        ndarray: Matriz X contendo os pontos gerados.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X


# ---------------------------------------------------------------------------
# Função 2: Avaliar vários k usando Silhouette Score
# ---------------------------------------------------------------------------
def avaliar_k_por_silhouette(X, k_min=2, k_max=11, random_state=42):
    """
    Avalia diferentes valores de k utilizando o Silhouette Score
    e retorna o melhor k encontrado.

    Obs.: k_min começa em 2 porque o Silhouette Score não é definido
    para apenas 1 cluster.

    Args:
        X (ndarray): Conjunto de dados.
        k_min (int): Número mínimo de clusters a testar.
        k_max (int): Número máximo (exclusivo) de clusters a testar.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (melhor_k, dict_scores) onde:
            melhor_k (int): valor de k com maior Silhouette Score.
            dict_scores (dict): mapeia k -> silhouette_score.
    """
    scores = {}

    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"k = {k}, Silhouette Score = {score:.4f}")

    # Escolhe o k com maior score
    melhor_k = max(scores, key=scores.get)
    return melhor_k, scores


# ---------------------------------------------------------------------------
# Função 3: Plotar Silhouette Score por k
# ---------------------------------------------------------------------------
def plotar_silhouette_por_k(scores):
    """
    Plota o Silhouette Score para cada valor de k testado.

    Args:
        scores (dict): Dicionário mapeando k -> silhouette_score.

    Returns:
        None
    """
    ks = list(scores.keys())
    valores = list(scores.values())

    plt.plot(ks, valores, marker="o")
    plt.title("Silhouette Score por número de clusters (k)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(ks)
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Função 4: Treinar KMeans com k ideal e retornar labels e centróides
# ---------------------------------------------------------------------------
def treinar_kmeans_com_k_ideal(X, k, random_state=42):
    """
    Treina o modelo KMeans com o número de clusters escolhido.

    Args:
        X (ndarray): Conjunto de dados.
        k (int): Número de clusters.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (kmeans, labels, centers)
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return kmeans, labels, centers


# ---------------------------------------------------------------------------
# Função 5: Plotar clusters com k ideal
# ---------------------------------------------------------------------------
def plotar_clusters(X, labels, centers, k):
    """
    Plota os clusters encontrados pelo KMeans com k ideal.

    Args:
        X (ndarray): Dados 2D.
        labels (ndarray): Rótulos dos clusters.
        centers (ndarray): Coordenadas dos centróides.
        k (int): Número de clusters usados.

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

    plt.title(f"KMeans com k = {k} (k ideal pelo Silhouette)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Função principal do pipeline
# ---------------------------------------------------------------------------
def executar_kmeans_com_k_automatico():
    """
    Pipeline completo:
    1. Gera dados sintéticos.
    2. Avalia vários valores de k com Silhouette Score.
    3. Escolhe automaticamente o k ideal.
    4. Treina o KMeans com esse k.
    5. Plota o Silhouette por k.
    6. Plota os clusters finais.
    """
    # 1. Geração dos dados
    X = gerar_dados()

    # 2. Avaliar k por Silhouette Score
    melhor_k, scores = avaliar_k_por_silhouette(X)

    print(f"\n>>> k ideal encontrado pelo Silhouette Score: {melhor_k}")

    # 3. Plotar o Silhouette Score por k
    plotar_silhouette_por_k(scores)

    # 4. Treinar modelo final com k ideal
    kmeans, labels, centers = treinar_kmeans_com_k_ideal(X, melhor_k)

    # 5. Plotar clusters finais
    plotar_clusters(X, labels, centers, melhor_k)


# Execução direta ------------------------------------------------------------
if __name__ == "__main__":
    executar_kmeans_com_k_automatico()
