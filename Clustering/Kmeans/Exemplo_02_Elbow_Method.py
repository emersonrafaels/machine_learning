"""
Exemplo 02 - Escolha do número de clusters (Método do Cotovelo)

Descrição:
Este exemplo demonstra como usar o Método do Cotovelo para determinar o número
ideal de clusters para o algoritmo KMeans.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# ---------------------------------------------------------------------------
# Função 1: Geração de dados sintéticos
# ---------------------------------------------------------------------------
def gerar_dados(n_samples=300, centers=5, cluster_std=0.7, random_state=42):
    """
    Gera dados sintéticos utilizando a função make_blobs.

    Args:
        n_samples (int): Quantidade de pontos a serem gerados.
        centers (int): Número de centros (clusters naturais).
        cluster_std (float): Dispersão dos pontos em torno dos centros.
        random_state (int): Semente para garantir reprodutibilidade.

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
# Função 2: Cálculo da inércia para diferentes valores de k
# ---------------------------------------------------------------------------
def calcular_inercias(X, k_min=1, k_max=11, random_state=42):
    """
    Calcula a inércia do modelo KMeans para vários valores de k,
    permitindo visualizar o Método do Cotovelo.

    Args:
        X (ndarray): Conjunto de dados.
        k_min (int): Valor mínimo de clusters para testar.
        k_max (int): Valor máximo (exclusivo) de clusters para testar.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (lista_k, lista_inercia)
            lista_k (list): Valores de k testados.
            lista_inercia (list): Valores de inércia correspondentes.
    """
    lista_inercia = []
    lista_k = list(range(k_min, k_max))

    for k in lista_k:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        lista_inercia.append(kmeans.inertia_)

    return lista_k, lista_inercia


# ---------------------------------------------------------------------------
# Função 3: Plot do Método do Cotovelo
# ---------------------------------------------------------------------------
def plotar_metodo_cotovelo(lista_k, lista_inercia):
    """
    Plota o gráfico do Método do Cotovelo, relacionando o número de clusters (k)
    com a inércia do modelo KMeans.

    Args:
        lista_k (list): Lista com valores de k.
        lista_inercia (list): Lista com valores de inércia.

    Returns:
        None
    """
    plt.plot(lista_k, lista_inercia, marker='o')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.xticks(lista_k)
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Função principal do pipeline
# ---------------------------------------------------------------------------
def executar_metodo_cotovelo():
    """
    Executa todo o pipeline:
    1. Geração dos dados
    2. Cálculo da inércia para vários valores de k
    3. Plotagem do Método do Cotovelo
    """
    X = gerar_dados()
    lista_k, lista_inercia = calcular_inercias(X)
    plotar_metodo_cotovelo(lista_k, lista_inercia)


# Execução direta --------------------------------------------------------------------
if __name__ == "__main__":
    executar_metodo_cotovelo()
