"""
Exemplo 05 - Aplicação em dados reais (Dataset Iris)

Descrição:
Este exemplo aplica o algoritmo KMeans no dataset Iris, incluindo:
1. Carregamento dos dados reais.
2. Pré-processamento com normalização (StandardScaler).
3. Treinamento do KMeans.
4. Visualização dos clusters utilizando as duas primeiras features normalizadas.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Função 1: Carregar o dataset Iris
# ---------------------------------------------------------------------------
def carregar_dataset_iris():
    """
    Carrega o dataset Iris a partir do scikit-learn.

    Returns:
        tuple: (X, y, feature_names)
            - X (ndarray): dados numéricos das flores.
            - y (ndarray): rótulos verdadeiros das espécies.
            - feature_names (list): nomes das features originais.
    """
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Função 2: Normalizar os dados com StandardScaler
# ---------------------------------------------------------------------------
def normalizar_dados(X):
    """
    Normaliza os dados usando StandardScaler.

    Args:
        X (ndarray): Matriz de dados originais.

    Returns:
        ndarray: Dados normalizados.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# ---------------------------------------------------------------------------
# Função 3: Treinar o KMeans
# ---------------------------------------------------------------------------
def treinar_kmeans(X_scaled, n_clusters=3, random_state=42):
    """
    Treina o algoritmo KMeans com os dados normalizados.

    Args:
        X_scaled (ndarray): Dados normalizados.
        n_clusters (int): Número de clusters desejado.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (kmeans, labels)
            - kmeans (KMeans): modelo treinado.
            - labels (ndarray): rótulos atribuídos a cada ponto.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    return kmeans, labels


# ---------------------------------------------------------------------------
# Função 4: Plotar os clusters usando as duas primeiras features
# ---------------------------------------------------------------------------
def plotar_clusters_iris(X_scaled, labels):
    """
    Plota os clusters formados pelo KMeans utilizando as duas primeiras features
    normalizadas do dataset Iris.

    Args:
        X_scaled (ndarray): Dados normalizados.
        labels (ndarray): Rótulos atribuídos pelo KMeans.

    Returns:
        None
    """
    plt.scatter(
        X_scaled[:, 0], X_scaled[:, 1],
        c=labels,
        cmap='viridis',
        s=50,
        alpha=0.6
    )

    plt.title('KMeans - Clusters no Dataset Iris')
    plt.xlabel('Feature 1 (normalizada)')
    plt.ylabel('Feature 2 (normalizada)')
    plt.show()


# ---------------------------------------------------------------------------
# Função principal do pipeline
# ---------------------------------------------------------------------------
def executar_kmeans_iris():
    """
    Executa o pipeline completo:
    1. Carrega o dataset Iris.
    2. Normaliza os dados.
    3. Treina o modelo KMeans.
    4. Plota os clusters utilizando duas features.
    """
    X, y, feature_names = carregar_dataset_iris()
    X_scaled = normalizar_dados(X)
    kmeans, labels = treinar_kmeans(X_scaled)
    plotar_clusters_iris(X_scaled, labels)


# Execução direta ----------------------------------------------------------------
if __name__ == "__main__":
    executar_kmeans_iris()
