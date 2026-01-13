"""
Exemplo 06 - Avaliação de clusters com Silhouette Score e Silhouette Samples

Descrição:
Este exemplo demonstra como calcular:
1. O Silhouette Score geral do modelo.
2. O Silhouette Score individual de cada ponto (Silhouette Samples).

O algoritmo utilizado é o KMeans com 3 clusters, aplicado sobre um conjunto
de dados simples definido manualmente.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


# ---------------------------------------------------------------------------
# Função 1: Criar o dataset manual
# ---------------------------------------------------------------------------
def gerar_dados_manualmente():
    """
    Cria um conjunto de dados 2D simples contendo 3 grupos bem separados.

    Returns:
        ndarray: Matriz X contendo os pontos.
    """
    X = np.array([
        [1, 2], [2, 1], [1, 3],        # Cluster 1
        [8, 8], [9, 7], [8, 9],        # Cluster 2
        [15, 2], [16, 3], [14, 1]      # Cluster 3
    ])
    return X


# ---------------------------------------------------------------------------
# Função 2: Treinar KMeans
# ---------------------------------------------------------------------------
def treinar_kmeans(X, n_clusters=3, random_state=42):
    """
    Treina o modelo KMeans e retorna os rótulos previstos.

    Args:
        X (ndarray): Dados de entrada.
        n_clusters (int): Número de clusters.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: (kmeans, labels)
            - kmeans: modelo treinado.
            - labels: rótulos atribuídos a cada ponto.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


# ---------------------------------------------------------------------------
# Função 3: Calcular Silhouette Score geral e por ponto
# ---------------------------------------------------------------------------
def calcular_silhouette(X, labels):
    """
    Calcula o Silhouette Score geral e o Silhouette Sample (por ponto).

    Args:
        X (ndarray): Dados de entrada.
        labels (ndarray): Rótulos atribuídos pelo modelo.

    Returns:
        tuple: (score_geral, scores_ponto)
            - score_geral (float): Silhouette Score geral.
            - scores_ponto (ndarray): Silhouette Score individual.
    """
    score_geral = silhouette_score(X, labels)
    scores_ponto = silhouette_samples(X, labels)
    return score_geral, scores_ponto


# ---------------------------------------------------------------------------
# Função 4: Exibir resultados
# ---------------------------------------------------------------------------
def exibir_resultados(score_geral, scores_ponto):
    """
    Imprime o Silhouette geral e o Silhouette de cada ponto.

    Args:
        score_geral (float): Silhouette Score geral.
        scores_ponto (ndarray): Valores por ponto.

    Returns:
        None
    """
    print(f"Silhouette Score geral: {score_geral:.4f}")
    print("Silhouette Score por ponto:")
    for i, score in enumerate(scores_ponto):
        print(f"  Ponto {i}: {score:.4f}")


# ---------------------------------------------------------------------------
# Função principal do pipeline
# ---------------------------------------------------------------------------
def executar_exemplo_silhouette_samples():
    """
    Executa o pipeline completo:
    1. Geração de dados manuais.
    2. Treinamento do KMeans.
    3. Cálculo dos Silhouette Scores.
    4. Impressão dos resultados.
    """
    X = gerar_dados_manualmente()
    kmeans, labels = treinar_kmeans(X)
    score_geral, scores_ponto = calcular_silhouette(X, labels)
    exibir_resultados(score_geral, scores_ponto)


# Execução direta ----------------------------------------------------------------------
if __name__ == "__main__":
    executar_exemplo_silhouette_samples()
