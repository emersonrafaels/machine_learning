from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


@dataclass(frozen=True)
class KMeansConfig:
    """
    Configurações para o modelo KMeans.
    """

    k_min: int = 2
    k_max: int = 10
    random_state: int = 42
    n_init: int = 20
    max_iter: int = 300


def create_kmeans_model(n_clusters: int, config: KMeansConfig) -> KMeans:
    """
    Cria e retorna um modelo KMeans com base nas configurações fornecidas.

    Args:
        n_clusters (int): Número de clusters.
        config (KMeansConfig): Configurações do modelo.

    Returns:
        KMeans: Modelo KMeans configurado.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        n_init=config.n_init,
        max_iter=config.max_iter,
    )

    return model


def find_optimal_clusters(data, min_clusters=2, max_clusters=10):
    """
    Detecta automaticamente o número ideal de clusters baseado no silhouette score.

    Args:
        data (np.ndarray or pd.DataFrame): Dados para clustering.
        min_clusters (int): Número mínimo de clusters a testar.
        max_clusters (int): Número máximo de clusters a testar.

    Returns:
        int: Número ideal de clusters.
    """
    best_score = -1
    best_k = min_clusters

    for k in range(min_clusters, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k
