from dataclasses import dataclass
from sklearn.cluster import KMeans
import mlflow


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

    # Logando as configurações do modelo no MLflow
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("random_state", config.random_state)
    mlflow.log_param("n_init", config.n_init)
    mlflow.log_param("max_iter", config.max_iter)

    return model
