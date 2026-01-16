from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from src.utils.visualization.kmeans_visualization import (
    plot_elbow_method,
    plot_silhouette_scores,
)


@dataclass(frozen=True)
class KMeansConfig:
    """
    Configurações para o modelo KMeans.
    """

    k_min: int = 2  # Número mínimo de clusters a testar
    k_max: int = 10  # Número máximo de clusters a testar
    random_state: int = 42  # Semente para reprodutibilidade
    n_init: int = 20  # Número de inicializações do algoritmo
    max_iter: int = 300  # Número máximo de iterações
    tol: float = 1e-4  # Tolerância para critério de convergência
    algorithm: str = "lloyd"  # Algoritmo KMeans ("lloyd", "elkan", etc.)
    init: str = "k-means++"  # Método de inicialização ("k-means++" ou "random")


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
        tol=config.tol,
        algorithm=config.algorithm,
        init=config.init,
    )

    return model


def find_optimal_clusters(
    data, min_clusters=2, max_clusters=10, method="silhouette", plot=False
):
    """
    Detecta automaticamente o número ideal de clusters baseado no método especificado.

    Métodos suportados:
      - "silhouette": Usa o coeficiente de silhueta para determinar o número ideal de clusters.
      - "elbow": Usa a soma dos erros quadráticos (inertia) para determinar o "cotovelo".

    Args:
        data (np.ndarray or pd.DataFrame): Dados para clustering.
        min_clusters (int): Número mínimo de clusters a testar.
        max_clusters (int): Número máximo de clusters a testar.
        method (str): Método para determinar o número ideal de clusters ("silhouette" ou "elbow").
        plot (bool): Se True, gera o gráfico correspondente ao método escolhido.

    Returns:
        int: Número ideal de clusters.

    Raises:
        ValueError: Se o método especificado não for suportado.
    """
    if method not in ["silhouette", "elbow"]:
        raise ValueError("Método não suportado. Use 'silhouette' ou 'elbow'.")

    if method == "silhouette":
        best_score = -1
        best_k = min_clusters
        silhouette_scores = []
        k_range = list(range(min_clusters, max_clusters + 1))

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                best_k = k

        if plot:
            plot_silhouette_scores(silhouette_scores, k_range)

        return best_k

    elif method == "elbow":
        inertia_values = []
        k_range = list(range(min_clusters, max_clusters + 1))

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(data)
            inertia_values.append(model.inertia_)

        # Detecta o "cotovelo" usando a diferença relativa da inércia
        deltas = [
            inertia_values[i] - inertia_values[i + 1]
            for i in range(len(inertia_values) - 1)
        ]
        best_k = deltas.index(max(deltas)) + min_clusters

        if plot:
            plot_elbow_method(inertia_values, k_range)

        return best_k
