import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tempfile
from itertools import combinations


def plot_elbow_method(inertia_values, k_range):
    """
    Plota o gráfico do método Elbow para determinar o número ideal de clusters.

    Args:
        inertia_values (list): Lista com os valores de inércia para cada número de clusters.
        k_range (list): Intervalo de valores de k (número de clusters).

    Returns:
        matplotlib.pyplot: Objeto do gráfico gerado.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_values, marker="o", linestyle="--", color="b")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia")
    plt.title("Método Elbow")
    plt.grid(True)
    plt.show()

    return plt


def plot_silhouette_scores(silhouette_scores, k_range):
    """
    Plota o gráfico de pontuações de Silhouette para diferentes números de clusters.

    Args:
        silhouette_scores (list): Lista com as pontuações de Silhouette para cada número de clusters.
        k_range (list): Intervalo de valores de k (número de clusters).

    Returns:
        matplotlib.pyplot: Objeto do gráfico gerado.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker="o", linestyle="--", color="g")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Pontuação de Silhouette")
    plt.title("Pontuações de Silhouette por Número de Clusters")
    plt.grid(True)
    plt.show()

    return plt


def plot_clusters(data, labels, centroids=None, view=True):
    """
    Plota os clusters gerados pelo KMeans em duas dimensões.

    Args:
        data (np.ndarray or pd.DataFrame): Dados utilizados para o clustering (deve ter 2 dimensões).
        labels (list or np.ndarray): Rótulos dos clusters para cada ponto.
        centroids (np.ndarray, optional): Coordenadas dos centróides dos clusters. Default é None.
        view (bool): Se True, exibe o gráfico imediatamente. Default é True.

    Returns:
        None
    """
    # Converte os dados para ndarray caso sejam DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    plt.figure(figsize=(8, 5))

    # Plota os pontos com base nos rótulos
    sns.scatterplot(
        x=data[:, 0], y=data[:, 1], hue=labels, palette="viridis", legend="full"
    )

    # Plota os centróides, se fornecidos
    if centroids is not None:
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=300,
            c="red",
            marker="X",
            label="Centroides",
        )

    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.title("Clusters e seus Centroides")
    plt.legend()
    plt.grid(True)

    if view:
        plt.show()

    return plt


def plot_clusters_pairwise(
    data, labels, feature_names=None, view=True, single_figure=False
):
    """
    Plota os clusters gerados pelo KMeans em pares de variáveis (2 a 2).

    Args:
        data (np.ndarray or pd.DataFrame): Dados utilizados para o clustering.
        labels (list or np.ndarray): Rótulos dos clusters para cada ponto.
        feature_names (list, optional): Lista com os nomes das variáveis. Default é None.
        view (bool): Se True, exibe os gráficos imediatamente. Default é True.
        single_figure (bool): Se True, gera todos os pares em subplots de uma única figura. Default é False.

    Returns:
        None
    """
    # Converte os dados para DataFrame se for ndarray
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(
            data,
            columns=(
                feature_names
                if feature_names
                else [f"Var{i}" for i in range(data.shape[1])]
            ),
        )

    # Gera todas as combinações de pares de variáveis
    pairs = list(combinations(data.columns, 2))

    if single_figure:
        # Configura a figura com subplots
        n_pairs = len(pairs)
        n_cols = 3  # Número de colunas nos subplots
        n_rows = (
            n_pairs + n_cols - 1
        ) // n_cols  # Calcula o número de linhas necessário

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()  # Garante que os eixos sejam iteráveis

        for i, (x_var, y_var) in enumerate(pairs):
            sns.scatterplot(
                x=data[x_var],
                y=data[y_var],
                hue=labels,
                palette="viridis",
                legend="full",
                ax=axes[i],
            )
            axes[i].set_xlabel(x_var)
            axes[i].set_ylabel(y_var)
            axes[i].set_title(f"{x_var} vs {y_var}")
            axes[i].grid(True)

        # Remove subplots vazios
        for j in range(len(pairs), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if view:
            plt.show()

    else:
        for x_var, y_var in pairs:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(
                x=data[x_var],
                y=data[y_var],
                hue=labels,
                palette="viridis",
                legend="full",
            )
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.title(f"Clusters: {x_var} vs {y_var}")
            plt.grid(True)

            if view:
                plt.show()


def save_plot_to_tempfile(plt):
    """
    Salva um gráfico matplotlib em um arquivo temporário.

    Args:
        plt: Objeto matplotlib.pyplot contendo o gráfico.

    Returns:
        str: Caminho do arquivo temporário onde o gráfico foi salvo.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(
            temp_file.name, bbox_inches="tight"
        )  # Garante que o gráfico seja salvo corretamente
        temp_file.close()  # Fecha o arquivo para evitar problemas
        return temp_file.name
