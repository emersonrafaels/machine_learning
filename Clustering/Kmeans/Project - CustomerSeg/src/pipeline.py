import sys
from pathlib import Path

import mlflow
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Resolve o diretório base do projeto
BASE_DIR = Path(__file__).resolve().parents[3]

# Inclui a raiz no sys.path para permitir imports do src/
sys.path.append(str(BASE_DIR))

from src.config.config_logger import logger
from src.utils.datasets.get_data import buscar_dados_wholesale_customers
from src.preprocessing.preprocessing import smart_preprocess_fit_transform
from src.model.model import create_kmeans_model, KMeansConfig, find_optimal_clusters
from src.postprocessing.postprocessing import analyze_clusters
from src.utils.visualization.kmeans_visualization import (
    plot_clusters,
    plot_clusters_pairwise,
    save_plot_to_tempfile,
)


def run_pipeline(
    list_columns_to_drop=None,
    dir_export_data=None,
    k_clusters=None,
    find_optimal_clusters_method="silhouette",
):
    """
    Executa a pipeline completa de ciência de dados para segmentação de clientes.

    Args:
        list_columns_to_drop (list[str], opcional):
            Lista de nomes de colunas a serem descartadas do conjunto de dados.
            Padrão é None (nenhuma coluna descartada).
        dir_export_data (str, opcional):
            Diretório para exportar os dados obtidos como um arquivo CSV.
        k_clusters (int, opcional):
            Número de clusters para o modelo KMeans.
            Definir como None para detectar automaticamente o número ideal de clusters.
            Padrão é None.
        find_optimal_clusters_method (str):
            Método para determinar o número ideal de clusters ("silhouette" ou "elbow").

    Returns:
        None

    """
    mlflow.set_experiment("Customer Segmentation")
    with mlflow.start_run():

        # 1. Obter os dados
        X, _ = buscar_dados_wholesale_customers(
            list_columns_to_drop=list_columns_to_drop, dir_export_data=dir_export_data
        )

        # 2. Pré-processamento
        X_transformed, report = smart_preprocess_fit_transform(df=X)

        # 3. Modelagem
        config = KMeansConfig()

        print("-" * 50)

        if not k_clusters:
            k_clusters = find_optimal_clusters(
                data=X_transformed,
                min_clusters=config.k_min,
                max_clusters=config.k_max,
                method=find_optimal_clusters_method,
                plot=True,
            )

            logger.info(f"Número ideal de clusters detectado: {k_clusters}")

        else:
            logger.info(f"Número de clusters fornecido: {k_clusters}")

        # Criando o modelo final
        model = create_kmeans_model(n_clusters=k_clusters, config=config)
        
        # Aplicando sobre os dados
        labels = model.fit_predict(X_transformed)

        mlflow.log_param("k_clusters", k_clusters)

        # 4. Pós-processamento
        cluster_info = analyze_clusters(X, labels)

        # 5. Avaliação
        silhouette_avg = silhouette_score(X_transformed, labels)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        print(f"Silhouette Score: {silhouette_avg}")

        # 6. Resultados
        print("-" * 50)
        print("Cluster Information:")
        for cluster_id, info in cluster_info.items():
            print(f"Cluster {cluster_id}: {info}")

        # Reduz a dimensionalidade para 2 componentes principais usando PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_transformed)

        # Plotando os clusters com as componentes principais
        plt = plot_clusters(
            data=X_pca,
            labels=labels,
            centroids=pca.transform(model.cluster_centers_),
            view=False,
        )

        # Salva o gráfico em um arquivo temporário e registra como artefato
        temp_file_path = save_plot_to_tempfile(plt)
        mlflow.log_artifact(temp_file_path, artifact_path="plots")

        # Plotando os clusters em pares de variáveis
        plot_clusters_pairwise(
            data=X,
            labels=labels,
            feature_names=X.columns.tolist(),
            view=True,
            single_figure=True,
        )

        logger.success("Pipeline concluída com sucesso.")
