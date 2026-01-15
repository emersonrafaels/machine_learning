import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Resolve o diretório base do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

# Inclui a raiz no sys.path para permitir imports do src/
sys.path.append(str(BASE_DIR))

from src.config.config_logger import logger
from src.config.config_dynaconf import get_settings
from src.pipeline import run_pipeline

# Inicializa configurações
settings = get_settings()

from src.utils.datasets.get_data import buscar_dados_wholesale_customers
from src.preprocessing.preprocessing import smart_preprocess_fit_transform
from src.model.model import create_kmeans_model, KMeansConfig
from src.postprocessing.postprocessing import analyze_clusters
from src.utils.visualization.kmeans_visualization import plot_clusters


def main_pipeline():
    """
    Executa a pipeline completa de ciência de dados para segmentação de clientes.
    """
    mlflow.set_experiment("Customer Segmentation")
    with mlflow.start_run():

        # Iniciando o flavor do sklearn no MLflow
        mlflow.sklearn.autolog()

        # 1. Obter os dados
        X, _ = buscar_dados_wholesale_customers()

        # 2. Pré-processamento
        X_transformed, report = smart_preprocess_fit_transform(df=X)

        # 3. Modelagem
        config = KMeansConfig()
        model = create_kmeans_model(n_clusters=5, config=config)
        labels = model.fit_predict(X_transformed)

        # Logando o modelo no MLflow
        mlflow.sklearn.log_model(model, "kmeans_model")

        # 4. Pós-processamento
        cluster_info = analyze_clusters(X, labels)

        # 5. Avaliação
        silhouette_avg = silhouette_score(X_transformed, labels)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        print(f"Silhouette Score: {silhouette_avg}")

        # 6. Resultados
        print("Cluster Information:")
        for cluster_id, info in cluster_info.items():
            print(f"Cluster {cluster_id}: {info}")

        # Reduz a dimensionalidade para 2 componentes principais usando PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_transformed)

        # Plotando os clusters com as componentes principais
        plot_clusters(data=X_pca, 
                      labels=labels, 
                      centroids=pca.transform(model.cluster_centers_))

        logger.info("Pipeline de segmentação de clientes concluída com sucesso.")


if __name__ == "__main__":
    run_pipeline()
