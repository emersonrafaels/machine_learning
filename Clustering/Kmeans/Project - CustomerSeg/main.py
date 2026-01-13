import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn

from src.utils.datasets.get_data import buscar_dados_wholesale_customers
from src.preprocessing.preprocessing import apply_scaling, apply_log_transformation
from src.model.model import create_kmeans_model, KMeansConfig
from src.postprocessing.postprocessing import analyze_clusters

def main_pipeline():
    """
    Executa a pipeline completa de ciência de dados para segmentação de clientes.
    """
    mlflow.set_experiment("Customer Segmentation")
    with mlflow.start_run():
        # 1. Obter os dados
        X, _ = buscar_dados_wholesale_customers()

        # 2. Pré-processamento
        X_scaled = apply_scaling(X, scaler_type="standard")
        X_transformed = apply_log_transformation(X_scaled)

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

if __name__ == "__main__":
    main_pipeline()