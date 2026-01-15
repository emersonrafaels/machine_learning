import mlflow
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.utils.datasets.get_data import buscar_dados_wholesale_customers
from src.preprocessing.preprocessing import smart_preprocess_fit_transform
from src.model.model import create_kmeans_model, KMeansConfig
from src.postprocessing.postprocessing import analyze_clusters
from src.utils.visualization.kmeans_visualization import plot_clusters

def run_pipeline():
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

        print("Pipeline concluída com sucesso.")