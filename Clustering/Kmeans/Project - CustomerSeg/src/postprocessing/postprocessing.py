def analyze_clusters(data, labels):
    """
    Analisa os clusters gerados pelo modelo.

    Args:
        data (pd.DataFrame): Dados originais.
        labels (np.ndarray): Rótulos dos clusters.

    Returns:
        dict: Informações sobre os clusters.
    """
    cluster_info = {}
    for cluster_id in set(labels):
        cluster_data = data[labels == cluster_id]
        cluster_info[cluster_id] = {
            "size": len(cluster_data),
            "mean": cluster_data.mean().to_dict(),
        }
    return cluster_info
