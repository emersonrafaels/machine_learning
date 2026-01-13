from sklearn.preprocessing import StandardScaler, FunctionTransformer

def apply_scaling(data, scaler_type="standard"):
    """
    Aplica escalonamento aos dados com base no tipo de escalonador fornecido.

    Args:
        data (pd.DataFrame): Dados a serem escalonados.
        scaler_type (str): Tipo de escalonador ("standard" ou "robust").

    Returns:
        pd.DataFrame: Dados escalonados.
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Tipo de escalonador não suportado.")

    return scaler.fit_transform(data)

def apply_log_transformation(data):
    """
    Aplica transformação log1p aos dados.

    Args:
        data (pd.DataFrame): Dados a serem transformados.

    Returns:
        pd.DataFrame: Dados transformados.
    """
    transformer = FunctionTransformer(func=np.log1p, validate=True)
    return transformer.transform(data)