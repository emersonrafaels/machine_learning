import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, MinMaxScaler

def apply_scaling(data, scaler_type="standard", **kwargs):
    """
    Aplica escalonamento aos dados com base no tipo de escalonador fornecido.

    Tipos de escalonadores suportados:
        - "standard": Remove a média e escala os dados para desvio padrão unitário.
        - "robust": Remove a mediana e escala os dados usando o intervalo interquartil (IQR), útil para lidar com outliers.
        - "minmax": Escala os dados para o intervalo [0, 1].

    Args:
        data (pd.DataFrame): Dados a serem escalonados.
        scaler_type (str): Tipo de escalonador ("standard", "robust" ou "minmax").
        **kwargs: Parâmetros adicionais para os escalonadores (e.g., feature_range para MinMaxScaler).

    Returns:
        pd.DataFrame: Dados escalonados.

    Raises:
        ValueError: Se o tipo de escalonador não for suportado.
    """
    if scaler_type == "standard":
        scaler = StandardScaler(**kwargs)
    elif scaler_type == "robust":
        scaler = RobustScaler(**kwargs)
    elif scaler_type == "minmax":
        scaler = MinMaxScaler(**kwargs)
    else:
        raise ValueError("Tipo de escalonador não suportado. Escolha entre 'standard', 'robust' ou 'minmax'.")

    return scaler.fit_transform(data)

def apply_log_transformation(data, **kwargs):
    """
    Aplica transformação log1p aos dados.

    Args:
        data (pd.DataFrame): Dados a serem transformados.
        **kwargs: Parâmetros adicionais para o FunctionTransformer (e.g., validate, inverse_func).

    Returns:
        pd.DataFrame: Dados transformados.
    """
    transformer = FunctionTransformer(func=np.log1p, **kwargs)
    return transformer.transform(data)