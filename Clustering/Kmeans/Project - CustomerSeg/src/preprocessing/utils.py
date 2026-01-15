# utils.py

"""
Este módulo contém funções utilitárias para o pré-processamento de dados.
Inclui funções para escalonamento, transformação logarítmica e análise de colunas.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from src.config.config_logger import logger

# Funções de escalonamento


def apply_scaling(
    data: pd.DataFrame | np.ndarray,
    *,
    scaler_type: str = "standard",
    column_name: str = "",
    **kwargs,
) -> np.ndarray:
    """
    Aplica escalonamento aos dados com base no tipo de escalonador especificado.

    Tipos suportados:
      - "standard": Z-score (média 0, desvio padrão 1).
      - "robust": Baseado em mediana e IQR (mais robusto a outliers).
      - "minmax": Escala os dados para um intervalo (por padrão [0, 1]).

    Args:
        data (pd.DataFrame | np.ndarray):
            Dados a serem escalonados.
        scaler_type (str):
            Tipo do escalonador: "standard", "robust", "minmax".
        column_name (str):
            Nome da coluna (para logs).
        **kwargs:
            Parâmetros adicionais para o escalonador (ex.: feature_range no MinMaxScaler).

    Returns:
        np.ndarray: Dados escalonados.

    Raises:
        ValueError: Se o tipo de escalonador não for suportado.
    """
    logger.info(
        f"Aplicando escalonamento do tipo {scaler_type} aos dados da coluna '{column_name}'."
    )

    # Mapeia os tipos de escalonadores para suas classes correspondentes
    scalers_map = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }

    # Obtém a classe do escalonador ou lança um erro se o tipo for inválido
    scaler_cls = scalers_map.get(scaler_type)
    if scaler_cls is None:
        raise ValueError(
            "Tipo de escalonador não suportado. Use: 'standard', 'robust' ou 'minmax'."
        )

    # Converte para DataFrame ou ndarray se for uma Series
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Instancia o escalonador com os parâmetros fornecidos e aplica o fit_transform
    scaler = scaler_cls(**kwargs)
    return scaler.fit_transform(data)


def apply_log_transformation(
    data: pd.DataFrame | np.ndarray, column_name: str = "", **kwargs
) -> np.ndarray:
    """
    Aplica transformação logarítmica log1p (log(1 + x)) aos dados.

    Útil para reduzir assimetrias (distribuições com caudas longas) e comprimir escalas.

    Args:
        data (pd.DataFrame | np.ndarray):
            Dados a serem transformados.
        column_name (str):
            Nome da coluna (para logs).
        **kwargs:
            Parâmetros adicionais para o FunctionTransformer.

    Returns:
        np.ndarray: Dados transformados.
    """
    logger.info(f"Aplicando transformação log1p aos dados da coluna '{column_name}'.")

    # Cria um transformador logarítmico usando log1p
    transformer = FunctionTransformer(func=np.log1p, **kwargs)

    # Aplica a transformação diretamente (não é necessário fit para log1p)
    return transformer.transform(data)


# Funções auxiliares para análise de colunas
def _is_binary(s: pd.Series) -> bool:
    """
    Detecta se uma série é binária (contém até 2 valores distintos, ignorando NaNs).

    Args:
        s (pd.Series): Série a ser analisada.

    Returns:
        bool: True se a série for binária, False caso contrário.
    """
    # Remove valores NaN para análise
    x = s.dropna()

    # Verifica se há no máximo 2 valores únicos
    return len(pd.unique(x)) <= 2 if not x.empty else False


def _outlier_ratio_iqr(
    s: pd.Series, *, iqr_k: float = 1.5, min_samples: int = 8
) -> float:
    """
    Calcula a proporção de outliers em uma série usando o método do IQR (Tukey fences).

    Um valor é considerado outlier se:
      x < Q1 - iqr_k * IQR  ou  x > Q3 + iqr_k * IQR

    Args:
        s (pd.Series): Série a ser analisada.
        iqr_k (float): Multiplicador do IQR (1.5 é padrão; 3.0 é mais conservador).
        min_samples (int): Número mínimo de amostras para cálculo confiável.

    Returns:
        float: Proporção de outliers (0 a 1).
    """
    # Converte para numérico e remove valores inválidos
    x = pd.to_numeric(s, errors="coerce").dropna()

    # Retorna 0.0 se houver poucas amostras
    if x.size < min_samples:
        return 0.0

    # Calcula os quartis e o IQR
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1

    # Retorna 0.0 se o IQR for zero (coluna constante)
    if iqr == 0:
        return 0.0

    # Define os limites inferior e superior para outliers
    lower_bound = q1 - iqr_k * iqr
    upper_bound = q3 + iqr_k * iqr

    # Calcula a proporção de outliers
    outliers = (x < lower_bound) | (x > upper_bound)
    return float(outliers.mean())


def _skewness(s: pd.Series, *, min_samples: int = 8) -> float:
    """
    Calcula a assimetria (skewness) de uma série, ignorando NaNs.

    Args:
        s (pd.Series): Série a ser analisada.
        min_samples (int): Número mínimo de amostras para cálculo confiável.

    Returns:
        float: Valor da assimetria. Retorna 0.0 se houver poucas amostras.
    """
    # Converte para numérico e remove valores inválidos
    x = pd.to_numeric(s, errors="coerce").dropna()

    # Retorna 0.0 se houver poucas amostras
    return float(pd.Series(x).skew()) if x.size >= min_samples else 0.0


def _is_nonnegative(s: pd.Series) -> bool:
    """
    Verifica se todos os valores válidos em uma série são não-negativos.

    Args:
        s (pd.Series): Série a ser analisada.

    Returns:
        bool: True se todos os valores forem >= 0, False caso contrário.
    """
    # Converte para numérico e remove valores inválidos
    x = pd.to_numeric(s, errors="coerce").dropna()

    # Retorna True se todos os valores forem não-negativos
    return (x >= 0).all() if not x.empty else True


def _near_constant(s: pd.Series, *, tol_unique: int = 1) -> bool:
    """
    Detecta se uma série é constante ou quase constante.

    Args:
        s (pd.Series): Série a ser analisada.
        tol_unique (int): Número máximo de valores únicos para considerar constante.

    Returns:
        bool: True se a série for constante ou quase constante, False caso contrário.
    """
    # Converte para numérico e remove valores inválidos
    x = pd.to_numeric(s, errors="coerce").dropna()

    # Verifica se o número de valores únicos é menor ou igual ao limite
    return x.nunique() <= tol_unique if not x.empty else True
