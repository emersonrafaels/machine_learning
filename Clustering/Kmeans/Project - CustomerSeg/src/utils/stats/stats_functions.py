from typing import Dict
import pandas as pd

def calculate_boxplot_stats(
    data: pd.DataFrame,
    x: str,
    whisker_coef: float = 1.5
) -> Dict[str, float]:
    """
    Calcula as estatísticas utilizadas em um boxplot (regra de Tukey).

    Args:
        data : pd.DataFrame
            DataFrame contendo os dados.
        x : str
            Nome da coluna numérica.
        whisker_coef : float, opcional
            Coeficiente do IQR para cálculo dos whiskers (default=1.5).

    Returns:
        dict
            Estatísticas do boxplot.
    """
    
    if x not in data.columns:
        raise ValueError(f"Coluna '{x}' não encontrada no DataFrame.")
        
    series = data[x].dropna()

    q1 = series.quantile(0.25)
    q2 = series.quantile(0.50)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_whisker = q1 - whisker_coef * iqr
    upper_whisker = q3 + whisker_coef * iqr

    outliers = series[(series < lower_whisker) | (series > upper_whisker)]

    return {
        "feature": x,
        "q1_25": q1,
        "median_50": q2,
        "q3_75": q3,
        "iqr": iqr,
        "whisker_lower": lower_whisker,
        "whisker_upper": upper_whisker,
        "n_outliers": len(outliers),
        "pct_outliers": len(outliers) / len(series) * 100
    }
