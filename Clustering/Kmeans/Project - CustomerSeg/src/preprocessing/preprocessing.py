"""
smart_preprocess.py

Pré-processamento "inteligente" (heurístico) para dados tabulares:
- Decide, coluna a coluna, se aplica log1p e qual scaler usar (Standard/Robust/MinMax)
- Ignora colunas binárias e não numéricas (quando numeric_only=True)
- Remove colunas constantes/quase constantes
- Retorna também um relatório (plans) explicando decisões

Observação importante (engenharia/produção):
Este módulo implementa um "fit_transform" de conveniência. Para produção,
o ideal é ter uma classe com `fit()` (guarda transformadores por coluna) e `transform()`
(reaplica em novos dados sem leakage). Se você quiser, eu te devolvo essa versão.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from src.preprocessing.utils import (
    apply_scaling,
    apply_log_transformation,
    _is_binary,
    _outlier_ratio_iqr,
    _skewness,
    _is_nonnegative,
    _near_constant,
)
from src.config.config_logger import logger

# ---------------------------------------------------------------------
# Relatórios (explicabilidade do que foi aplicado)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnPlan:
    """Plano de preprocessamento de uma coluna."""

    col: str
    do_log1p: bool
    scaler_type: str  # "standard" | "robust" | "minmax" | "none"
    reasons: Tuple[str, ...]  # tupla imutável (mais segura)


@dataclass(frozen=True)
class PreprocessReport:
    """Relatório final do preprocessamento."""

    plans: Tuple[ColumnPlan, ...]
    dropped_cols: Tuple[str, ...]
    passthrough_cols: Tuple[str, ...]


# ---------------------------------------------------------------------
# Orquestrador inteligente (fit_transform)
# ---------------------------------------------------------------------


def smart_preprocess_fit_transform(
    df: pd.DataFrame,
    *,
    numeric_only: bool = True,
    prefer_minmax: bool = False,
    outlier_ratio_threshold: float = 0.03,
    skew_threshold_for_log: float = 1.0,
    log_requires_nonnegative: bool = True,
    robust_iqr_k: float = 1.5,
    min_samples_stats: int = 8,
    near_constant_unique_tol: int = 1,
    scaler_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, PreprocessReport]:
    """
    Orquestra preprocessamento "inteligente" com base nas propriedades dos dados.

    Estratégia (coluna a coluna):
      1) Seleciona colunas-alvo (numéricas, se numeric_only=True).
      2) Remove colunas constantes/quase constantes.
      3) Colunas binárias passam sem scaling (passthrough).
      4) Se skew alto e (opcionalmente) coluna não-negativa -> aplica log1p.
      5) Se proporção de outliers (IQR) alta -> RobustScaler,
         senão StandardScaler (ou MinMaxScaler se prefer_minmax=True).

    Importante:
      - Este método aplica transformações diretamente (fit_transform) e NÃO guarda estado.
      - Para produção, prefira uma classe com fit/transform.

    Args:
        df:
            DataFrame de entrada.
        numeric_only:
            Se True, aplica heurísticas apenas em colunas numéricas.
        prefer_minmax:
            Se True, usa MinMaxScaler quando não houver outliers relevantes
            (em vez de StandardScaler).
        outlier_ratio_threshold:
            Proporção mínima para considerar que há outliers relevantes e usar RobustScaler.
        skew_threshold_for_log:
            Se |skew| >= esse valor, considera aplicar log1p.
        log_requires_nonnegative:
            Se True, só aplica log1p quando todos os valores válidos forem >= 0.
        robust_iqr_k:
            Multiplicador do IQR para detecção de outliers.
        min_samples_stats:
            Mínimo de amostras válidas para calcular skew/outliers com confiança.
        near_constant_unique_tol:
            Se nunique <= esse valor, a coluna é considerada constante e será removida.
        scaler_kwargs:
            Dict com kwargs por scaler:
              {
                "standard": {...},
                "robust": {...},
                "minmax": {...},
              }

    Returns:
        (df_transformado, report):
            - df_transformado: DataFrame com colunas processadas e colunas constantes removidas.
            - report: relatório com planos e colunas passthrough/dropped.
    """
    # Garante dict padrão (evita None checks espalhados)
    scaler_kwargs = scaler_kwargs or {}

    # Copia defensiva para não mutar o df original
    df_out = df.copy()

    # Define colunas candidatas (apenas colunas numéricas, se pedido)
    if numeric_only:
        candidate_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    else:
        candidate_cols = df_out.columns.tolist()

    # Colunas que não serão transformadas (por não serem candidatas)
    passthrough_cols: List[str] = [c for c in df_out.columns if c not in candidate_cols]

    dropped_cols: List[str] = []
    plans: List[ColumnPlan] = []

    # --------------------------------------------------------------
    # 1) Planejamento: decide o que fazer em cada coluna
    # --------------------------------------------------------------
    for col in candidate_cols:
        s = df_out[col]
        reasons: List[str] = []

        # (a) Remove colunas constantes/quase constantes
        if _near_constant(s, tol_unique=near_constant_unique_tol):
            dropped_cols.append(col)
            reasons.append("Coluna constante ou quase constante.")
            plans.append(
                ColumnPlan(
                    col=col, do_log1p=False, scaler_type="none", reasons=tuple(reasons)
                )
            )
            continue

        # (b) Mantém binárias sem scaling
        if _is_binary(s):
            passthrough_cols.append(col)
            reasons.append("Coluna binária detectada.")
            plans.append(
                ColumnPlan(
                    col=col, do_log1p=False, scaler_type="none", reasons=tuple(reasons)
                )
            )
            continue

        # (c) Calcula heurísticas (outliers/skew/nonneg)
        out_ratio = _outlier_ratio_iqr(
            s, iqr_k=robust_iqr_k, min_samples=min_samples_stats
        )
        skew = _skewness(s, min_samples=min_samples_stats)
        nonneg = _is_nonnegative(s)

        # (d) Decide log1p
        do_log1p = False
        if abs(skew) >= skew_threshold_for_log:
            if log_requires_nonnegative and not nonneg:
                reasons.append("Assimetria alta, mas valores negativos impedem log1p.")
            else:
                do_log1p = True
                reasons.append("Transformação log1p aplicada devido à alta assimetria.")

        # (e) Decide scaler
        scaler_type = "robust" if out_ratio > outlier_ratio_threshold else "standard"
        if prefer_minmax and scaler_type == "standard":
            scaler_type = "minmax"
        reasons.append(f"Escalonador escolhido: {scaler_type}.")

        plans.append(
            ColumnPlan(
                col=col,
                do_log1p=do_log1p,
                scaler_type=scaler_type,
                reasons=tuple(reasons),
            )
        )

    # --------------------------------------------------------------
    # 2) Execução: aplica transformações de fato
    # --------------------------------------------------------------
    for plan in plans:
        s = df_out[plan.col]
        if plan.do_log1p:
            df_out[plan.col] = apply_log_transformation(data=s, column_name=plan.col)
        if plan.scaler_type != "none":
            df_out[plan.col] = apply_scaling(
                s,
                scaler_type=plan.scaler_type,
                column_name=plan.col,
                **scaler_kwargs.get(plan.scaler_type, {}),
            )

    # --------------------------------------------------------------
    # 3) Drop final (colunas constantes)
    # --------------------------------------------------------------
    if dropped_cols:
        df_out.drop(columns=dropped_cols, inplace=True)

    # --------------------------------------------------------------
    # 4) Monta relatório imutável (bom para auditoria/explicabilidade)
    # --------------------------------------------------------------
    report = PreprocessReport(
        plans=tuple(plans),
        dropped_cols=tuple(dropped_cols),
        passthrough_cols=tuple(passthrough_cols),
    )

    return df_out, report
