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

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

# ---------------------------------------------------------------------
# Setup de path e configs do seu projeto
# ---------------------------------------------------------------------

# Resolve o diretório base do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

# Inclui a raiz no sys.path para permitir imports do src/
sys.path.append(str(BASE_DIR))

from src.config.config_logger import logger
from src.config.config_dynaconf import get_settings

# Inicializa configurações 


# ---------------------------------------------------------------------
# Funções de baixo nível (wrappers simples) - opcionais
# ---------------------------------------------------------------------

def apply_scaling(
    data: pd.DataFrame | np.ndarray,
    *,
    scaler_type: str = "standard",
    **kwargs: Any,
) -> np.ndarray:
    """
    Aplica escalonamento aos dados com base no tipo de escalonador.

    Tipos suportados:
      - "standard": z-score (média 0, desvio 1)
      - "robust": baseado em mediana e IQR (mais robusto a outliers)
      - "minmax": escala para intervalo (por padrão [0, 1])

    Args:
        data:
            Dados a serem escalonados (DataFrame ou ndarray).
        scaler_type:
            Tipo do escalonador: "standard", "robust", "minmax".
        **kwargs:
            Parâmetros adicionais do scaler (ex.: feature_range no MinMaxScaler).

    Returns:
        np.ndarray:
            Matriz escalonada.

    Raises:
        ValueError:
            Se scaler_type não for suportado.
    """
    # Loga a escolha do escalonador
    logger.info("Aplicando escalonamento do tipo %s aos dados.", scaler_type)

    # Mapeia string -> classe (mais pythonico do que if/elif repetido)
    scalers_map = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }

    # Busca a classe do scaler, ou lança erro com mensagem clara
    scaler_cls = scalers_map.get(scaler_type)
    if scaler_cls is None:
        raise ValueError("Tipo de escalonador não suportado. Use: 'standard', 'robust' ou 'minmax'.")

    # Instancia o scaler com kwargs e aplica fit_transform
    scaler = scaler_cls(**kwargs)
    return scaler.fit_transform(data)


def apply_log_transformation(
    data: pd.DataFrame | np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """
    Aplica transformação log1p (log(1 + x)) aos dados.

    Útil para reduzir assimetria (heavy tails) e comprimir escalas.

    Args:
        data:
            Dados a serem transformados.
        **kwargs:
            Parâmetros adicionais do FunctionTransformer.

    Returns:
        np.ndarray:
            Dados transformados.
    """
    # Loga a transformação aplicada
    logger.info("Aplicando transformação log1p aos dados.")

    # FunctionTransformer padroniza a interface sklearn
    transformer = FunctionTransformer(func=np.log1p, **kwargs)

    # Observação: aqui não chamamos fit(), pois log1p não precisa aprender parâmetros
    return transformer.transform(data)


# ---------------------------------------------------------------------
# Utilitários / Heurísticas (coluna a coluna)
# ---------------------------------------------------------------------

def _is_binary(s: pd.Series) -> bool:
    """
    Detecta se uma série é binária (até 2 valores distintos ignorando NaNs).

    Por que importa?
      - Colunas binárias geralmente NÃO precisam de scaling e,
        dependendo do modelo, escalonar pode até piorar interpretabilidade.

    Args:
        s: Série a ser analisada.

    Returns:
        True se a coluna tiver <= 2 valores únicos (sem NaN), senão False.
    """
    # Remove NaNs para analisar apenas valores válidos
    x = s.dropna()

    # Se ficou vazio, não dá pra afirmar que é binário
    if x.empty:
        return False

    # Conta valores únicos (pd.unique é rápido e funciona com tipos mistos)
    return len(pd.unique(x)) <= 2


def _outlier_ratio_iqr(s: pd.Series, *, iqr_k: float = 1.5, min_samples: int = 8) -> float:
    """
    Calcula a proporção de outliers pelo método do IQR (Tukey fences).

    Um valor é outlier se:
      x < Q1 - iqr_k * IQR  ou  x > Q3 + iqr_k * IQR

    Heurística importante:
      - Com amostras muito pequenas, quartis/IQR ficam instáveis.
        Por isso retornamos 0.0 quando n < min_samples.

    Args:
        s:
            Série a ser analisada (pode conter NaN e strings).
        iqr_k:
            Multiplicador do IQR (1.5 é padrão; 3.0 é mais conservador).
        min_samples:
            Mínimo de amostras válidas para confiar em quartis/IQR.

    Returns:
        Proporção de outliers (0 a 1).
    """
    # Converte para numérico; valores inválidos viram NaN
    x = pd.to_numeric(s, errors="coerce").dropna()

    # Amostra pequena -> evitar falsos positivos
    if x.size < min_samples:
        return 0.0

    # Calcula quartis Q1 e Q3
    q1, q3 = np.percentile(x, [25, 75])

    # IQR é uma medida robusta de dispersão (Q3 - Q1)
    iqr = q3 - q1

    # Se IQR é zero, não há dispersão (coluna constante na prática)
    if iqr == 0:
        return 0.0

    # Define cercas inferiores/superiores (Tukey fences)
    lower_bound = q1 - iqr_k * iqr
    upper_bound = q3 + iqr_k * iqr

    # Máscara booleana de outliers
    outliers = (x < lower_bound) | (x > upper_bound)

    # Média de booleano = proporção True (outliers)
    return float(outliers.mean())


def _skewness(s: pd.Series, *, min_samples: int = 8) -> float:
    """
    Calcula a assimetria (skewness) de forma robusta a NaNs.

    Args:
        s: Série a ser analisada.
        min_samples: mínimo de amostras para estimativa confiável.

    Returns:
        Skewness (float). Retorna 0.0 se a amostra for pequena.
    """
    # Converte para numérico e remove NaNs
    x = pd.to_numeric(s, errors="coerce").dropna()

    # Skewness com poucos dados é instável
    if x.size < min_samples:
        return 0.0

    # pandas skew (Fisher-Pearson)
    return float(pd.Series(x).skew())


def _is_nonnegative(s: pd.Series) -> bool:
    """
    Verifica se todos os valores válidos (numéricos) são >= 0.

    Útil porque log1p exige valores não-negativos.

    Args:
        s: Série a ser analisada.

    Returns:
        True se não houver negativos; True também para série vazia após dropna.
    """
    x = pd.to_numeric(s, errors="coerce").dropna()
    return True if x.empty else bool((x >= 0).all())


def _near_constant(s: pd.Series, *, tol_unique: int = 1) -> bool:
    """
    Detecta se uma coluna é constante/quase constante.

    Args:
        s: Série a ser analisada.
        tol_unique:
            Número máximo de valores únicos para considerar constante.
            - 1: constante
            - 2+: "quase constante" (útil em alguns cenários)

    Returns:
        True se nunique <= tol_unique, senão False.
    """
    x = pd.to_numeric(s, errors="coerce").dropna()
    return True if x.empty else (x.nunique() <= tol_unique)


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

    # Define colunas candidatas
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
            reasons.append(f"near_constant(nunique<={near_constant_unique_tol})")
            plans.append(ColumnPlan(col=col, do_log1p=False, scaler_type="none", reasons=tuple(reasons)))
            continue

        # (b) Mantém binárias sem scaling
        if _is_binary(s):
            passthrough_cols.append(col)
            reasons.append("binary_passthrough")
            plans.append(ColumnPlan(col=col, do_log1p=False, scaler_type="none", reasons=tuple(reasons)))
            continue

        # (c) Calcula heurísticas (outliers/skew/nonneg)
        out_ratio = _outlier_ratio_iqr(s, iqr_k=robust_iqr_k, min_samples=min_samples_stats)
        skew = _skewness(s, min_samples=min_samples_stats)
        nonneg = _is_nonnegative(s)

        # (d) Decide log1p
        do_log1p = False
        if abs(skew) >= skew_threshold_for_log:
            if (not log_requires_nonnegative) or nonneg:
                do_log1p = True
                reasons.append(f"log1p(skew={skew:.2f})")
            else:
                reasons.append(f"skip_log(has_negative, skew={skew:.2f})")

        # (e) Decide scaler
        if out_ratio >= outlier_ratio_threshold:
            scaler_type = "robust"
            reasons.append(f"robust(outlier_ratio={out_ratio:.3f}, k={robust_iqr_k})")
        else:
            scaler_type = "minmax" if prefer_minmax else "standard"
            reasons.append(f"{scaler_type}(outlier_ratio={out_ratio:.3f})")

        plans.append(ColumnPlan(col=col, do_log1p=do_log1p, scaler_type=scaler_type, reasons=tuple(reasons)))

    # --------------------------------------------------------------
    # 2) Execução: aplica transformações de fato
    # --------------------------------------------------------------
    for plan in plans:
        col = plan.col

        # Pula colunas marcadas para drop
        if col in dropped_cols:
            continue

        # Converte a coluna para numérico e forma (n, 1) para sklearn
        # Nota: valores inválidos viram NaN; scalers do sklearn não aceitam NaN.
        # Se você tem NaNs, recomendo plugar um SimpleImputer antes do scaling.
        x = pd.to_numeric(df_out[col], errors="coerce").to_numpy().reshape(-1, 1)

        # Aplica log1p (não precisa fit)
        if plan.do_log1p:
            logger.info("[%s] Aplicando log1p. Motivos: %s", col, list(plan.reasons))
            x = FunctionTransformer(func=np.log1p, validate=False).transform(x)

        # Aplica o scaler escolhido
        if plan.scaler_type == "standard":
            logger.info("[%s] Aplicando StandardScaler. Motivos: %s", col, list(plan.reasons))
            scaler = StandardScaler(**scaler_kwargs.get("standard", {}))
            x = scaler.fit_transform(x)

        elif plan.scaler_type == "robust":
            logger.info("[%s] Aplicando RobustScaler. Motivos: %s", col, list(plan.reasons))
            scaler = RobustScaler(**scaler_kwargs.get("robust", {}))
            x = scaler.fit_transform(x)

        elif plan.scaler_type == "minmax":
            logger.info("[%s] Aplicando MinMaxScaler. Motivos: %s", col, list(plan.reasons))
            scaler = MinMaxScaler(**scaler_kwargs.get("minmax", {}))
            x = scaler.fit_transform(x)

        else:
            # "none" -> passthrough (binárias etc.)
            logger.info("[%s] Sem scaling (passthrough). Motivos: %s", col, list(plan.reasons))

        # Reinsere no DataFrame como vetor 1D
        df_out[col] = x.ravel()

    # --------------------------------------------------------------
    # 3) Drop final (colunas constantes)
    # --------------------------------------------------------------
    if dropped_cols:
        logger.info("Removendo colunas constantes/quase constantes: %s", dropped_cols)
        df_out = df_out.drop(columns=dropped_cols)

    # --------------------------------------------------------------
    # 4) Monta relatório imutável (bom para auditoria/explicabilidade)
    # --------------------------------------------------------------
    report = PreprocessReport(
        plans=tuple(plans),
        dropped_cols=tuple(dropped_cols),
        passthrough_cols=tuple(passthrough_cols),
    )

    return df_out, report
