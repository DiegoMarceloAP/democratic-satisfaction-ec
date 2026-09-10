"""
pruebas_estadisticas.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Prueba estadística formal de la
comparación entre los 5 modelos (Regresión Logística, XGBoost,
LightGBM, CNN-LSTM, TabNet). Este script reemplaza la inspección visual
informal de las barras de error por una prueba formal.

Metodología:
  1. Prueba de Friedman (no paramétrica, para >2 muestras pareadas
     sobre las mismas unidades -- aquí, los 8 folds de prueba del
     Time Series Split) sobre cada métrica de interés (PR-AUC y F1
     de la clase "Satisfecho").
  2. Si Friedman resulta significativa (alfa=0.05), se calculan las
     10 comparaciones pareadas posibles entre los 5 modelos mediante
     la prueba de Wilcoxon (signed-rank), y se corrige el p-valor de
     cada una con Holm-Bonferroni para controlar la inflación del
     error tipo I por comparaciones múltiples.

Los 8 valores por modelo se toman de los CSV por fold ya generados
por cada script de modelado (resultados_baseline_logreg.csv,
resultados_baseline_xgboost.csv, resultados_baseline_lightgbm.csv,
resultados_cnn_lstm.csv, resultados_tabnet.csv), ordenados por
anio_test para que la prueba pareada compare siempre el mismo fold
entre modelos.

TAMAÑOS DE EFECTO (observación de revisores: un p-valor no significativo
tras la corrección de Holm-Bonferroni no implica que los modelos sean
"indistinguibles" o "equivalentes" -- solo que, con la potencia
estadística disponible (8 folds pareados), no se detectó una diferencia
que sobreviva la corrección por comparaciones múltiples). Para cada par
de modelos se añaden dos magnitudes complementarias, calculadas siempre
(no solo cuando el p-valor es significativo), para que el lector pueda
juzgar el tamaño de una posible diferencia real más allá de su
significancia formal:
  - Correlación rank-biserial pareada (r): efecto no paramétrico
    asociado a Wilcoxon, calculado manualmente como
    (W+ - W-) / (W+ + W-) sobre los rangos de las diferencias no nulas
    -- rango [-1, 1], magnitud interpretable con las mismas convenciones
    aproximadas que r de Pearson (~0.1 pequeño, ~0.3 mediano, ~0.5
    grande), pero sin asumir normalidad.
  - Diferencia media pareada, con intervalo de confianza bootstrap
    percentil al 95% (10,000 remuestreos de los 8 folds, con
    reemplazo) -- en las unidades originales de la métrica (PR-AUC o
    F1), para una lectura directa de cuánto podría diferir un modelo de
    otro en la práctica.
"""
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

RUTA_TABLAS = Path("reports/tablas")
ARCHIVOS_MODELOS = {
    "Regresión Logística": RUTA_TABLAS / "resultados_baseline_logreg.csv",
    "XGBoost": RUTA_TABLAS / "resultados_baseline_xgboost.csv",
    "LightGBM": RUTA_TABLAS / "resultados_baseline_lightgbm.csv",
    "CNN-LSTM": RUTA_TABLAS / "resultados_cnn_lstm.csv",
    "TabNet": RUTA_TABLAS / "resultados_tabnet.csv",
}
METRICAS = ["pr_auc", "f1_satisfecho"]
OUT_PATH = RUTA_TABLAS / "resultados_estadisticos.csv"
N_BOOTSTRAP = 10_000
SEMILLA = 42  # random_state=42, mismo criterio de reproducibilidad que el resto del proyecto


def correlacion_rank_biserial(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Tamaño de efecto no paramétrico asociado a Wilcoxon signed-rank:
    r = (W+ - W-) / (W+ + W-), calculado sobre los rangos de |diferencia|
    de los pares con diferencia distinta de cero (mismo criterio de
    descarte de empates que scipy.stats.wilcoxon con zero_method='wilcox',
    el valor por defecto). Se calcula a mano (no se toma el 'statistic'
    de scipy.stats.wilcoxon) para no depender de la convención interna de
    esa función sobre cuál de las dos sumas de rangos devuelve.

    Retorna 0.0 si no quedan pares con diferencia distinta de cero
    (caso degenerado, no esperado con los datos de esta tesis).
    """
    diferencias = np.asarray(x1) - np.asarray(x2)
    diferencias = diferencias[diferencias != 0]
    if len(diferencias) == 0:
        return 0.0
    rangos = rankdata(np.abs(diferencias))
    w_positivo = rangos[diferencias > 0].sum()
    w_negativo = rangos[diferencias < 0].sum()
    return (w_positivo - w_negativo) / (w_positivo + w_negativo)


def bootstrap_ci_diferencia_media(
    x1: np.ndarray, x2: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05, random_state: int = SEMILLA,
) -> tuple[float, float, float]:
    """
    Diferencia media pareada (x1 - x2) con intervalo de confianza
    bootstrap percentil, remuestreando con reemplazo los PARES
    fold-a-fold (no cada valor por separado, para conservar el
    emparejamiento). Con solo 8 folds, el intervalo es necesariamente
    amplio -- se reporta igual, de forma transparente, en vez de omitirlo
    por ser poco informativo.
    """
    rng = np.random.default_rng(random_state)
    diferencias = np.asarray(x1) - np.asarray(x2)
    n = len(diferencias)
    remuestreos = rng.choice(diferencias, size=(n_boot, n), replace=True)
    medias_boot = remuestreos.mean(axis=1)
    ci_bajo, ci_alto = np.percentile(medias_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diferencias.mean(), ci_bajo, ci_alto


def cargar_valores_por_modelo(metrica: str) -> dict:
    """Carga los 8 valores de `metrica`, uno por fold, para cada modelo,
    ordenados por anio_test para garantizar el emparejamiento correcto."""
    valores = {}
    for nombre, ruta in ARCHIVOS_MODELOS.items():
        df = pd.read_csv(ruta).sort_values("anio_test")
        valores[nombre] = df[metrica].to_numpy()
    return valores


def holm_bonferroni(p_valores: dict) -> dict:
    """Corrección de Holm-Bonferroni sobre un diccionario {par: p_valor}."""
    pares_ordenados = sorted(p_valores.items(), key=lambda kv: kv[1])
    m = len(pares_ordenados)
    corregidos = {}
    p_previo = 0.0
    for i, (par, p) in enumerate(pares_ordenados):
        ajustado = min((m - i) * p, 1.0)
        ajustado = max(ajustado, p_previo)  # monotonicidad de Holm
        corregidos[par] = round(ajustado, 4)
        p_previo = ajustado
    return corregidos


def evaluar_metrica(metrica: str) -> pd.DataFrame:
    valores = cargar_valores_por_modelo(metrica)
    modelos = list(valores.keys())

    chi2, p_friedman = friedmanchisquare(*[valores[m] for m in modelos])

    filas = []
    p_wilcoxon = {}
    efecto_r = {}
    diff_media = {}
    ci_bajo = {}
    ci_alto = {}
    for m1, m2 in combinations(modelos, 2):
        _, p = wilcoxon(valores[m1], valores[m2])
        p_wilcoxon[f"{m1} vs {m2}"] = p
        efecto_r[f"{m1} vs {m2}"] = correlacion_rank_biserial(valores[m1], valores[m2])
        media, bajo, alto = bootstrap_ci_diferencia_media(valores[m1], valores[m2])
        diff_media[f"{m1} vs {m2}"] = media
        ci_bajo[f"{m1} vs {m2}"] = bajo
        ci_alto[f"{m1} vs {m2}"] = alto

    p_holm = holm_bonferroni(p_wilcoxon)

    for par in p_wilcoxon:
        filas.append({
            "metrica": metrica,
            "comparacion": par,
            "friedman_chi2": round(chi2, 4),
            "friedman_p": round(p_friedman, 4),
            "wilcoxon_p_sin_corregir": round(p_wilcoxon[par], 4),
            "wilcoxon_p_holm": p_holm[par],
            # Renombrado de 'significativo_holm_0.05' -- ver docstring del
            # módulo: no significativo NO implica "sin diferencia real",
            # solo "no detectada con la potencia disponible".
            "diferencia_detectada_holm_0.05": p_holm[par] < 0.05,
            "rank_biserial_r": round(efecto_r[par], 4),
            "diferencia_media": round(diff_media[par], 4),
            "ci95_bootstrap_bajo": round(ci_bajo[par], 4),
            "ci95_bootstrap_alto": round(ci_alto[par], 4),
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    resultados = pd.concat([evaluar_metrica(m) for m in METRICAS], ignore_index=True)
    print(resultados.to_string(index=False))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(OUT_PATH, index=False)
    print(f"\n[pruebas_estadisticas] Resultados guardados en '{OUT_PATH}'.")
