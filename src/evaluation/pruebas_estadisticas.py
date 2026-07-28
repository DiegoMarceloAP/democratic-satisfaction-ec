"""
pruebas_estadisticas.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Prueba estadística formal de la
comparación entre los 4 modelos (XGBoost, LightGBM, CNN-LSTM, TabNet),
la Sección 5.2 afirmaba, solo a partir de la inspección
visual de las barras de error, que las diferencias entre modelos "no
alcanzan a ser estadísticamente concluyentes". Este script reemplaza
esa afirmación informal por una prueba formal.

Metodología:
  1. Prueba de Friedman (no paramétrica, para >2 muestras pareadas
     sobre las mismas unidades -- aquí, los 8 folds de prueba del
     Time Series Split) sobre cada métrica de interés (PR-AUC y F1
     de la clase "Satisfecho").
  2. Si Friedman resulta significativa (alfa=0.05), se calculan las
     6 comparaciones pareadas posibles entre los 4 modelos mediante
     la prueba de Wilcoxon (signed-rank), y se corrige el p-valor de
     cada una con Holm-Bonferroni para controlar la inflación del
     error tipo I por comparaciones múltiples.

Los 8 valores por modelo se toman de los CSV por fold ya generados
por cada script de modelado (resultados_baseline_xgboost.csv,
resultados_baseline_lightgbm.csv, resultados_cnn_lstm.csv,
resultados_tabnet.csv), ordenados por anio_test para que la prueba
pareada compare siempre el mismo fold entre modelos.
"""
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

RUTA_TABLAS = Path("reports/tablas")
ARCHIVOS_MODELOS = {
    "XGBoost": RUTA_TABLAS / "resultados_baseline_xgboost.csv",
    "LightGBM": RUTA_TABLAS / "resultados_baseline_lightgbm.csv",
    "CNN-LSTM": RUTA_TABLAS / "resultados_cnn_lstm.csv",
    "TabNet": RUTA_TABLAS / "resultados_tabnet.csv",
}
METRICAS = ["pr_auc", "f1_satisfecho"]
OUT_PATH = RUTA_TABLAS / "resultados_estadisticos.csv"


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
    for m1, m2 in combinations(modelos, 2):
        _, p = wilcoxon(valores[m1], valores[m2])
        p_wilcoxon[f"{m1} vs {m2}"] = p

    p_holm = holm_bonferroni(p_wilcoxon)

    for par in p_wilcoxon:
        filas.append({
            "metrica": metrica,
            "comparacion": par,
            "friedman_chi2": round(chi2, 4),
            "friedman_p": round(p_friedman, 4),
            "wilcoxon_p_sin_corregir": round(p_wilcoxon[par], 4),
            "wilcoxon_p_holm": p_holm[par],
            "significativo_holm_0.05": p_holm[par] < 0.05,
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    resultados = pd.concat([evaluar_metrica(m) for m in METRICAS], ignore_index=True)
    print(resultados.to_string(index=False))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(OUT_PATH, index=False)
    print(f"\n[pruebas_estadisticas] Resultados guardados en '{OUT_PATH}'.")
