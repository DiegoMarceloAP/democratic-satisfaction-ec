"""
ablacion_estadistica.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Prueba estadística formal sobre la
ablación de fuentes de datos (C2, observación mayor de revisores):
aplica el mismo marco de `pruebas_estadisticas.py` (Friedman + Wilcoxon
con corrección de Holm-Bonferroni, más tamaño de efecto rank-biserial e
intervalo de confianza bootstrap) a las 4 variantes de features de
`ablacion_fuentes.py` -- solo_latinobarometro, latinobarometro_vdem,
latinobarometro_enemdu, las_tres_fuentes -- en vez de a los 5 modelos.

Con 4 variantes hay 6 comparaciones pareadas posibles (en vez de 10),
así que la corrección de Holm es menos severa que en la comparación
entre los 5 modelos.

Requiere que `ablacion_fuentes.py` ya se haya ejecutado (usa los CSV
por variante que ese script guarda en reports/tablas/).

Ejecutar desde la raíz del proyecto:
    python src/evaluation/ablacion_estadistica.py
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))

from pruebas_estadisticas import (  # noqa: E402
    correlacion_rank_biserial, bootstrap_ci_diferencia_media, holm_bonferroni,
)

RUTA_TABLAS = Path("reports/tablas")
ARCHIVOS_VARIANTES = {
    "solo_latinobarometro": RUTA_TABLAS / "ablacion_xgboost_solo_latinobarometro.csv",
    "latinobarometro_vdem": RUTA_TABLAS / "ablacion_xgboost_latinobarometro_vdem.csv",
    "latinobarometro_enemdu": RUTA_TABLAS / "ablacion_xgboost_latinobarometro_enemdu.csv",
    "las_tres_fuentes": RUTA_TABLAS / "ablacion_xgboost_las_tres_fuentes.csv",
}
METRICAS = ["pr_auc", "f1_satisfecho"]
OUT_PATH = RUTA_TABLAS / "ablacion_estadistica.csv"


def cargar_valores_por_variante(metrica: str) -> dict:
    """Carga los 8 valores de `metrica`, uno por fold, para cada variante,
    ordenados por anio_test para garantizar el emparejamiento correcto."""
    valores = {}
    for nombre, ruta in ARCHIVOS_VARIANTES.items():
        df = pd.read_csv(ruta).sort_values("anio_test")
        valores[nombre] = df[metrica].to_numpy()
    return valores


def evaluar_metrica(metrica: str) -> pd.DataFrame:
    from itertools import combinations

    valores = cargar_valores_por_variante(metrica)
    variantes = list(valores.keys())

    chi2, p_friedman = friedmanchisquare(*[valores[v] for v in variantes])

    filas = []
    p_wilcoxon, efecto_r, diff_media, ci_bajo, ci_alto = {}, {}, {}, {}, {}
    for v1, v2 in combinations(variantes, 2):
        _, p = wilcoxon(valores[v1], valores[v2])
        p_wilcoxon[f"{v1} vs {v2}"] = p
        efecto_r[f"{v1} vs {v2}"] = correlacion_rank_biserial(valores[v1], valores[v2])
        media, bajo, alto = bootstrap_ci_diferencia_media(valores[v1], valores[v2])
        diff_media[f"{v1} vs {v2}"] = media
        ci_bajo[f"{v1} vs {v2}"] = bajo
        ci_alto[f"{v1} vs {v2}"] = alto

    p_holm = holm_bonferroni(p_wilcoxon)

    for par in p_wilcoxon:
        filas.append({
            "metrica": metrica,
            "comparacion": par,
            "friedman_chi2": round(chi2, 4),
            "friedman_p": round(p_friedman, 4),
            "wilcoxon_p_sin_corregir": round(p_wilcoxon[par], 4),
            "wilcoxon_p_holm": p_holm[par],
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
    print(f"\n[ablacion_estadistica] Resultados guardados en '{OUT_PATH}'.")
