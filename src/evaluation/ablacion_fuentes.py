"""
ablacion_fuentes.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Ablación de fuentes de datos (C2,
observación mayor de revisores): cuantifica el aporte real de integrar
las tres fuentes (Latinobarómetro, V-Dem, ENEMDU) frente a usar solo
Latinobarómetro, entrenando XGBoost (el mejor modelo tabular por
PR-AUC, Tabla 5.1) sobre 4 variantes de features:
    1. solo_latinobarometro
    2. latinobarometro_vdem
    3. latinobarometro_enemdu
    4. las_tres_fuentes (variante ya reportada en la Tabla 5.1)

Usa exactamente los mismos 8 folds de Time Series Split y los mismos
hiperparámetros conservadores ya validados en baseline_xgboost.py
(max_depth=3, n_estimators=200, learning_rate=0.05) -- entrenar_evaluar_xgboost
ya acepta features_categoricas/features_numericas como parámetros, así
que no se modifica baseline_xgboost.py.

Todas las variables categóricas (FEATURES_CATEGORICAS) son de
Latinobarómetro y se mantienen en las 4 variantes -- la ablación es
sobre las variables NUMÉRICAS de contexto (V-Dem/ENEMDU), que es donde
se concentra el aporte de integrar fuentes adicionales de datos.
"""
import sys
from pathlib import Path

import pandas as pd

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))
sys.path.insert(0, str(_RAIZ_SRC / "features"))
sys.path.insert(0, str(_RAIZ_SRC / "models"))

from config_features import FEATURES_CATEGORICAS  # noqa: E402
from preparacion_modelado import cargar_dataset  # noqa: E402
from baseline_xgboost import entrenar_evaluar_xgboost  # noqa: E402

RUTA_TABLAS = Path("reports/tablas")

# --- Variables numéricas por fuente (ver config_features.py) ---
NUM_LATINOBAROMETRO = [
    "resp_age", "job_concern",
    "confidence_congress_alta", "confidence_judiciary_alta", "confidence_church_alta",
    "confidence_police_alta", "confidence_army_alta", "confidence_political_parties_alta",
    "goods_wash_mach_bin", "goods_car_bin", "goods_sewage_bin", "goods_hot_water_bin",
]
NUM_ENEMDU = [
    "tasa_participacion_global", "tasa_desempleo", "empleo_formal", "empleo_informal",
    "ingreso_promedio_pc", "ingreso_promedio_laboral", "gini_ingpc",
    "pobreza_ingresos", "pobreza_extrema_ingresos",
]
NUM_VDEM = [
    "v2x_polyarchy", "v2x_libdem", "v2x_partipdem", "v2x_delibdem", "v2x_egaldem",
    "v2x_freexp_altinf", "v2xel_frefair", "v2xcl_rol", "v2x_jucon", "v2xlg_legcon",
    "v2xeg_eqprotec", "v2xeg_eqaccess", "v2xeg_eqdr",
    "v2pepwrses", "v2pepwrsoc", "v2pepwrgen", "v2pepwrort", "v2pepwrgeo",
]

VARIANTES = {
    "solo_latinobarometro": NUM_LATINOBAROMETRO,
    "latinobarometro_vdem": NUM_LATINOBAROMETRO + NUM_VDEM,
    "latinobarometro_enemdu": NUM_LATINOBAROMETRO + NUM_ENEMDU,
    "las_tres_fuentes": NUM_LATINOBAROMETRO + NUM_VDEM + NUM_ENEMDU,
}


def ejecutar_ablacion(min_anios_train: int = 5) -> pd.DataFrame:
    df = cargar_dataset()
    filas_resumen = []
    for nombre_variante, features_numericas in VARIANTES.items():
        print(f"\n=== Variante: {nombre_variante} ({len(features_numericas)} numéricas + {len(FEATURES_CATEGORICAS)} categóricas) ===")
        resultados, _ = entrenar_evaluar_xgboost(
            df,
            features_categoricas=FEATURES_CATEGORICAS,
            features_numericas=features_numericas,
            min_anios_train=min_anios_train,
        )
        resultados["variante"] = nombre_variante
        resultados.to_csv(RUTA_TABLAS / f"ablacion_xgboost_{nombre_variante}.csv", index=False)

        filas_resumen.append({
            "variante": nombre_variante,
            "n_features_numericas": len(features_numericas),
            "pr_auc_media": round(resultados["pr_auc"].mean(), 4),
            "pr_auc_std": round(resultados["pr_auc"].std(), 4),
            "f1_satisfecho_media": round(resultados["f1_satisfecho"].mean(), 4),
            "f1_satisfecho_std": round(resultados["f1_satisfecho"].std(), 4),
            "accuracy_media": round(resultados["accuracy"].mean(), 4),
        })

    resumen = pd.DataFrame(filas_resumen)
    resumen.to_csv(RUTA_TABLAS / "ablacion_xgboost_resumen.csv", index=False)
    print("\n=== Resumen de la ablación ===")
    print(resumen.to_string(index=False))
    return resumen


if __name__ == "__main__":
    ejecutar_ablacion()
