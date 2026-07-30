"""
baseline_trivial.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Línea base trivial (clasificador de
clase mayoritaria), agregada a pedido del tutor y del revisor externo
tras la defensa preliminar: la Tabla 5.1 comparaba XGBoost/LightGBM/
CNN-LSTM/TabNet entre sí, pero no contra ningún punto de referencia
"sin información", así que no quedaba claro cuánto valor agregan
realmente los 4 modelos entrenados.

Este script NO entrena ningún modelo real: para cada fold del mismo
esquema de Time Series Split (generar_folds_temporales), calcula qué
tan bien le iría a un clasificador que:
  - predice siempre la clase mayoritaria observada en el propio
    X_train de ese fold (nunca mirando el test, para no filtrar
    información), y
  - como "probabilidad", usa la prevalencia de la clase "Satisfecho"
    en X_train (constante para todas las filas de test).

Con una probabilidad constante, el PR-AUC (average_precision_score)
de este clasificador es, por construcción, igual a la prevalencia real
de "Satisfecho" en el conjunto de prueba -- el valor de referencia
estándar de "azar" para una curva Precision-Recall.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))
sys.path.insert(0, str(_RAIZ_SRC / "features"))

from time_series_split import generar_folds_temporales  # noqa: E402
from config_features import COLUMNA_ANIO, COLUMNA_TARGET  # noqa: E402

RUTA_DATASET = Path("data/processed/dataset_modelado_personas.csv")
OUT_RESULTADOS_PATH = Path("reports/tablas/resultados_baseline_trivial.csv")


def calcular_baseline_trivial(
    df: pd.DataFrame,
    columna_anio: str = COLUMNA_ANIO,
    columna_target: str = COLUMNA_TARGET,
    min_anios_train: int = 5,
) -> pd.DataFrame:
    """Calcula, fold por fold, el desempeño de un clasificador de clase
    mayoritaria (sin entrenamiento real), bajo las mismas métricas y la
    misma partición temporal que XGBoost/LightGBM/CNN-LSTM/TabNet."""
    folds = generar_folds_temporales(
        df, columna_anio=columna_anio, columna_target=columna_target, min_anios_train=min_anios_train
    )
    resultados = []
    for fold in folds:
        y_train = df.loc[fold["train_idx"], columna_target]
        y_test = df.loc[fold["test_idx"], columna_target]

        prevalencia_train = y_train.mean()
        clase_mayoritaria = int(round(prevalencia_train))
        pred_test = np.full(len(y_test), clase_mayoritaria)
        proba_test = np.full(len(y_test), prevalencia_train)

        resultados.append({
            "anio_test": fold["anio_test"],
            "n_anios_train": len(fold["anios_train"]),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "accuracy": round(accuracy_score(y_test, pred_test), 4),
            "f1_macro": round(f1_score(y_test, pred_test, average="macro", zero_division=0), 4),
            "f1_weighted": round(f1_score(y_test, pred_test, average="weighted", zero_division=0), 4),
            "f1_satisfecho": round(f1_score(y_test, pred_test, pos_label=1, zero_division=0), 4),
            "pr_auc": round(average_precision_score(y_test, proba_test), 4),
            "tiempo_entrenamiento_seg": 0.0,
        })
    return pd.DataFrame(resultados)


def guardar_resultados(resultados: pd.DataFrame, out_path: Path = OUT_RESULTADOS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(out_path, index=False)
    print(f"[baseline_trivial] Resultados guardados en '{out_path}'.")


if __name__ == "__main__":
    df = pd.read_csv(RUTA_DATASET)
    resultados = calcular_baseline_trivial(df, min_anios_train=5)
    print("\n[baseline_trivial] Resultados por fold:")
    print(resultados.to_string(index=False))
    print("\n[baseline_trivial] Promedio across folds:")
    print(resultados[["accuracy", "f1_macro", "f1_weighted", "f1_satisfecho", "pr_auc"]].mean().round(4))
    guardar_resultados(resultados)
