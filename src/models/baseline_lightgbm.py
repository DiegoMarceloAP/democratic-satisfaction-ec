"""
baseline_lightgbm.py
---------------------------------------------------------------------
Fase de Modelado (CRISP-DM) - Modelo baseline 2/2 de Machine Learning:
LightGBM sobre dataset_modelado_personas.csv, con la MISMA metodología
de evaluación que baseline_xgboost.py (Time Series Split + SMOTE por
fold) y EXACTAMENTE las mismas features (src/features/config_features.py),
para que ambos baselines sean comparables entre sí en la tabla final.

Manejo de variables categóricas: LightGBM también soporta categóricas
nativas (sin one-hot manual). Se le pasan explícitamente vía el
parámetro 'categorical_feature' de .fit(), además de dejarlas en dtype
'category' de pandas (igual que en baseline_xgboost.py), para no
depender de que la autodetección por dtype funcione igual en todas las
versiones de la librería.
"""
import sys
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))
sys.path.insert(0, str(_RAIZ_SRC / "features"))

from time_series_split import generar_folds_temporales  # noqa: E402
from smote_train import aplicar_smote_train  # noqa: E402
from config_features import (  # noqa: E402
    COLUMNA_ANIO, COLUMNA_TARGET, FEATURES_CATEGORICAS, FEATURES_NUMERICAS,
)
from preparacion_modelado import (  # noqa: E402
    cargar_dataset, preparar_features, imputar_faltantes, castear_categoricas,
)

OUT_RESULTADOS_PATH = Path("reports/tablas/resultados_baseline_lightgbm.csv")


def entrenar_evaluar_lightgbm(
    df: pd.DataFrame,
    features_categoricas: list[str] = FEATURES_CATEGORICAS,
    features_numericas: list[str] = FEATURES_NUMERICAS,
    min_anios_train: int = 5,
    random_state: int = 42,
    **kwargs_lightgbm,
) -> tuple[pd.DataFrame, list]:
    """
    Entrena y evalúa LightGBM con Time Series Split + SMOTE por fold
    (misma partición y misma regla de remuestreo que baseline_xgboost.py).
    Retorna (resultados por fold, lista de modelos entrenados).

    Hiperparámetros por defecto: punto de partida razonable, no el
    resultado de una búsqueda; el ajuste fino (guiado por PR-AUC) es un
    paso posterior.
    """
    X, y = preparar_features(df, features_categoricas, features_numericas, COLUMNA_ANIO, COLUMNA_TARGET)
    folds = generar_folds_temporales(
        X.assign(**{COLUMNA_TARGET: y}),
        columna_anio=COLUMNA_ANIO,
        columna_target=COLUMNA_TARGET,
        min_anios_train=min_anios_train,
    )

    columnas_features = features_categoricas + features_numericas
    resultados = []
    modelos = []

    for fold in folds:
        X_train = X.loc[fold["train_idx"], columnas_features]
        y_train = y.loc[fold["train_idx"]]
        X_test = X.loc[fold["test_idx"], columnas_features]
        y_test = y.loc[fold["test_idx"]]

        X_train, X_test = imputar_faltantes(X_train, X_test, features_numericas)
        X_train_bal, y_train_bal = aplicar_smote_train(X_train, y_train, random_state=random_state)

        X_train_bal, X_test_cat = castear_categoricas(X_train_bal, X_test, features_categoricas)

        params = dict(
            random_state=random_state,
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            objective="binary",
            verbosity=-1,
        )
        params.update(kwargs_lightgbm)
        modelo = LGBMClassifier(**params)

        t0 = time.time()
        modelo.fit(X_train_bal, y_train_bal, categorical_feature=features_categoricas)
        tiempo_entrenamiento = time.time() - t0

        y_pred = modelo.predict(X_test_cat)
        y_proba = modelo.predict_proba(X_test_cat)[:, 1]

        resultados.append({
            "anio_test": fold["anio_test"],
            "n_anios_train": len(fold["anios_train"]),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "f1_satisfecho": round(f1_score(y_test, y_pred, pos_label=1), 4),
            "pr_auc": round(average_precision_score(y_test, y_proba), 4),
            "tiempo_entrenamiento_seg": round(tiempo_entrenamiento, 2),
        })
        modelos.append(modelo)

    resultados_df = pd.DataFrame(resultados)
    print("\n[baseline_lightgbm] Resultados por fold:")
    print(resultados_df.to_string(index=False))
    print("\n[baseline_lightgbm] Promedio across folds:")
    print(resultados_df[["accuracy", "f1_macro", "f1_weighted", "f1_satisfecho", "pr_auc"]].mean().round(4))

    return resultados_df, modelos


def guardar_resultados(resultados: pd.DataFrame, out_path: Path = OUT_RESULTADOS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(out_path, index=False)
    print(f"[baseline_lightgbm] Resultados guardados en '{out_path}'.")


if __name__ == "__main__":
    df = cargar_dataset()
    resultados, modelos = entrenar_evaluar_lightgbm(df, min_anios_train=5)
    guardar_resultados(resultados)
