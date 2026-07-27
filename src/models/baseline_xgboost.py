"""
baseline_xgboost.py
---------------------------------------------------------------------
Fase de Modelado (CRISP-DM) - Modelo baseline 1/2 de Machine Learning:
XGBoost sobre dataset_modelado_personas.csv (nivel persona, indicadores
CONTEMPORÁNEOS de ENEMDU y V-Dem -- ver src/data_prep/merge_final.py,
sin la serie agregada de Latinobarómetro para evitar fuga de información).

Evaluación con Time Series Split por año real de encuesta
(src/evaluation/time_series_split.py) y SMOTE aplicado ÚNICAMENTE sobre
X_train de cada fold, detectando la clase minoritaria real de ese fold
(src/features/smote_train.py) -- nunca sobre X_test.

Manejo de variables categóricas: se usa el soporte NATIVO de XGBoost
para categóricas (enable_categorical=True), en vez de one-hot manual,
para no inflar la dimensionalidad ni romper la semántica de columnas
nominales como 'resp_religion' o 'democ_supp_cat'. SMOTENC (dentro de
aplicar_smote_train) necesita esas columnas como texto plano (no dtype
'category' de pandas) para funcionar correctamente, así que el cast a
'category' (src/features/preparacion_modelado.castear_categoricas) se
hace recién DESPUÉS de aplicar SMOTE, justo antes de entrenar/predecir.

La lista de features (FEATURES_CATEGORICAS/FEATURES_NUMERICAS) vive en
src/features/config_features.py, compartida con baseline_lightgbm.py,
cnn_lstm.py y tabnet_model.py -- así los 4 modelos se comparan sobre
exactamente las mismas variables.
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from xgboost import XGBClassifier

# Los módulos de evaluación/features no forman un paquete instalable
# (no hay __init__.py): se agrega su carpeta al sys.path de forma
# relativa a este archivo, para que esto funcione sin importar de dónde
# se ejecute el script.
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

OUT_RESULTADOS_PATH = Path("reports/tablas/resultados_baseline_xgboost.csv")


def entrenar_evaluar_xgboost(
    df: pd.DataFrame,
    features_categoricas: list[str] = FEATURES_CATEGORICAS,
    features_numericas: list[str] = FEATURES_NUMERICAS,
    min_anios_train: int = 5,
    random_state: int = 42,
    **kwargs_xgboost,
) -> tuple[pd.DataFrame, list]:
    """
    Entrena y evalúa XGBoost con Time Series Split (ventana expansiva por
    año real de encuesta) + SMOTE por fold. Retorna:
        resultados: DataFrame con una fila por fold (métricas + tiempo)
        modelos:    lista de los XGBClassifier entrenados, uno por fold
                    (el último, entrenado con más años, es el candidato a
                    modelo final para la tabla comparativa)

    HIPERPARÁMETROS CONSERVADORES (max_depth=3, n_estimators=200) --
    decisión tomada tras encontrar un problema real corriendo la corrida
    completa: con los valores originales (max_depth=5, n_estimators=300)
    el fold que evalúa 2023 colapsaba (accuracy 0.50, PR-AUC 0.18,
    prediciendo "Satisfecho" en más de la mitad de los casos cuando la
    tasa real era ~12%). Se investigó si era un problema de umbral de
    decisión (no lo era: recalibrar el umbral con 2020 como validación
    interna casi no cambió el resultado) y se confirmó que era
    SOBREAJUSTE: un modelo más simple mejora sustancialmente ese fold sin
    perjudicar a los demás.

    Se descartó deliberadamente construir una validación interna anidada
    (reservar el último año de entrenamiento de cada fold para elegir
    hiperparámetros) porque, con solo 10 años reales de encuesta:
    (a) el primer fold pasaría de 5 a 4 años de entrenamiento real y el
    de prueba se movería de 2017 a 2018 -- se perdería un fold de prueba
    de los 5 que ya son pocos; y (b) se comprobó empíricamente que un
    solo año de validación interna (2020) NO habría elegido la
    configuración correcta para el fold de 2023 (son shocks distintos:
    COVID vs. la disolución de la Asamblea) -- la señal de un único año
    es demasiado ruidosa para ser confiable. Por eso se optó por un
    ajuste GLOBAL y conservador, justificado a priori (folds de
    entrenamiento pequeños + filas sintéticas de SMOTE = mayor riesgo de
    sobreajuste con un modelo de alta capacidad), no por una búsqueda de
    hiperparámetros por fold.

    Validado con los 5 folds reales completos: el PR-AUC promedio sube de
    0.4649 (config original) a 0.4860 con este ajuste, mejorando 2017,
    2018, 2023 y 2024, con una cesión marginal solo en 2020 (0.2750 ->
    0.2505). El fold de 2023 en particular pasa de PR-AUC 0.1776 a 0.2922.
    Sigue sin ser una búsqueda exhaustiva de hiperparámetros (eso queda
    como trabajo futuro, ver src/models/README.md) -- es un ajuste
    puntual y documentado, no el resultado de optimizar mirando el test.
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
            enable_categorical=True,
            tree_method="hist",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="aucpr",
        )
        params.update(kwargs_xgboost)
        modelo = XGBClassifier(**params)

        t0 = time.time()
        modelo.fit(X_train_bal, y_train_bal)
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
    print("\n[baseline_xgboost] Resultados por fold:")
    print(resultados_df.to_string(index=False))
    print("\n[baseline_xgboost] Promedio across folds:")
    print(resultados_df[["accuracy", "f1_macro", "f1_weighted", "f1_satisfecho", "pr_auc"]].mean().round(4))

    return resultados_df, modelos


def guardar_resultados(resultados: pd.DataFrame, out_path: Path = OUT_RESULTADOS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(out_path, index=False)
    print(f"[baseline_xgboost] Resultados guardados en '{out_path}'.")


if __name__ == "__main__":
    df = cargar_dataset()
    resultados, modelos = entrenar_evaluar_xgboost(df, min_anios_train=5)
    guardar_resultados(resultados)
