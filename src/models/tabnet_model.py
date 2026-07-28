"""
tabnet_model.py
---------------------------------------------------------------------
Fase de Modelado (CRISP-DM) - Arquitectura de Deep Learning 2/2:
TabNet (Arik & Pfister, 2019) sobre dataset_modelado_personas.csv,
elegida como arquitectura nativa para datos tabulares frente a un MLP
genérico (segunda arquitectura de Deep Learning del plan de modelado).
Usa las MISMAS features que baseline_xgboost.py /
baseline_lightgbm.py / la rama estática de cnn_lstm.py
(src/features/config_features.py), para que los 4 modelos del plan se
comparen sobre exactamente las mismas variables.

Manejo de variables categóricas: a diferencia de XGBoost/LightGBM,
TabNet (como cualquier red neuronal) no tiene soporte nativo de
categóricas de tipo pandas -- usa embeddings internos, indicados vía
los parámetros 'cat_idxs' (posición de cada columna categórica dentro
de la matriz de entrada) y 'cat_dims' (cardinalidad de cada una).
Se reutiliza la misma codificación entero + bucket "desconocido" que
la rama estática de cnn_lstm.py (src/features/preparacion_modelado.
codificar_categoricas), ajustada solo con X_train de cada fold.

Orden de columnas de la matriz final: [categóricas codificadas |
numéricas escaladas], por eso cat_idxs = range(len(features_categoricas)).

Escalado: solo las columnas NUMÉRICAS se estandarizan (las categóricas
usan embeddings, no lo necesitan) -- igual que en cnn_lstm.py, ajustado
solo con X_train de cada fold, después de SMOTE (que corre en unidades
originales, igual que en los demás modelos).

NOTA DE VALIDACIÓN: la construcción de los arreglos de entrada (código
categórico + escalado + hstack + límites de cat_idxs/cat_dims) se
validó primero con datos sintéticos, reutilizando la misma preparación
de datos que baseline_xgboost.py/baseline_lightgbm.py (ya probada con
ejecución real) -- necesario porque el paquete 'torch' excedía el
tamaño soportado por el entorno de desarrollo usado para escribir este
módulo (mismo problema documentado en cnn_lstm.py). El entrenamiento
real (con 'torch'/'pytorch-tabnet' instalados) ya se ejecutó y validó
sobre el dataset completo; ver README.md de esta carpeta para los
resultados por fold.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))
sys.path.insert(0, str(_RAIZ_SRC / "features"))

from time_series_split import generar_folds_temporales  # noqa: E402
from smote_train import aplicar_smote_train  # noqa: E402
from config_features import (  # noqa: E402
    COLUMNA_ANIO, COLUMNA_TARGET, FEATURES_CATEGORICAS, FEATURES_NUMERICAS,
)
from preparacion_modelado import (  # noqa: E402
    cargar_dataset, preparar_features, imputar_faltantes, codificar_categoricas,
)

OUT_RESULTADOS_PATH = Path("reports/tablas/resultados_tabnet.csv")


def _construir_matriz_tabnet(
    X_cat_codes: np.ndarray, X_num_esc: np.ndarray
) -> np.ndarray:
    """Concatena [categóricas codificadas | numéricas escaladas] en una sola matriz float32, orden requerido por cat_idxs."""
    return np.hstack([X_cat_codes.astype(np.float32), X_num_esc.astype(np.float32)])


def entrenar_evaluar_tabnet(
    df: pd.DataFrame,
    features_categoricas: list[str] = FEATURES_CATEGORICAS,
    features_numericas: list[str] = FEATURES_NUMERICAS,
    min_anios_train: int = 5,
    random_state: int = 42,
    max_epochs: int = 100,
    batch_size: int = 256,
    virtual_batch_size: int = 64,
    cat_emb_dim: int = 4,
    **kwargs_tabnet,
) -> tuple[pd.DataFrame, list]:
    """
    Entrena y evalúa TabNet con Time Series Split + SMOTE por fold
    (misma partición y misma regla de remuestreo que los demás modelos).
    Retorna (resultados por fold, lista de modelos entrenados).

    No se usa 'eval_set' de TabNet (que habilitaría early stopping):
    como el único conjunto disponible aparte de train es el propio test
    del fold, usarlo para early stopping filtraría información de
    evaluación hacia la selección del modelo (mismo criterio aplicado
    en cnn_lstm.py: número de épocas fijo, no ajustado mirando el test).
    """
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = preparar_features(df, features_categoricas, features_numericas, COLUMNA_ANIO, COLUMNA_TARGET)
    folds = generar_folds_temporales(
        X.assign(**{COLUMNA_TARGET: y}),
        columna_anio=COLUMNA_ANIO, columna_target=COLUMNA_TARGET, min_anios_train=min_anios_train,
    )

    columnas_features = features_categoricas + features_numericas
    cat_idxs = list(range(len(features_categoricas)))
    resultados = []
    modelos = []

    for fold in folds:
        X_train = X.loc[fold["train_idx"], columnas_features]
        y_train = y.loc[fold["train_idx"]]
        X_test = X.loc[fold["test_idx"], columnas_features]
        y_test = y.loc[fold["test_idx"]]

        X_train, X_test = imputar_faltantes(X_train, X_test, features_numericas)
        X_train_bal, y_train_bal = aplicar_smote_train(X_train, y_train, random_state=random_state)

        cod_train, cod_test, cardinalidades = codificar_categoricas(
            X_train_bal[features_categoricas], X_test[features_categoricas], features_categoricas
        )
        cat_dims = [cardinalidades[c] for c in features_categoricas]

        scaler = StandardScaler().fit(X_train_bal[features_numericas])
        num_train_esc = scaler.transform(X_train_bal[features_numericas])
        num_test_esc = scaler.transform(X_test[features_numericas])

        X_train_final = _construir_matriz_tabnet(cod_train, num_train_esc)
        X_test_final = _construir_matriz_tabnet(cod_test, num_test_esc)
        y_train_final = y_train_bal.to_numpy().astype(np.int64)
        y_test_final = y_test.to_numpy().astype(np.int64)

        params = dict(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_d=8, n_a=8, n_steps=3, gamma=1.3,
            seed=random_state,
            device_name=device_name,
            verbose=0,
        )
        params.update(kwargs_tabnet)
        modelo = TabNetClassifier(**params)

        t0 = time.time()
        modelo.fit(
            X_train_final, y_train_final,
            max_epochs=max_epochs, batch_size=batch_size, virtual_batch_size=virtual_batch_size,
        )
        tiempo_entrenamiento = time.time() - t0

        proba_test = modelo.predict_proba(X_test_final)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)

        resultados.append({
            "anio_test": fold["anio_test"],
            "n_anios_train": len(fold["anios_train"]),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "accuracy": round(accuracy_score(y_test_final, pred_test), 4),
            "f1_macro": round(f1_score(y_test_final, pred_test, average="macro"), 4),
            "f1_weighted": round(f1_score(y_test_final, pred_test, average="weighted"), 4),
            "f1_satisfecho": round(f1_score(y_test_final, pred_test, pos_label=1), 4),
            "pr_auc": round(average_precision_score(y_test_final, proba_test), 4),
            "tiempo_entrenamiento_seg": round(tiempo_entrenamiento, 2),
        })
        modelos.append(modelo)

    resultados_df = pd.DataFrame(resultados)
    print("\n[tabnet_model] Resultados por fold:")
    print(resultados_df.to_string(index=False))
    print("\n[tabnet_model] Promedio across folds:")
    print(resultados_df[["accuracy", "f1_macro", "f1_weighted", "f1_satisfecho", "pr_auc"]].mean().round(4))

    return resultados_df, modelos


def guardar_resultados(resultados: pd.DataFrame, out_path: Path = OUT_RESULTADOS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(out_path, index=False)
    print(f"[tabnet_model] Resultados guardados en '{out_path}'.")


if __name__ == "__main__":
    df = cargar_dataset()
    resultados, modelos = entrenar_evaluar_tabnet(df, min_anios_train=5)
    guardar_resultados(resultados)
