"""
baseline_logistic_regression.py
---------------------------------------------------------------------
Fase de Modelado (CRISP-DM) - Baseline estadístico adicional: Regresión
Logística, incorporado a pedido del tutor (revisión de comentarios,
Sección 2.1 / sec:mt-xgboost-lightgbm) para responder de forma empírica
-- no solo textual -- si los modelos de ensamble (XGBoost, LightGBM) y
las arquitecturas de deep learning (CNN-LSTM, TabNet) realmente superan
al clasificador lineal más simple posible.

Usa la MISMA metodología de evaluación que baseline_xgboost.py /
baseline_lightgbm.py (Time Series Split + SMOTENC aplicado únicamente
sobre X_train de cada fold) y EXACTAMENTE las mismas features
(src/features/config_features.py), para que los 5 modelos sean
comparables entre sí en la tabla final.

Manejo de variables categóricas: a diferencia de XGBoost/LightGBM (que
aceptan categóricas nativas), scikit-learn's LogisticRegression exige
entrada estrictamente numérica. Se codifica cada categórica con one-hot
(pd.get_dummies) en vez de códigos enteros (0,1,2,...) porque un código
entero implicaría un orden numérico arbitrario entre categorías
nominales (p. ej. 'resp_religion') que no existe -- el mismo argumento
por el que XGBoost/LightGBM usan su soporte nativo de categóricas en
vez de codificarlas como enteros. Las columnas dummy se generan sobre
las categorías ya alineadas entre X_train/X_test por
castear_categoricas() (src/features/preparacion_modelado.py), así que
X_train_ohe y X_test_ohe terminan con exactamente las mismas columnas
en el mismo orden.

Las variables numéricas se estandarizan (media 0, desviación 1) con
StandardScaler ajustado ÚNICAMENTE sobre X_train de cada fold -- una
regresión logística optimizada por descenso de gradiente converge mejor
y sus coeficientes son comparables en magnitud cuando las variables
están en la misma escala; XGBoost/LightGBM no lo necesitan porque son
invariantes a la escala de las features (particionan por umbrales, no
combinan linealmente).

No se aplica SHAP a este modelo: una regresión logística ya es
inherentemente transparente a través de sus propios coeficientes (el
signo y la magnitud de cada coeficiente es directamente interpretable
como dirección/fuerza del efecto marginal en la escala logit), por lo
que agregar una capa de explicabilidad post-hoc no aporta información
adicional -- a diferencia de XGBoost/LightGBM/CNN-LSTM/TabNet, que sí
son modelos de caja negra y requieren SHAP (Sección 2.7).
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    cargar_dataset, preparar_features, imputar_faltantes, castear_categoricas,
)

OUT_RESULTADOS_PATH = Path("reports/tablas/resultados_baseline_logreg.csv")


def _codificar_one_hot(
    X_train_cat: pd.DataFrame, X_test_cat: pd.DataFrame, features_categoricas: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot de las columnas ya casteadas a 'category' (mismas categorías
    en train y test, ver castear_categoricas). Al compartir dtype 'category'
    con las mismas categorías declaradas, pd.get_dummies genera el mismo
    conjunto de columnas dummy para ambos frames sin necesidad de reindexar."""
    X_train_ohe = pd.get_dummies(X_train_cat, columns=features_categoricas, drop_first=False)
    X_test_ohe = pd.get_dummies(X_test_cat, columns=features_categoricas, drop_first=False)
    X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)
    return X_train_ohe, X_test_ohe


def entrenar_evaluar_logreg(
    df: pd.DataFrame,
    features_categoricas: list[str] = FEATURES_CATEGORICAS,
    features_numericas: list[str] = FEATURES_NUMERICAS,
    min_anios_train: int = 5,
    random_state: int = 42,
    **kwargs_logreg,
) -> tuple[pd.DataFrame, list]:
    """
    Entrena y evalúa Regresión Logística con Time Series Split (ventana
    expansiva por año real de encuesta) + SMOTENC por fold. Retorna:
        resultados: DataFrame con una fila por fold (métricas + tiempo)
        modelos:    lista de los LogisticRegression entrenados, uno por
                    fold (junto con el StandardScaler y las columnas
                    one-hot usadas, necesarios para reusar el modelo)

    class_weight=None (a diferencia de la opción 'balanced' de sklearn):
    el balance de clases ya se corrige explícitamente vía SMOTENC sobre
    X_train antes de entrenar, así que agregar además class_weight
    'balanced' sobre-corregiría el desbalance dos veces.

    max_iter=1000 (por encima del default de 100): con ~30-40 columnas
    tras el one-hot y datos ya estandarizados, el solver por defecto
    (lbfgs) necesita más iteraciones para converger de forma estable en
    los folds con más filas de entrenamiento (después de SMOTENC).
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

        X_train_cat, X_test_cat = castear_categoricas(X_train_bal, X_test, features_categoricas)
        X_train_ohe, X_test_ohe = _codificar_one_hot(X_train_cat, X_test_cat, features_categoricas)

        escalador = StandardScaler()
        X_train_ohe[features_numericas] = escalador.fit_transform(X_train_ohe[features_numericas])
        X_test_ohe[features_numericas] = escalador.transform(X_test_ohe[features_numericas])

        params = dict(random_state=random_state, max_iter=1000, class_weight=None)
        params.update(kwargs_logreg)
        modelo = LogisticRegression(**params)

        t0 = time.time()
        modelo.fit(X_train_ohe, y_train_bal)
        tiempo_entrenamiento = time.time() - t0

        y_pred = modelo.predict(X_test_ohe)
        y_proba = modelo.predict_proba(X_test_ohe)[:, 1]

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
        modelos.append({"modelo": modelo, "escalador": escalador, "columnas": list(X_train_ohe.columns)})

    resultados_df = pd.DataFrame(resultados)
    print("\n[baseline_logistic_regression] Resultados por fold:")
    print(resultados_df.to_string(index=False))
    print("\n[baseline_logistic_regression] Promedio across folds:")
    print(resultados_df[["accuracy", "f1_macro", "f1_weighted", "f1_satisfecho", "pr_auc"]].mean().round(4))

    return resultados_df, modelos


def guardar_resultados(resultados: pd.DataFrame, out_path: Path = OUT_RESULTADOS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(out_path, index=False)
    print(f"[baseline_logistic_regression] Resultados guardados en '{out_path}'.")


if __name__ == "__main__":
    df = cargar_dataset()
    resultados, modelos = entrenar_evaluar_logreg(df, min_anios_train=5)
    guardar_resultados(resultados)
