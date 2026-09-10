"""
diagnostico_smotenc_temporal.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Diagnóstico puntual (C3, observación
mayor de revisores): SMOTENC busca los k vecinos más cercanos de cada
fila de la clase minoritaria en el espacio de features -- no por año --
así que en teoría podría interpolar entre encuestados de años lejanos
dentro del mismo fold de entrenamiento, produciendo combinaciones
sintéticas de contexto macro (V-Dem/ENEMDU) que nunca existieron.

Este script NO reentrena ningún modelo: replica EXACTAMENTE el mecanismo
interno de selección de vecinos de imblearn.SMOTENC (ver
_fit_resample en el código fuente de la librería) -- codificación
one-hot de categóricas con el valor "mediana de desviación estándar /
sqrt(2)" en las entradas activas (Chawla et al. 2002, Sección 6) y
NearestNeighbors sobre esa representación combinada, restringido a la
clase minoritaria de cada fold (la misma que aplicar_smote_train
detecta) -- y mide, para cada encuestado de la clase minoritaria, en
cuántos años reales difieren sus k=5 vecinos más cercanos (los que
SMOTENC usaría para generar muestras sintéticas).

No requiere GPU ni reentrenar modelos -- es un análisis directo sobre
los mismos X_train ya usados en baseline_xgboost.py, con las mismas
reglas de imputación (imputar_faltantes) y el mismo k_neighbors=5 por
defecto de SMOTENC (ver Tabla B3, "SMOTENC -- k_neighbors").
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))
sys.path.insert(0, str(_RAIZ_SRC / "features"))

from time_series_split import generar_folds_temporales  # noqa: E402
from config_features import (  # noqa: E402
    COLUMNA_ANIO, COLUMNA_TARGET, FEATURES_CATEGORICAS, FEATURES_NUMERICAS,
)
from preparacion_modelado import (  # noqa: E402
    cargar_dataset, preparar_features, imputar_faltantes,
)

RUTA_TABLAS = Path("reports/tablas")
K_NEIGHBORS = 5  # mismo valor por defecto de SMOTENC usado en todo el proyecto


def _vecinos_smotenc_replica(
    X_train: pd.DataFrame, y_train: pd.Series, anio_train: np.ndarray,
    features_categoricas: list[str], features_numericas: list[str], k: int = K_NEIGHBORS,
) -> tuple[np.ndarray, object]:
    """
    Replica el mecanismo interno de SMOTENC._fit_resample para encontrar,
    dentro de la clase minoritaria de este fold, los k vecinos más
    cercanos de cada fila -- exactamente los que la librería usaría para
    generar muestras sintéticas. Retorna (brechas_de_anio, clase_minoritaria).
    """
    balance = y_train.value_counts(normalize=True)
    clase_minoritaria = balance.idxmin()
    idx_min = np.flatnonzero(y_train.to_numpy() == clase_minoritaria)

    if len(idx_min) <= k:
        return np.array([]), clase_minoritaria  # fold degenerado, no esperado en los datos reales

    X_num_min = X_train[features_numericas].to_numpy(dtype=float)[idx_min]
    X_cat_min = X_train[features_categoricas].astype(str).to_numpy()[idx_min]
    anio_min = anio_train[idx_min]

    ohe = OneHotEncoder(handle_unknown="ignore")
    X_ohe_min = ohe.fit_transform(X_cat_min).toarray()

    # Mediana de la desviación estándar de las numéricas (misma fórmula que
    # imblearn.SMOTENC._fit_resample: csr_mean_variance_axis0 -> sqrt -> mediana).
    median_std = np.median(np.std(X_num_min, axis=0, ddof=0))
    # Las entradas activas del one-hot se reemplazan por median_std/sqrt(2)
    # (en vez de 1), para que un desacuerdo categórico aporte exactamente
    # median_std**2 a la distancia euclidiana al cuadrado -- fórmula textual
    # de imblearn (ver comentario "will be equal to the median of the
    # standard deviation as in the original paper").
    X_ohe_min_escalado = X_ohe_min * (median_std / np.sqrt(2))

    X_encoded_min = np.hstack([X_num_min, X_ohe_min_escalado])

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_encoded_min)))
    nn.fit(X_encoded_min)
    _, indices = nn.kneighbors(X_encoded_min)
    indices_vecinos = indices[:, 1:]  # columna 0 es la propia fila (distancia 0)

    brechas = np.abs(anio_min[:, None] - anio_min[indices_vecinos])
    return brechas.flatten(), clase_minoritaria


def ejecutar_diagnostico(min_anios_train: int = 5) -> pd.DataFrame:
    df = cargar_dataset()
    X, y = preparar_features(df, FEATURES_CATEGORICAS, FEATURES_NUMERICAS, COLUMNA_ANIO, COLUMNA_TARGET)
    folds = generar_folds_temporales(
        X.assign(**{COLUMNA_TARGET: y}),
        columna_anio=COLUMNA_ANIO, columna_target=COLUMNA_TARGET, min_anios_train=min_anios_train,
    )

    columnas_features = FEATURES_CATEGORICAS + FEATURES_NUMERICAS
    filas_resumen = []
    todas_las_brechas = []

    for fold in folds:
        X_train = X.loc[fold["train_idx"], [COLUMNA_ANIO] + columnas_features]
        y_train = y.loc[fold["train_idx"]]
        anio_train = X_train[COLUMNA_ANIO].to_numpy()

        X_train_feat = X_train[columnas_features]
        X_test_dummy = X_train_feat.iloc[:1]  # imputar_faltantes exige un X_test, no se usa aquí
        X_train_imp, _ = imputar_faltantes(X_train_feat, X_test_dummy, FEATURES_NUMERICAS)

        brechas, clase_min = _vecinos_smotenc_replica(
            X_train_imp, y_train, anio_train, FEATURES_CATEGORICAS, FEATURES_NUMERICAS
        )
        todas_las_brechas.append(brechas)

        anio_test = fold["anio_test"]
        filas_resumen.append({
            "anio_test": anio_test,
            "n_anios_train": len(fold["anios_train"]),
            "clase_minoritaria": "Satisfecho" if clase_min == 1 else "No Satisfecho",
            "n_filas_clase_minoritaria": int((y_train == clase_min).sum()),
            "brecha_anios_media": round(float(brechas.mean()), 3) if len(brechas) else None,
            "brecha_anios_mediana": float(np.median(brechas)) if len(brechas) else None,
            "pct_vecinos_mismo_anio": round(100 * float((brechas == 0).mean()), 2) if len(brechas) else None,
            "pct_vecinos_5_mas_anios": round(100 * float((brechas >= 5).mean()), 2) if len(brechas) else None,
            "brecha_anios_maxima": int(brechas.max()) if len(brechas) else None,
        })

    resumen = pd.DataFrame(filas_resumen)
    resumen.to_csv(RUTA_TABLAS / "diagnostico_smotenc_temporal.csv", index=False)

    brechas_totales = np.concatenate(todas_las_brechas)
    print(resumen.to_string(index=False))
    print(f"\n=== Agregado sobre los {len(brechas_totales)} pares vecino-consulta de los 8 folds ===")
    print(f"Brecha media de años: {brechas_totales.mean():.3f}")
    print(f"Brecha mediana de años: {np.median(brechas_totales):.1f}")
    print(f"% de vecinos del mismo año: {100*(brechas_totales==0).mean():.2f}%")
    print(f"% de vecinos a 5+ años de distancia: {100*(brechas_totales>=5).mean():.2f}%")
    print(f"Brecha máxima observada: {brechas_totales.max()} años")
    return resumen


if __name__ == "__main__":
    ejecutar_diagnostico()
