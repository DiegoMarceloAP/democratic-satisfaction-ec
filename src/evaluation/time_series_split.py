"""
time_series_split.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Validación cruzada temporal.

Reemplaza el K-Fold clásico por una partición de ventana expansiva
sobre los AÑOS REALES de encuesta de Latinobarómetro (no años
calendario: la serie tiene huecos -- ver latinobarometro_processing.py
y sql/README_extraccion.md), garantizando que ningún fold entrene con
datos de años posteriores al que evalúa.

Uso típico:
    folds = generar_folds_temporales(df, min_anios_train=5)
    for fold in folds:
        X_train, y_train = df.loc[fold["train_idx"], features], df.loc[fold["train_idx"], "satisfecho_democracia"]
        X_test,  y_test  = df.loc[fold["test_idx"], features],  df.loc[fold["test_idx"], "satisfecho_democracia"]
        ...
"""
import pandas as pd


def generar_folds_temporales(
    df: pd.DataFrame,
    columna_anio: str = "anio",
    columna_target: str = "satisfecho_democracia",
    min_anios_train: int = 5,
) -> list[dict]:
    """
    Genera folds de ventana expansiva: el fold i entrena con TODOS los
    años reales de encuesta anteriores al año de prueba, y evalúa sobre
    un único año siguiente (walk-forward, un año a la vez). Empieza a
    generar folds solo después de acumular 'min_anios_train' años de
    entrenamiento, para no evaluar con una base de entrenamiento
    demasiado pequeña.

    Filas con target nulo se excluyen de antemano (no tiene sentido
    usarlas ni para entrenar ni para evaluar).

    Retorna una lista de diccionarios, cada uno con:
        anios_train: lista de años usados para entrenar en este fold
        anio_test:   año usado para evaluar en este fold
        train_idx:   índice de filas de entrenamiento
        test_idx:    índice de filas de prueba
    """
    df_validos = df[df[columna_target].notna()]
    anios = sorted(df_validos[columna_anio].dropna().unique())

    if len(anios) <= min_anios_train:
        raise ValueError(
            f"Solo hay {len(anios)} años con target válido, pero min_anios_train={min_anios_train}. "
            "Baja min_anios_train o revisa que dataset_modelado_personas.csv tenga suficientes años."
        )

    folds = []
    for i in range(min_anios_train, len(anios)):
        anios_train = anios[:i]
        anio_test = anios[i]
        train_idx = df_validos.index[df_validos[columna_anio].isin(anios_train)]
        test_idx = df_validos.index[df_validos[columna_anio] == anio_test]
        folds.append({
            "anios_train": anios_train,
            "anio_test": anio_test,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })
    return folds


def reportar_balance_por_fold(
    df: pd.DataFrame,
    folds: list[dict],
    columna_target: str = "satisfecho_democracia",
) -> pd.DataFrame:
    """
    Para cada fold, calcula el % de la clase 'Satisfecho' (1) en el
    conjunto de entrenamiento y en el de prueba. Esto es lo que decide
    si SMOTENC debe aplicarse (y sobre cuál clase) en cada fold en
    particular -- ver conclusiones de notebooks/01_eda_balance_clases.ipynb.
    """
    filas = []
    for fold in folds:
        y_train = df.loc[fold["train_idx"], columna_target]
        y_test = df.loc[fold["test_idx"], columna_target]
        pct_satisfecho_train = 100 * y_train.mean()
        filas.append({
            "anio_test": fold["anio_test"],
            "n_anios_train": len(fold["anios_train"]),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "pct_satisfecho_train": round(pct_satisfecho_train, 2),
            "pct_satisfecho_test": round(100 * y_test.mean(), 2),
            "clase_minoritaria_train": "Satisfecho" if pct_satisfecho_train < 50 else "No Satisfecho",
        })
    reporte = pd.DataFrame(filas)
    if (reporte["clase_minoritaria_train"] == "No Satisfecho").any():
        anios_invertidos = reporte.loc[
            reporte["clase_minoritaria_train"] == "No Satisfecho", "anio_test"
        ].tolist()
        print(
            f"AVISO: en los folds que evalúan {anios_invertidos}, la clase minoritaria de "
            "entrenamiento es 'No Satisfecho', no 'Satisfecho' -- smote_train.py debe "
            "detectar esto automáticamente, no asumir que siempre se sobremuestrea 'Satisfecho'."
        )
    return reporte


if __name__ == "__main__":
    df = pd.read_csv("data/processed/dataset_modelado_personas.csv")
    folds = generar_folds_temporales(df, min_anios_train=5)
    reporte = reportar_balance_por_fold(df, folds)
    print(reporte.to_string(index=False))
