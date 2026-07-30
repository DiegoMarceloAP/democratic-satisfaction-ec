"""
smote_train.py
---------------------------------------------------------------------
Fase de Modelado (CRISP-DM) - preparación de features: SMOTENC aplicado
ÚNICAMENTE sobre X_train, después del split, y de forma independiente
POR FOLD de Time Series Split (ver src/evaluation/time_series_split.py
y las conclusiones de notebooks/01_eda_balance_clases.ipynb: el balance
de clases cambia según el rango de años de cada fold, así que no existe
una única proporción de remuestreo válida para todo el dataset).

No asume que la clase minoritaria siempre es "Satisfecho" (1): la
detecta automáticamente en cada fold y avisa si en algún fold la clase
minoritaria real fuera "No Satisfecho" (0), para no sobremuestrear la
clase equivocada por accidente.

Este módulo usa exclusivamente SMOTENC (la variante de SMOTE para
datos mixtos, Chawla et al. 2002, Sección 6), nunca el SMOTE original:
el conjunto de features de esta tesis (src/features/config_features.py)
siempre incluye variables categóricas (resp_sex, resp_education,
democ_supp_cat, etc.), así que el caso "todas las columnas son
numéricas" -- el único en el que tendría sentido usar SMOTE estándar --
nunca ocurre en el pipeline real. Por eso `aplicar_smote_train` no
implementa un SMOTE genérico como alternativa: si algún día se llamara
con un conjunto de features sin ninguna categórica, es una señal de que
se pasó el conjunto de features equivocado, y la función lo señala con
un error explícito en vez de degradar silenciosamente a un SMOTE
distinto del que describe y justifica la tesis (Sección~2.4/4.6).
"""
import pandas as pd
from imblearn.over_sampling import SMOTENC


def _detectar_columnas_categoricas(X: pd.DataFrame) -> list[int]:
    """Índices posicionales de columnas no numéricas (para SMOTENC)."""
    return [i for i, dtype in enumerate(X.dtypes) if not pd.api.types.is_numeric_dtype(dtype)]


def aplicar_smote_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    umbral_aviso_balanceado: float = 0.45,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Aplica SMOTENC sobre X_train/y_train de UN fold. y_train debe ser
    binario (0/1) y sin nulos -- filtrar el target nulo antes de llamar
    a esta función (nunca sobre el target, eso ya se hace al construir
    los folds). X_train tampoco debe tener nulos (imputar antes de
    llamar a esto).

    Lanza ValueError si X_train no tiene ninguna columna categórica: el
    conjunto de features de esta tesis siempre tiene al menos una (ver
    docstring del módulo), así que ese caso indica un error de features
    pasadas por el llamador, no un escenario válido que deba resolverse
    con un SMOTE genérico.
    """
    if y_train.isna().any():
        raise ValueError("y_train tiene valores nulos -- filtrar antes de aplicar SMOTENC.")

    balance = y_train.value_counts(normalize=True)
    clase_minoritaria = balance.idxmin()
    pct_minoritaria = balance.min()

    print(
        f"Balance de este fold -- clase 1 (Satisfecho): {100*balance.get(1, 0):.2f}%, "
        f"clase 0 (No Satisfecho): {100*balance.get(0, 0):.2f}%. "
        f"Clase minoritaria detectada: {'Satisfecho' if clase_minoritaria == 1 else 'No Satisfecho'}."
    )
    if pct_minoritaria >= umbral_aviso_balanceado:
        print(
            f"AVISO: este fold está casi balanceado (clase minoritaria = {100*pct_minoritaria:.1f}%). "
            "SMOTENC va a generar pocas muestras sintéticas; considera si de verdad hace falta en este fold."
        )

    columnas_categoricas = _detectar_columnas_categoricas(X_train)
    if not columnas_categoricas:
        raise ValueError(
            "X_train no tiene ninguna columna categórica -- el conjunto de features de esta "
            "tesis siempre incluye al menos una (config_features.py). Revisa qué features se "
            "están pasando: esta función es SMOTENC-only por diseño, no cae a un SMOTE genérico."
        )
    print(f"Columnas categóricas detectadas (usando SMOTENC): {[X_train.columns[i] for i in columnas_categoricas]}")
    sampler = SMOTENC(categorical_features=columnas_categoricas, random_state=random_state)

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f"Filas antes de SMOTENC: {len(X_train)}  ->  después: {len(X_resampled)}")
    return X_resampled, y_resampled


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src/evaluation")
    from time_series_split import generar_folds_temporales

    df = pd.read_csv("data/processed/dataset_modelado_personas.csv")
    features = ["resp_age", "tasa_desempleo", "gini_ingpc", "econ_situation_cat", "ideologia_cat"]

    folds = generar_folds_temporales(df, min_anios_train=5)
    fold = folds[0]  # ejemplo: primer fold
    X_train = df.loc[fold["train_idx"], features].copy()
    y_train = df.loc[fold["train_idx"], "satisfecho_democracia"].astype(int)

    # SMOTENC no acepta NaN -- imputación simple solo para este ejemplo;
    # el pipeline real de Fase 2/features debe definir la imputación final.
    for c in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            X_train[c] = X_train[c].fillna(X_train[c].median())
        else:
            X_train[c] = X_train[c].fillna("__faltante__")

    print(f"\n--- Fold de ejemplo: entrena con {fold['anios_train']}, evalúa {fold['anio_test']} ---")
    X_res, y_res = aplicar_smote_train(X_train, y_train)
    print(f"Balance después de SMOTENC: {y_res.value_counts(normalize=True).round(3).to_dict()}")
