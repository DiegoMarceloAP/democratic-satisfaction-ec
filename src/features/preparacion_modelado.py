"""
preparacion_modelado.py
---------------------------------------------------------------------
Funciones de preparación de datos COMUNES a los modelos no secuenciales
del plan de modelado (XGBoost, LightGBM, TabNet): cargar el dataset a
nivel persona, seleccionar/validar columnas de features (ver
src/features/config_features.py), e imputar valores faltantes de forma
segura para Time Series Split (estadísticas calculadas solo con
X_train de cada fold, nunca con X_test -- evita fuga de información
entre folds).

Se separa de cada baseline_*.py para que los 4 modelos del plan usen
EXACTAMENTE la misma lógica de preparación; solo cambia la arquitectura
del modelo en sí.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_PATH = Path("data/processed/dataset_modelado_personas.csv")


def cargar_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path}'. Corre primero src/data_prep/merge_final.py (Fase 2)."
        )
    return pd.read_csv(path)


def preparar_features(
    df: pd.DataFrame,
    features_categoricas: list[str],
    features_numericas: list[str],
    columna_anio: str,
    columna_target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Selecciona las columnas de features (descarta el resto: identificadores,
    'democ_satis' que es la fuente directa del target, columnas crudas ya
    recodificadas en sus versiones '_cat'/'_bin'/'_alta', etc.) y filtra las
    filas con target nulo (respuesta no válida en democ_satis -- no se
    imputa el target, se descarta, igual que en el resto del pipeline).

    Las columnas categóricas se dejan en dtype texto (no 'category' todavía)
    para que SMOTENC (src/features/smote_train.py) las detecte y procese
    correctamente; el cast a 'category' se hace más adelante, justo antes
    de entrenar cada modelo específico.
    """
    faltantes = [c for c in features_categoricas + features_numericas if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"Columnas de features no encontradas en el dataset: {faltantes}. "
            "Revisar merge_final.py / la extracción real de Superset."
        )

    df_validos = df[df[columna_target].notna()].copy()

    X = df_validos[[columna_anio] + features_categoricas + features_numericas].copy()
    for c in features_categoricas:
        X[c] = X[c].astype(str).replace({"nan": np.nan, "<NA>": np.nan})
    for c in features_numericas:
        # Fuerza dtype float explícito: algunas columnas numéricas/binarias
        # (ej. confidence_*_alta, construidas con .isin(...).where(...) en
        # latinobarometro_processing.py) se guardan en el CSV como
        # True/False/vacío y, al releerlas con pd.read_csv, quedan en dtype
        # "object" (no numérico) en vez de float -- lo cual rompe SMOTE
        # (las trata como categóricas) y a XGBoost/LightGBM/TabNet (rechazan
        # columnas dtype object). astype(float) convierte True/False/NaN a
        # 1.0/0.0/NaN de forma segura sin alterar los valores ya numéricos.
        # Detectado corriendo el pipeline contra el dataset real completo.
        X[c] = X[c].astype(float)
    y = df_validos[columna_target].astype(int)

    print(
        f"[preparacion_modelado] {len(X)} filas con target válido, "
        f"{len(features_categoricas)} features categóricas + {len(features_numericas)} numéricas."
    )
    return X, y


def muestrear_estratificado(
    df: pd.DataFrame,
    fraccion: float,
    columna_anio: str = "anio",
    columna_target: str = "satisfecho_democracia",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Muestra estratificada por (año, clase del target) -- para
    SAMPLING_MODE=True (ver notebooks/00_parametros_globales.ipynb):
    reduce el volumen de filas para poder entrenar/probar rápido en una
    laptop de capacidad media, sin distorsionar ni la cobertura de años
    (se preservan TODOS los años, solo con menos filas cada uno) ni el
    balance de clases por año (que ya de por sí es volátil -- ver
    notebooks/01_eda_balance_clases.ipynb).

    Las filas con target nulo (respuesta inválida, se descartan de
    todas formas en preparar_features) se muestrean aparte con la misma
    fracción pero SIN estratificar por clase (no tienen clase) -- esto
    evita un error conocido de pandas al usar groupby(dropna=False)
    seguido de .sample() con NaN en una columna de agrupación.
    """
    df_valido = df[df[columna_target].notna()]
    df_invalido = df[df[columna_target].isna()]

    muestra_valida = (
        df_valido.groupby([columna_anio, columna_target], group_keys=False)
        .sample(frac=fraccion, random_state=random_state)
    )
    muestra_invalida = (
        df_invalido.sample(frac=fraccion, random_state=random_state) if len(df_invalido) else df_invalido
    )
    return pd.concat([muestra_valida, muestra_invalida], ignore_index=True)


def imputar_faltantes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features_numericas: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputación simple (mediana para numéricas, categoría constante
    '__faltante__' para categóricas), con las estadísticas calculadas
    SOLO sobre X_train de este fold y aplicadas a X_train y X_test --
    necesaria antes de SMOTE (SMOTE/SMOTENC no aceptan NaN) y consistente
    con la regla de no fuga de información entre folds.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    for c in X_train.columns:
        if c in features_numericas:
            mediana = X_train[c].median()
            X_train[c] = X_train[c].fillna(mediana)
            X_test[c] = X_test[c].fillna(mediana)
        else:
            X_train[c] = X_train[c].fillna("__faltante__")
            X_test[c] = X_test[c].fillna("__faltante__")
    return X_train, X_test


def castear_categoricas(
    X_train: pd.DataFrame, X_test: pd.DataFrame, features_categoricas: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cast a dtype 'category' de pandas para AMBOS X_train y X_test, usando
    la MISMA lista de categorías (unión de los valores únicos vistos en
    train y en test) para cada columna -- lo requieren XGBoost
    (enable_categorical=True) y LightGBM (auto-detección de categóricas).

    Es imprescindible declarar categorías IDÉNTICAS en ambos frames:
    corriendo el pipeline contra el dataset real completo se detectó que
    XGBoost RECHAZA en 'predict' cualquier categoría de X_test que no
    haya sido declarada en el dtype de X_train (error real: "Found a
    category not in the training set"), algo que ocurre con frecuencia
    aquí porque train y test son años distintos (Time Series Split) y
    una columna puede no tener NINGÚN valor faltante (por lo tanto nunca
    genera la categoría '__faltante__') en los años de entrenamiento de
    un fold, pero sí tenerlos en el año de prueba -- o viceversa con
    categorías genuinamente raras. Antes este cast se hacía de forma
    INDEPENDIENTE por frame (cada uno infería sus propias categorías de
    sus propios valores), lo cual fallaba exactamente en ese escenario.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    for c in features_categoricas:
        categorias = sorted(set(X_train[c].astype(str).unique()) | set(X_test[c].astype(str).unique()))
        X_train[c] = pd.Categorical(X_train[c].astype(str), categories=categorias)
        X_test[c] = pd.Categorical(X_test[c].astype(str), categories=categorias)
    return X_train, X_test


def codificar_categoricas(
    X_train_cat: pd.DataFrame, X_test_cat: pd.DataFrame, columnas_cat: list[str]
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Codifica cada columna categórica a enteros [0..n_categorias-1] según
    las categorías vistas en ESTE fold de entrenamiento (después de
    SMOTE, si aplica). Categorías de prueba no vistas en entrenamiento
    se mapean a un índice adicional reservado ('desconocido'), para que
    el modelo (embeddings de CNN-LSTM o de TabNet) tenga una
    representación aprendible en vez de fallar con una categoría nunca
    vista.

    Retorna (codigos_train, codigos_test, cardinalidades), donde
    cardinalidades[col] = n_categorias_vistas + 1 (por el índice de
    'desconocido') -- usado para dimensionar cada nn.Embedding
    (cnn_lstm.py) o cat_dims (tabnet_model.py).
    """
    codigos_train = np.zeros((len(X_train_cat), len(columnas_cat)), dtype=np.int64)
    codigos_test = np.zeros((len(X_test_cat), len(columnas_cat)), dtype=np.int64)
    cardinalidades = {}
    for j, col in enumerate(columnas_cat):
        categorias = pd.Index(X_train_cat[col].astype(str).unique())
        mapa = {cat: i for i, cat in enumerate(categorias)}
        idx_desconocido = len(categorias)
        codigos_train[:, j] = X_train_cat[col].astype(str).map(mapa).to_numpy()
        codigos_test[:, j] = (
            X_test_cat[col].astype(str).map(mapa).fillna(idx_desconocido).astype(int).to_numpy()
        )
        cardinalidades[col] = idx_desconocido + 1
    return codigos_train, codigos_test, cardinalidades
