"""
shap_explicabilidad.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - IA Explicable (XAI), marco DUAL:
    - GLOBAL: impacto agregado de las variables socioeconómicas y
      políticas sobre la predicción, para cada fold/año evaluado.
    - LOCAL: auditoría de predicciones individuales (un encuestado, un
      año, un perfil sociodemográfico particular).

Motores de SHAP usados según el tipo de modelo (pedido explícito de la
propuesta de tesis):
    - TreeSHAP (shap.TreeExplainer) para XGBoost y LightGBM: exacto y
      rápido, no necesita background ni muestreo -- explota la
      estructura de árbol directamente.
    - KernelSHAP (shap.KernelExplainer) para CNN-LSTM y TabNet: ambos
      son redes neuronales de caja negra para SHAP, así que se necesita
      un "background" (referencia de qué es un valor "normal" de cada
      variable) y una función de predicción numpy -> numpy.

DECISIÓN DE DISEÑO -- reconstrucción EXACTA de cada fold: las funciones
entrenar_evaluar_* (baseline_xgboost.py, baseline_lightgbm.py,
tabnet_model.py, cnn_lstm.py) devuelven los modelos ya entrenados, pero
NO devuelven el X_test procesado de cada fold (para no inflar la firma
de esas funciones). Este módulo reconstruye ese X_test repitiendo
EXACTAMENTE los mismos pasos (imputación con medianas/constante de
X_train, SMOTENC con la misma random_state, codificación categórica) que
usó cada script de entrenamiento -- así las explicaciones se calculan
sobre los datos reales, en las mismas unidades/códigos que el modelo
aprendió. Si algún día cambia el preprocesamiento de esos scripts, hay
que reflejar el cambio aquí también: este módulo no comparte una única
fuente de verdad con los scripts de entrenamiento, así que es un punto
de mantenimiento a vigilar si el pipeline de preparación cambia.

DECISIÓN DE DISEÑO -- nunca explicar filas sintéticas de SMOTENC: todas
las funciones de este módulo calculan SHAP únicamente sobre el X_test
de cada fold (encuestados reales nunca vistos en entrenamiento), jamás
sobre X_train_bal (que incluye filas sintéticas). Explicar una fila
sintética no tendría sentido sustantivo para la tesis -- no es una
persona real.

DECISIÓN DE DISEÑO -- SHAP por fold, no un único resumen global pooled:
como ya se documentó en metricas.py, el desempeño de los 4 modelos varía
mucho entre folds (std de PR-AUC entre 0.17 y 0.25 entre años) -- señal
de que el contexto de cada año importa. Promediar SHAP de todos los
folds en un solo resumen escondería esa heterogeneidad. Por defecto
'indices_folds=None' calcula SHAP para TODOS los folds por separado
(un CSV/PNG por año de prueba), no un pool.

DECISIÓN DE DISEÑO -- aplanado del CNN-LSTM para KernelSHAP: el CNN-LSTM
recibe 4 tensores (ventana secuencial, máscara, categóricas estáticas,
numéricas estáticas), pero KernelSHAP necesita una función
matriz_2d -> vector. Se aplana todo en una sola fila por persona, con la
MISMA convención de nombres de columna que ya usa cnn_lstm.py para su
propio aplanado de SMOTENC ('seq_t{t}_{variable}', 'mask_t{t}') --
reutilizando exactamente la misma idea que rezagos_macro.py: exponer la
ventana temporal como columnas planas para poder analizarla con
herramientas que esperan un vector, no un tensor.

NOTA DE VALIDACIÓN: la ruta de TreeSHAP (XGBoost/LightGBM) y la ruta de
KernelSHAP (CNN-LSTM/TabNet) están ambas validadas ejecutándolas de
verdad contra dataset_modelado_personas.csv real y los 4 modelos ya
entrenados (ver src/evaluation/README.md para el detalle de la corrida
completa, incluyendo la corrección de una inestabilidad numérica
puntual en KernelSHAP). Los imports de torch/cnn_lstm/tabnet_model son
LAZY (dentro de cada función, no al principio del archivo) porque
'torch'/'pytorch-tabnet' no estaban disponibles en el entorno usado
para escribir este módulo inicialmente -- así la parte de árboles podía
probarse sin depender de esas librerías; con ellas instaladas, todo el
módulo funciona igual desde un único entorno.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))
sys.path.insert(0, str(_RAIZ_SRC / "features"))
sys.path.insert(0, str(_RAIZ_SRC / "models"))

from time_series_split import generar_folds_temporales  # noqa: E402
from smote_train import aplicar_smote_train  # noqa: E402
from config_features import (  # noqa: E402
    COLUMNA_ANIO, COLUMNA_TARGET, FEATURES_CATEGORICAS, FEATURES_NUMERICAS,
)
from preparacion_modelado import (  # noqa: E402
    cargar_dataset, preparar_features, imputar_faltantes, castear_categoricas, codificar_categoricas,
)

RUTA_TABLAS = Path("reports/tablas")
RUTA_FIGURES = Path("reports/figures")
VENTANA_TEMPORAL_ANIOS = 3  # debe coincidir con notebooks/00_parametros_globales.ipynb y cnn_lstm.py


def _parchear_shap_xgboost_base_score() -> None:
    """
    Shim de compatibilidad -- BUG REAL encontrado corriendo esto contra
    XGBoost real (xgboost 3.2.0 + shap 0.49.1): xgboost serializa
    'base_score' en su volcado interno UBJSON como una cadena tipo
    '[5.05E-1]' (formato de arreglo de 1 elemento en vez de un número
    plano), y shap.TreeExplainer intenta convertirla directamente con
    float(), lo que lanza 'ValueError: could not convert string to
    float'. No depende de los datos de la tesis -- se reprodujo también
    con un modelo XGBoost sintético mínimo, así que es una
    incompatibilidad general de versiones, no algo específico de este
    proyecto.

    Se parchea 'decode_ubjson_buffer' (la función que shap usa para leer
    el volcado binario del modelo) para quitar los corchetes ANTES de que
    shap intente convertir el valor -- no cambia ningún resultado
    numérico, solo corrige el formato de parsing. Se aplica una sola vez
    por proceso (idempotente) y NO afecta a LightGBM (que no tiene este
    problema).
    """
    import shap.explainers._tree as _shap_tree

    if getattr(_shap_tree, "_parche_base_score_xgboost_aplicado", False):
        return

    _decode_original = _shap_tree.decode_ubjson_buffer

    def _decode_parcheado(fd):
        jmodel = _decode_original(fd)
        try:
            parametros = jmodel["learner"]["learner_model_param"]
            base_score = parametros.get("base_score")
            if isinstance(base_score, str):
                parametros["base_score"] = base_score.strip("[]")
        except (KeyError, TypeError):
            pass
        return jmodel

    _shap_tree.decode_ubjson_buffer = _decode_parcheado
    _shap_tree._parche_base_score_xgboost_aplicado = True


# ======================================================================
# Utilidades genéricas de SHAP (independientes del tipo de modelo)
# ======================================================================

UMBRAL_SHAP_IMPLAUSIBLE = 10.0  # ver _neutralizar_columnas_implausibles


def _neutralizar_columnas_implausibles(shap_values: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    BUG REAL encontrado corriendo KernelSHAP contra el CNN-LSTM real
    (fold 2017, el más chico de los 5): aparecieron valores de SHAP del
    orden de 1e10-1e11 en 1-2 columnas, mientras el resto del mismo fold
    tenía valores normales (0.01-0.09). Como la predicción explicada es
    una probabilidad en [0,1], eso es matemáticamente imposible -- es un
    artefacto numérico de la regresión ponderada interna de KernelSHAP
    (que queda mal condicionada cuando hay columnas casi colineales en el
    background, más probable en folds con pocos años reales de
    entrenamiento, donde la ventana temporal del CNN-LSTM tiene muy poca
    variedad real entre los encuestados de ese mismo año).

    IMPORTANTE: subir 'nsamples' (100 -> 800) NO fue suficiente por sí
    solo -- se probó y el problema reapareció en OTRAS columnas distintas
    (mismo patrón: 1-2 columnas con valores casi idénticos entre sí y
    absurdamente altos). Por eso, en vez de solo avisar, esta función
    NEUTRALIZA (pone en 0) cualquier columna cuya |SHAP| promedio supere
    UMBRAL_SHAP_IMPLAUSIBLE, para que un fold puntualmente inestable no
    contamine el CSV/gráfico final -- e imprime un AVISO explícito
    listando qué columnas se descartaron y por qué, para que quede
    trazable y no pase desapercibido.
    """
    importancia = np.abs(shap_values).mean(axis=0)
    idx_culpables = np.where(importancia > UMBRAL_SHAP_IMPLAUSIBLE)[0]
    if len(idx_culpables) > 0:
        detalle = ", ".join(f"{feature_names[i]}={importancia[i]:.3e}" for i in idx_culpables)
        print(
            f"AVISO: {len(idx_culpables)} columna(s) con SHAP implausible (> {UMBRAL_SHAP_IMPLAUSIBLE}, "
            f"una probabilidad no puede depender tanto de una sola variable) -- se NEUTRALIZARON "
            f"(puestas en 0) para no contaminar el resumen; probable inestabilidad numérica de "
            f"KernelSHAP en este fold específico, no un hallazgo real. Columnas descartadas: {detalle}"
        )
        shap_values = shap_values.copy()
        shap_values[:, idx_culpables] = 0.0
    return shap_values


def _submuestrear_test_para_kernelshap(
    X_test: pd.DataFrame, y_test: pd.Series, max_filas: int | None, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """
    OPTIMIZACIÓN DE COSTO (a pedido del tutor, especialmente relevante para
    CNN-LSTM): KernelSHAP evalúa el modelo 'nsamples' veces POR CADA fila
    que se explica, así que el costo total escala linealmente con el
    tamaño del fold de prueba explicado -- con folds de ~1200 encuestados
    y nsamples=800 (CNN-LSTM) eso son ~960,000 pasadas del modelo por
    fold, la mayor parte del tiempo de toda la fase de Evaluación/XAI.

    Si el fold tiene más de 'max_filas' encuestados, toma una submuestra
    ALEATORIA ESTRATIFICADA por la clase real (y_test) antes de calcular
    SHAP. Esto NO afecta nsamples (que se mantiene en el valor ya validado
    para evitar la inestabilidad numérica documentada en
    explicar_cnn_lstm) ni el entrenamiento del modelo -- solo reduce
    CUÁNTOS encuestados del fold se usan para estimar el resumen GLOBAL
    (promedio de |SHAP|), que es un estadístico agregado: unos cientos de
    filas ya lo estabilizan (ley de los grandes números), sin necesidad de
    explicar cada encuestado real del fold. No se aplica a TreeSHAP
    (explicar_arbol), que es exacto y no necesita muestreo.

    Si 'max_filas' es None, se explica el fold completo (comportamiento
    anterior, sin recorte).
    """
    if max_filas is None or len(X_test) <= max_filas:
        return X_test, y_test
    estratificar = y_test if y_test.nunique() > 1 else None
    idx_sub, _ = train_test_split(
        np.arange(len(X_test)), train_size=max_filas, random_state=random_state, stratify=estratificar,
    )
    idx_sub = np.sort(idx_sub)
    return X_test.iloc[idx_sub].reset_index(drop=True), y_test.iloc[idx_sub].reset_index(drop=True)


def resumen_global_shap(shap_values: np.ndarray, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    """
    Importancia GLOBAL: promedio del valor absoluto de SHAP por feature
    (el "impacto macro" que pide la propuesta de tesis), ordenado
    descendente. Se usa |SHAP| (no el promedio con signo) porque una
    variable puede empujar la predicción hacia arriba en unos
    encuestados y hacia abajo en otros -- promediar con signo cancelaría
    ese impacto real en vez de medirlo.

    Antes de agregar, pasa por _neutralizar_columnas_implausibles: como la
    predicción explicada es una probabilidad en [0,1], ningún valor de
    SHAP debería salir por varios órdenes de magnitud fuera de ese rango
    -- si aparece, se neutraliza esa columna y se avisa explícitamente
    (ver docstring de esa función), en vez de dejar que un artefacto
    numérico se cuele en el reporte final.
    """
    shap_values = _neutralizar_columnas_implausibles(shap_values, feature_names)
    importancia = np.abs(shap_values).mean(axis=0)
    resumen = pd.DataFrame({"feature": feature_names, "importancia_media_abs_shap": importancia})
    return resumen.sort_values("importancia_media_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def explicar_perfil_local(
    shap_values: np.ndarray, X_legible: pd.DataFrame, feature_names: list[str], idx: int, top_n: int = 10
) -> pd.DataFrame:
    """
    Explicación LOCAL de un encuestado puntual (una fila de X_legible,
    identificada por su posición 'idx' dentro del fold de prueba): qué
    variables empujaron su predicción hacia "Satisfecho" (shap_value>0)
    o hacia "No Satisfecho" (shap_value<0), ordenadas por magnitud de
    impacto. Útil para auditar un perfil sociodemográfico o provincia
    específica en un año particular, como pide la propuesta.
    """
    fila_shap = shap_values[idx]
    valores_observados = X_legible.iloc[idx][feature_names].to_numpy()
    resumen = pd.DataFrame({
        "feature": feature_names,
        "valor_observado": valores_observados,
        "shap_value": fila_shap,
    })
    resumen["impacto_abs"] = resumen["shap_value"].abs()
    return resumen.sort_values("impacto_abs", ascending=False).head(top_n).drop(columns="impacto_abs").reset_index(drop=True)


def guardar_resumen_shap(resumen: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(out_path, index=False)
    print(f"[shap_explicabilidad] Resumen guardado en '{out_path}'.")


def graficar_resumen_shap(resumen: pd.DataFrame, titulo: str, out_path: Path) -> None:
    """Gráfico de barras horizontal de importancia SHAP global -- mismo estilo que metricas.graficar_comparacion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(resumen) + 1.5))
    orden = resumen.iloc[::-1]
    ax.barh(orden["feature"], orden["importancia_media_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Importancia media |SHAP|")
    ax.set_title(titulo)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[shap_explicabilidad] Gráfico guardado en '{out_path}'.")


def _generar_folds(
    df: pd.DataFrame, features_categoricas: list[str], features_numericas: list[str], min_anios_train: int
):
    X, y = preparar_features(df, features_categoricas, features_numericas, COLUMNA_ANIO, COLUMNA_TARGET)
    folds = generar_folds_temporales(
        X.assign(**{COLUMNA_TARGET: y}),
        columna_anio=COLUMNA_ANIO, columna_target=COLUMNA_TARGET, min_anios_train=min_anios_train,
    )
    return X, y, folds


# ======================================================================
# TreeSHAP -- XGBoost / LightGBM (idéntico para ambos: son ensambles de árboles)
# ======================================================================

def _preparar_fold_arbol(X, y, fold, features_categoricas, features_numericas, random_state):
    """Reconstruye X_test EXACTAMENTE como baseline_xgboost.py/baseline_lightgbm.py (misma imputación + categorías unificadas train/test)."""
    columnas_features = features_categoricas + features_numericas
    X_train = X.loc[fold["train_idx"], columnas_features]
    y_train = y.loc[fold["train_idx"]]
    X_test = X.loc[fold["test_idx"], columnas_features]
    y_test = y.loc[fold["test_idx"]]

    X_train, X_test = imputar_faltantes(X_train, X_test, features_numericas)
    X_train_bal, y_train_bal = aplicar_smote_train(X_train, y_train, random_state=random_state)
    _, X_test_cat = castear_categoricas(X_train_bal, X_test, features_categoricas)
    return X_test_cat, y_test


def explicar_arbol(
    df: pd.DataFrame,
    modelos: list,
    nombre_modelo: str,
    features_categoricas: list[str] = FEATURES_CATEGORICAS,
    features_numericas: list[str] = FEATURES_NUMERICAS,
    min_anios_train: int = 5,
    random_state: int = 42,
    indices_folds: list[int] | None = None,
    top_n: int = 15,
) -> dict:
    """
    TreeSHAP sobre uno o varios folds de un modelo de árboles (XGBoost o
    LightGBM) ya entrenado por entrenar_evaluar_xgboost/lightgbm. 'modelos'
    es la lista que devuelven esas funciones (un modelo por fold, mismo
    orden). Por defecto explica TODOS los folds por separado (ver nota de
    diseño del módulo sobre no promediar entre años).

    Retorna {anio_test: {"resumen_global": DataFrame, "shap_values": array,
    "X_test": DataFrame, "y_test": Series}}.
    """
    import shap

    _parchear_shap_xgboost_base_score()

    X, y, folds = _generar_folds(df, features_categoricas, features_numericas, min_anios_train)
    if indices_folds is None:
        indices_folds = range(len(folds))

    columnas_features = features_categoricas + features_numericas
    resultados = {}
    for i in indices_folds:
        fold = folds[i]
        modelo = modelos[i]
        X_test_cat, y_test = _preparar_fold_arbol(X, y, fold, features_categoricas, features_numericas, random_state)

        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_test_cat)
        # Compat entre versiones de shap: en clasificación binaria, algunas
        # devuelven una lista [contribución_clase_0, contribución_clase_1];
        # nos quedamos con la clase 1 ("Satisfecho"), que es la de interés.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        # Se neutraliza ANTES de guardar en 'resultados' (no solo dentro de
        # resumen_global_shap) para que la explicación LOCAL
        # (explicar_perfil_local) también quede protegida -- de lo
        # contrario, el resumen global saldría limpio pero un perfil
        # individual todavía podría mostrar el valor implausible crudo.
        shap_values = _neutralizar_columnas_implausibles(shap_values, columnas_features)

        anio_test = fold["anio_test"]
        resumen = resumen_global_shap(shap_values, columnas_features, top_n=top_n)
        resultados[anio_test] = {
            "resumen_global": resumen, "shap_values": shap_values, "X_test": X_test_cat, "y_test": y_test,
        }
        print(f"[{nombre_modelo}] SHAP calculado para fold de prueba {anio_test} ({len(X_test_cat)} encuestados reales, no sintéticos).")
    return resultados


# ======================================================================
# KernelSHAP -- TabNet
# ======================================================================

def _preparar_fold_tabnet(X, y, fold, features_categoricas, features_numericas, random_state):
    """
    Reconstruye X_train_final/X_test_final EXACTAMENTE como
    tabnet_model.py (misma imputación, SMOTENC, codificación categórica y
    escalado numérico) -- necesario para que el background y los códigos
    categóricos de KernelSHAP coincidan con lo que el modelo aprendió.
    """
    from sklearn.preprocessing import StandardScaler
    from tabnet_model import _construir_matriz_tabnet

    columnas_features = features_categoricas + features_numericas
    X_train = X.loc[fold["train_idx"], columnas_features]
    y_train = y.loc[fold["train_idx"]]
    X_test = X.loc[fold["test_idx"], columnas_features]
    y_test = y.loc[fold["test_idx"]]

    X_train, X_test = imputar_faltantes(X_train, X_test, features_numericas)
    X_train_bal, y_train_bal = aplicar_smote_train(X_train, y_train, random_state=random_state)

    cod_train, cod_test, cardinalidades = codificar_categoricas(
        X_train_bal[features_categoricas], X_test[features_categoricas], features_categoricas
    )
    scaler = StandardScaler().fit(X_train_bal[features_numericas])
    num_train_esc = scaler.transform(X_train_bal[features_numericas])
    num_test_esc = scaler.transform(X_test[features_numericas])

    X_train_final = _construir_matriz_tabnet(cod_train, num_train_esc)
    X_test_final = _construir_matriz_tabnet(cod_test, num_test_esc)
    return X_train_final, X_test_final, y_test


def explicar_tabnet(
    df: pd.DataFrame,
    modelos: list,
    features_categoricas: list[str] = FEATURES_CATEGORICAS,
    features_numericas: list[str] = FEATURES_NUMERICAS,
    min_anios_train: int = 5,
    random_state: int = 42,
    indices_folds: list[int] | None = None,
    tamano_background: int = 50,
    nsamples: int = 300,
    max_test_explicar: int | None = 300,
    top_n: int = 15,
) -> dict:
    """
    KernelSHAP sobre uno o varios folds de TabNet ya entrenado. El
    background (referencia de "valor normal" de cada variable) se resume
    con shap.kmeans sobre X_train_final del propio fold (ya balanceado por
    SMOTENC, en las mismas unidades que vio el modelo) -- 'tamano_background'
    controla cuántos centroides usar (no todo X_train, por costo
    computacional de KernelSHAP). 'nsamples' controla cuántas
    perturbaciones evalúa KernelSHAP por instancia explicada (más =
    más preciso y más lento).

    nsamples=300 (subido de 100): la matriz aplanada de TabNet tiene 48
    columnas (9 categóricas + 39 numéricas) -- la regresión ponderada
    interna de KernelSHAP necesita bastantes más muestras que columnas
    para no quedar subdeterminada (ver el mismo problema, mucho más
    grave, en explicar_cnn_lstm). No se detectó ningún valor implausible
    con nsamples=100 en las corridas reales de TabNet, pero se sube igual
    por margen de seguridad -- ver _neutralizar_columnas_implausibles.

    'max_test_explicar' (nuevo, optimización de costo): si el fold de
    prueba tiene más encuestados que este límite, se submuestrea de forma
    estratificada antes de correr KernelSHAP -- ver
    _submuestrear_test_para_kernelshap. Usa None para explicar el fold
    completo (comportamiento anterior).
    """
    import shap

    X, y, folds = _generar_folds(df, features_categoricas, features_numericas, min_anios_train)
    if indices_folds is None:
        indices_folds = range(len(folds))

    columnas_features = features_categoricas + features_numericas
    resultados = {}
    for i in indices_folds:
        fold = folds[i]
        modelo = modelos[i]
        X_train_final, X_test_final, y_test = _preparar_fold_tabnet(
            X, y, fold, features_categoricas, features_numericas, random_state
        )
        X_test_df = pd.DataFrame(X_test_final, columns=columnas_features)
        X_test_df, y_test = _submuestrear_test_para_kernelshap(
            X_test_df, y_test, max_test_explicar, random_state
        )
        X_test_final = X_test_df.to_numpy()

        background = shap.kmeans(X_train_final, min(tamano_background, len(X_train_final)))
        funcion_prediccion = lambda m: modelo.predict_proba(m.astype("float32"))[:, 1]  # noqa: E731
        explainer = shap.KernelExplainer(funcion_prediccion, background)
        shap_values = explainer.shap_values(X_test_final, nsamples=nsamples, silent=True, l1_reg="num_features(20)")
        shap_values = _neutralizar_columnas_implausibles(shap_values, columnas_features)  # ver explicar_arbol

        anio_test = fold["anio_test"]
        resumen = resumen_global_shap(shap_values, columnas_features, top_n=top_n)
        resultados[anio_test] = {
            "resumen_global": resumen, "shap_values": shap_values, "X_test": X_test_df, "y_test": y_test,
        }
        print(f"[tabnet] SHAP (KernelSHAP) calculado para fold de prueba {anio_test} ({len(X_test_final)} encuestados reales).")
    return resultados


# ======================================================================
# KernelSHAP -- CNN-LSTM (aplanado de la ventana temporal + estáticas)
# ======================================================================

def _preparar_base_cnn_lstm(df_personas, panel_macro, features_categoricas, features_numericas, ventana, min_anios_train):
    """Setup que en cnn_lstm.py se hace UNA vez fuera del loop de folds (ventanas temporales + folds)."""
    from ventanas_temporales import construir_ventanas_temporales

    X_static, y = preparar_features(df_personas, features_categoricas, features_numericas, COLUMNA_ANIO, COLUMNA_TARGET)
    tensor_seq, mask_seq, columnas_macro = construir_ventanas_temporales(
        X_static, panel_macro, ventana=ventana, columna_anio=COLUMNA_ANIO
    )
    folds = generar_folds_temporales(
        X_static.assign(**{COLUMNA_TARGET: y}),
        columna_anio=COLUMNA_ANIO, columna_target=COLUMNA_TARGET, min_anios_train=min_anios_train,
    )
    return X_static, y, tensor_seq, mask_seq, columnas_macro, folds


def _preparar_fold_cnn_lstm(
    X_static, y, tensor_seq, mask_seq, columnas_macro, fold,
    features_categoricas, features_numericas, ventana, random_state,
):
    """
    Reconstruye, para un fold de CNN-LSTM, las matrices PLANAS (background
    de entrenamiento y X_test a explicar) en las MISMAS unidades finales
    que vio el modelo (post-SMOTENC, escaladas) -- réplica de
    cnn_lstm.entrenar_evaluar_cnn_lstm líneas ~263-309, pero deteniéndose
    antes de construir el DataLoader/entrenar (el modelo ya está entrenado).

    Retorna: X_train_flat (DataFrame, background), X_test_flat (DataFrame,
    a explicar), y_test, columnas_flat (orden de columnas), cardinalidades_cat
    (lista, mismo orden que features_categoricas), n_features_macro.
    """
    from sklearn.preprocessing import StandardScaler
    from cnn_lstm import _imputar_ventanas

    pos_train = X_static.index.get_indexer(fold["train_idx"])
    pos_test = X_static.index.get_indexer(fold["test_idx"])

    X_train_static = X_static.loc[fold["train_idx"], features_categoricas + features_numericas]
    X_test_static = X_static.loc[fold["test_idx"], features_categoricas + features_numericas]
    y_train = y.loc[fold["train_idx"]]
    y_test = y.loc[fold["test_idx"]]
    seq_train, mask_train = tensor_seq[pos_train], mask_seq[pos_train]
    seq_test, mask_test = tensor_seq[pos_test], mask_seq[pos_test]

    X_train_static, X_test_static = imputar_faltantes(X_train_static, X_test_static, features_numericas)
    seq_train, seq_test = _imputar_ventanas(seq_train, mask_train, seq_test, mask_test)

    n_train, _, n_features_macro = seq_train.shape
    columnas_seq = [f"seq_t{t}_{feat}" for t in range(ventana) for feat in columnas_macro]
    columnas_mask = [f"mask_t{t}" for t in range(ventana)]
    X_flat_train = pd.DataFrame(
        seq_train.reshape(n_train, ventana * n_features_macro), columns=columnas_seq, index=X_train_static.index
    )
    X_flat_train[columnas_mask] = mask_train.astype(float)
    X_flat_train = pd.concat([X_flat_train, X_train_static], axis=1)

    X_flat_bal, y_bal = aplicar_smote_train(X_flat_train, y_train, random_state=random_state)

    seq_train_bal = X_flat_bal[columnas_seq].to_numpy(dtype=float).reshape(-1, ventana, n_features_macro)
    mask_train_bal = X_flat_bal[columnas_mask].to_numpy(dtype=float) >= 0.5
    X_static_bal = X_flat_bal[features_categoricas + features_numericas]

    scaler_estatica = StandardScaler().fit(X_static_bal[features_numericas])
    num_train_esc = scaler_estatica.transform(X_static_bal[features_numericas])
    num_test_esc = scaler_estatica.transform(X_test_static[features_numericas])

    scaler_seq = StandardScaler().fit(seq_train_bal.reshape(-1, n_features_macro))
    seq_train_esc = scaler_seq.transform(seq_train_bal.reshape(-1, n_features_macro)).reshape(seq_train_bal.shape)
    seq_test_esc = scaler_seq.transform(seq_test.reshape(-1, n_features_macro)).reshape(seq_test.shape)

    cod_train, cod_test, cardinalidades = codificar_categoricas(
        X_static_bal[features_categoricas], X_test_static[features_categoricas], features_categoricas
    )
    cardinalidades_cat = [cardinalidades[c] for c in features_categoricas]

    columnas_flat = columnas_seq + columnas_mask + features_categoricas + features_numericas

    def _aplanar(seq_esc, mask_bool, cod, num_esc, n):
        return np.hstack([
            seq_esc.reshape(n, ventana * n_features_macro),
            mask_bool.astype(float),
            cod.astype(float),
            num_esc,
        ])

    X_train_flat = pd.DataFrame(
        _aplanar(seq_train_esc, mask_train_bal, cod_train, num_train_esc, len(y_bal)), columns=columnas_flat
    )
    X_test_flat = pd.DataFrame(
        _aplanar(seq_test_esc, mask_test, cod_test, num_test_esc, len(y_test)), columns=columnas_flat
    )
    return X_train_flat, X_test_flat, y_test, columnas_flat, cardinalidades_cat, n_features_macro


def _funcion_prediccion_cnn_lstm(modelo, ventana, n_features_macro, n_cat, cardinalidades_cat, device=None):
    """
    Construye la función matriz_2d -> P(Satisfecho) que necesita
    KernelSHAP: recibe filas aplanadas (mismo orden de columnas de
    _preparar_fold_cnn_lstm), las "des-aplana" en los 4 tensores que
    espera CNNLSTMClasificador.forward(seq, mask, x_num, x_cat_codes), y
    devuelve la probabilidad de la clase "Satisfecho".

    Los códigos categóricos se redondean y se recortan al rango válido de
    cada embedding, porque KernelSHAP perturba las columnas con promedios
    continuos (no respeta que sean códigos enteros) -- sin este recorte,
    un código fuera de rango rompería nn.Embedding.
    """
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)
    modelo.eval()

    idx_fin_seq = ventana * n_features_macro
    idx_fin_mask = idx_fin_seq + ventana
    idx_fin_cat = idx_fin_mask + n_cat

    def f(matriz: np.ndarray) -> np.ndarray:
        matriz = np.asarray(matriz, dtype=np.float64)
        seq = matriz[:, :idx_fin_seq].reshape(-1, ventana, n_features_macro)
        mask = matriz[:, idx_fin_seq:idx_fin_mask] >= 0.5
        cat = matriz[:, idx_fin_mask:idx_fin_cat]
        num = matriz[:, idx_fin_cat:]

        cat_validado = np.zeros_like(cat, dtype=np.int64)
        for j, card in enumerate(cardinalidades_cat):
            cat_validado[:, j] = np.clip(np.round(cat[:, j]), 0, card - 1).astype(np.int64)

        with torch.no_grad():
            logits = modelo(
                torch.tensor(seq, dtype=torch.float32, device=device),
                torch.tensor(mask, dtype=torch.bool, device=device),
                torch.tensor(num, dtype=torch.float32, device=device),
                torch.tensor(cat_validado, dtype=torch.long, device=device),
            )
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba
    return f


def explicar_cnn_lstm(
    df_personas: pd.DataFrame,
    panel_macro: pd.DataFrame,
    modelos: list,
    features_categoricas: list[str] = FEATURES_CATEGORICAS,
    features_numericas: list[str] = FEATURES_NUMERICAS,
    ventana: int = VENTANA_TEMPORAL_ANIOS,
    min_anios_train: int = 5,
    random_state: int = 42,
    indices_folds: list[int] | None = None,
    tamano_background: int = 50,
    nsamples: int = 800,
    max_test_explicar: int | None = 300,
    top_n: int = 15,
) -> dict:
    """
    KernelSHAP sobre uno o varios folds del CNN-LSTM ya entrenado. Ver
    nota de diseño del módulo: la ventana temporal se aplana en columnas
    'seq_t{t}_{variable}' / 'mask_t{t}', igual que en el aplanado de SMOTENC
    de cnn_lstm.py -- así el resumen global de SHAP puede mostrar, por
    ejemplo, si 'tasa_desempleo' pesa más en t0 (hace 2 años) o en t2
    (el propio año de la encuesta).

    BUG REAL, dos intentos de corrección (fold 2017, el más chico de los 5):
    1er intento (insuficiente): al correr esto contra el CNN-LSTM real, el
    fold de 2017 devolvió un valor de SHAP de ~3.46e11 para
    'econ_situation_cat' y 'seq_t0_pobreza_ingresos' -- imposible, dado que
    la predicción explicada es una probabilidad en [0,1]. Se subió
    'nsamples' de 100 a 800 (la matriz aplanada tiene ~132 columnas:
    ventana=3 x ~27 variables macro + 3 de máscara + 9 categóricas + 39
    numéricas -- con 100 muestras la regresión ponderada interna de
    KernelSHAP queda muy subdeterminada). Al re-ejecutar el fold 2017 con
    nsamples=800, el problema REAPARECIÓ, esta vez en DOS columnas
    distintas ('seq_t2_gini_ingpc' y 'seq_t1_v2xel_frefair', ~3.57e10) --
    es decir, subir nsamples por sí solo NO alcanza; el fold
    2017 parece tener alguna colinealidad/degeneración estructural en su
    background (es el fold con menos años reales de entrenamiento, así que
    la ventana temporal del CNN-LSTM tiene poca variedad real entre
    encuestados del mismo año) que sigue mal condicionando la regresión
    interna de KernelSHAP sin importar cuántas muestras se usen.

    2do intento (defensivo, sin diagnóstico 100% cerrado): se intentó
    reproducir el blow-up de forma sintética (columnas colineales/
    constantes/con poca variedad, a la escala de p=132) para entender la
    causa exacta y no se logró replicar -- así que el mecanismo numérico
    preciso queda sin confirmar. En su lugar se blindó el resultado desde
    dos frentes: (a) 'l1_reg=\"num_features(20)\"' en la llamada a
    'explainer.shap_values', que regulariza la regresión interna de
    KernelSHAP (puede ayudar si hay colinealidad, aunque no se verificó de
    forma aislada); (b) '_neutralizar_columnas_implausibles', que
    NEUTRALIZA (pone en 0) cualquier columna cuyo |SHAP| promedio supere
    UMBRAL_SHAP_IMPLAUSIBLE=10.0 e imprime un AVISO explícito -- se aplica
    tanto al resumen global como al array 'shap_values' guardado en
    'resultados' (para que explicar_perfil_local tampoco herede el valor
    contaminado). Esto GARANTIZA que el CSV/gráfico/explicación local nunca
    queden contaminados, pero si el AVISO vuelve a aparecer para el fold
    2017, la lectura honesta es que ese fold específico no tiene SHAP
    global confiable para esas 1-2 columnas puntuales -- considerar excluir
    o poner una nota al pie sobre el fold 2017 en la narrativa de SHAP del
    CNN-LSTM en la tesis, en vez de insistir en "arreglarlo" indefinidamente.

    OPTIMIZACIÓN DE COSTO -- 'max_test_explicar' (a pedido del tutor):
    este es, con diferencia, el paso más caro de toda la fase de
    Evaluación/XAI -- KernelSHAP llama al modelo 'nsamples' (800) veces
    POR CADA encuestado explicado, y cada fold de prueba tiene ~1200
    encuestados/año, es decir, ~960,000 pasadas del modelo por fold si se
    explica el fold COMPLETO. Por defecto se submuestrea a
    'max_test_explicar' encuestados (estratificado por la clase real,
    ver _submuestrear_test_para_kernelshap) antes de llamar a
    'explainer.shap_values' -- reduce el costo ~4x sin tocar 'nsamples'
    (que se mantiene en 800, el valor que evita la inestabilidad numérica
    documentada arriba) ni el entrenamiento del modelo. El resumen GLOBAL
    (promedio de |SHAP|) es un estadístico agregado que no necesita cada
    encuestado real del fold para estabilizarse. Usa 'max_test_explicar=None'
    para volver al comportamiento anterior (fold completo).
    """
    import shap

    X_static, y, tensor_seq, mask_seq, columnas_macro, folds = _preparar_base_cnn_lstm(
        df_personas, panel_macro, features_categoricas, features_numericas, ventana, min_anios_train
    )
    if indices_folds is None:
        indices_folds = range(len(folds))

    resultados = {}
    for i in indices_folds:
        fold = folds[i]
        modelo = modelos[i]
        X_train_flat, X_test_flat, y_test, columnas_flat, cardinalidades_cat, n_features_macro = _preparar_fold_cnn_lstm(
            X_static, y, tensor_seq, mask_seq, columnas_macro, fold,
            features_categoricas, features_numericas, ventana, random_state,
        )
        X_test_flat, y_test = _submuestrear_test_para_kernelshap(
            X_test_flat, y_test, max_test_explicar, random_state
        )

        funcion_prediccion = _funcion_prediccion_cnn_lstm(
            modelo, ventana, n_features_macro, len(features_categoricas), cardinalidades_cat
        )
        background = shap.kmeans(X_train_flat, min(tamano_background, len(X_train_flat)))
        explainer = shap.KernelExplainer(funcion_prediccion, background)
        # l1_reg: regularización adicional de la regresión interna de
        # KernelSHAP -- ayuda a que el problema quede mejor condicionado
        # cuando hay columnas casi colineales (ver _neutralizar_columnas_implausibles).
        shap_values = explainer.shap_values(X_test_flat, nsamples=nsamples, silent=True, l1_reg="num_features(20)")
        shap_values = _neutralizar_columnas_implausibles(shap_values, columnas_flat)  # ver explicar_arbol

        anio_test = fold["anio_test"]
        resumen = resumen_global_shap(shap_values, columnas_flat, top_n=top_n)
        resultados[anio_test] = {
            "resumen_global": resumen, "shap_values": shap_values, "X_test": X_test_flat, "y_test": y_test,
        }
        print(f"[cnn_lstm] SHAP (KernelSHAP) calculado para fold de prueba {anio_test} ({len(X_test_flat)} encuestados reales).")
    return resultados


# ======================================================================
# Demostración rápida (solo la ruta de árboles, ejecutable sin torch)
# ======================================================================

if __name__ == "__main__":
    import baseline_xgboost

    df = cargar_dataset()
    resultados_xgb, modelos_xgb = baseline_xgboost.entrenar_evaluar_xgboost(df, min_anios_train=5)

    explicaciones = explicar_arbol(df, modelos_xgb, "xgboost", indices_folds=[len(modelos_xgb) - 1])
    ultimo_anio = max(explicaciones.keys())
    resumen = explicaciones[ultimo_anio]["resumen_global"]
    print(f"\n[shap_explicabilidad] Top variables (importancia global) -- fold {ultimo_anio}:")
    print(resumen.to_string(index=False))

    guardar_resumen_shap(resumen, RUTA_TABLAS / f"shap_global_xgboost_{ultimo_anio}.csv")
    graficar_resumen_shap(resumen, f"XGBoost -- importancia SHAP global ({ultimo_anio})", RUTA_FIGURES / f"shap_global_xgboost_{ultimo_anio}.png")

    ejemplo_local = explicar_perfil_local(
        explicaciones[ultimo_anio]["shap_values"], explicaciones[ultimo_anio]["X_test"],
        FEATURES_CATEGORICAS + FEATURES_NUMERICAS, idx=0,
    )
    print(f"\n[shap_explicabilidad] Explicación LOCAL del primer encuestado del fold {ultimo_anio}:")
    print(ejemplo_local.to_string(index=False))
