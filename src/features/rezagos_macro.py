"""
rezagos_macro.py
---------------------------------------------------------------------
Fase de Modelado (CRISP-DM) - preparación de features: variables macro REZAGADAS (lags) para
los modelos NO secuenciales (XGBoost, LightGBM, TabNet), a pedido del
tutor de tesis: sustentar el tema de "temporalidad" también en los
modelos tabulares, no solo en el CNN-LSTM.

Para cada persona de dataset_modelado_personas.csv, agrega columnas
adicionales con el valor de un subconjunto de indicadores macro
(ENEMDU/V-Dem) en años ANTERIORES a su año de encuesta (ej.
'tasa_desempleo_lag1' = tasa_desempleo del año t-1, 'tasa_desempleo_lag3'
= la de t-3), usando panel_macro_anual.csv como fuente -- la MISMA
fuente que ventanas_temporales.py usa para la ventana del CNN-LSTM,
solo que aquí se "desenrolla" en columnas planas en vez de un tensor
secuencial. Esto permite comparar de forma justa arquitectura
secuencial explícita (CNN-LSTM) vs. variables de rezago diseñadas a
mano (XGBoost/LightGBM/TabNet) -- ambas parten de la misma información
histórica, solo que empaquetada distinto.

DECISIÓN DE DISEÑO (evitar fuga de información y explosión de
features): por defecto solo se rezagan un subconjunto reducido de
variables clave (VARIABLES_A_REZAGAR), NO las 39 variables numéricas
completas de config_features.py -- rezagar todo multiplicaría la
dimensionalidad y crearía alta multicolinealidad entre rezagos
consecutivos de la misma variable (el desempleo de un año está muy
correlacionado con el del año anterior). Se excluyen SIEMPRE las
columnas 'latino_*' (fuga de información, mismo motivo documentado en
ventanas_temporales.py) y las de identificación/metadata (país, tamaño
de muestra).

HUECOS TEMPORALES DE LATINOBARÓMETRO: NO afectan a estos rezagos,
porque las variables rezagadas vienen de ENEMDU/V-Dem, fuentes anuales
SIN huecos (verificado contra panel_macro_anual.csv real: tasa_desempleo
y v2x_polyarchy tienen valor en los 18 años 2007-2024 sin excepción,
incluyendo 2011-2015/2019/2021-2022, que sí son huecos de la serie
agregada de Latinobarómetro -- por eso esa serie no se usa aquí). El
único faltante real posible es de BORDE del panel (ej. una persona
encuestada en 2007 pidiendo lag3 -> año 2004, anterior al inicio del
panel) -- se deja como NaN y se imputa después con la mediana de
X_train de cada fold, igual que el resto de columnas numéricas (ver
preparacion_modelado.imputar_faltantes).
"""
import pandas as pd

# Subconjunto de indicadores macro a rezagar -- elegidos por su
# relevancia directa para el tema de tesis (mercado laboral, ingreso,
# desigualdad, calidad democrática), no las 39 variables completas de
# config_features.py.
VARIABLES_A_REZAGAR = [
    "tasa_desempleo", "gini_ingpc", "ingreso_promedio_pc", "pobreza_ingresos",
    "v2x_polyarchy", "v2x_libdem",
]

# Rezagos a construir: t-1 (cambio inmediato) y t-3 (cambio de mediano
# plazo) -- no todos los años intermedios, para no inflar la
# dimensionalidad ni la multicolinealidad entre rezagos consecutivos.
LAGS = [1, 3]


def construir_variables_rezagadas(
    df_personas: pd.DataFrame,
    panel_macro: pd.DataFrame,
    variables: list[str] = VARIABLES_A_REZAGAR,
    lags: list[int] = LAGS,
    columna_anio: str = "anio",
) -> pd.DataFrame:
    """
    Retorna un DataFrame con el MISMO índice que df_personas y una
    columna nueva '{variable}_lag{n}' por cada combinación de variable
    y rezago pedido -- listo para pd.concat([df_personas, rezagos], axis=1).

    NaN cuando el año rezagado (anio_persona - n) no existe en el panel
    (borde inicial de la serie, ver docstring del módulo).
    """
    panel = panel_macro.set_index(columna_anio).sort_index()

    faltantes = [v for v in variables if v not in panel.columns]
    if faltantes:
        raise KeyError(
            f"Variables a rezagar no encontradas en panel_macro_anual.csv: {faltantes}. "
            "Revisar VARIABLES_A_REZAGAR o merge_final.py."
        )

    anios_persona = df_personas[columna_anio].astype(int)

    columnas_nuevas = {}
    for variable in variables:
        serie_variable = panel[variable]
        for lag in lags:
            anios_objetivo = anios_persona - lag
            columnas_nuevas[f"{variable}_lag{lag}"] = anios_objetivo.map(serie_variable)

    return pd.DataFrame(columnas_nuevas, index=df_personas.index)


def nombres_columnas_rezagadas(variables: list[str] = VARIABLES_A_REZAGAR, lags: list[int] = LAGS) -> list[str]:
    """Lista de nombres '{variable}_lag{n}' en el mismo orden que genera construir_variables_rezagadas -- útil para extender FEATURES_NUMERICAS."""
    return [f"{variable}_lag{lag}" for variable in variables for lag in lags]


def agregar_rezagos_a_dataset(
    df_personas: pd.DataFrame,
    panel_macro: pd.DataFrame,
    variables: list[str] = VARIABLES_A_REZAGAR,
    lags: list[int] = LAGS,
    columna_anio: str = "anio",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Atajo de conveniencia: retorna (dataset_extendido, columnas_nuevas).
    dataset_extendido = df_personas + las columnas de rezago pegadas al
    lado (mismas filas, mismo orden). Se usa así en los scripts de
    entrenamiento, sin tocar preparar_features ni los baseline_*.py:

        df_ext, columnas_lag = agregar_rezagos_a_dataset(df, panel_macro)
        entrenar_evaluar_xgboost(df_ext, features_numericas=FEATURES_NUMERICAS + columnas_lag)
    """
    rezagos = construir_variables_rezagadas(df_personas, panel_macro, variables, lags, columna_anio)
    columnas_nuevas = list(rezagos.columns)
    dataset_extendido = pd.concat([df_personas, rezagos], axis=1)
    return dataset_extendido, columnas_nuevas


def resumen_cobertura_rezagos(df_rezagos: pd.DataFrame) -> pd.DataFrame:
    """% de valores faltantes por columna de rezago -- para reportar en la tesis cuántas filas quedan sin historia completa (esperable solo en los primeros años de la serie, por el borde del panel)."""
    faltantes_pct = (df_rezagos.isna().mean() * 100).round(2)
    faltantes_pct.name = "pct_faltante"
    return faltantes_pct.to_frame()


if __name__ == "__main__":
    personas = pd.read_csv("data/processed/dataset_modelado_personas.csv")
    panel = pd.read_csv("data/processed/panel_macro_anual.csv")

    dataset_extendido, columnas_nuevas = agregar_rezagos_a_dataset(personas, panel)
    print(f"Columnas de rezago creadas: {columnas_nuevas}")
    print(f"\nCobertura (% faltante por columna):\n{resumen_cobertura_rezagos(dataset_extendido[columnas_nuevas])}")
    print(f"\nShape original: {personas.shape}  ->  con rezagos: {dataset_extendido.shape}")
