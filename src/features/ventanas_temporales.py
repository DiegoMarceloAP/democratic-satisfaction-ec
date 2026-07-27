"""
ventanas_temporales.py
---------------------------------------------------------------------
Fase de Preparación de Features - construcción del input temporal del
CNN-LSTM: por cada persona de dataset_modelado_personas.csv, arma una
ventana de VENTANA_TEMPORAL_ANIOS (3, según el plan de modelado) con
los indicadores macro de panel_macro_anual.csv correspondientes a los
años [T-2, T-1, T], donde T es el año de encuesta de esa persona.

Devuelve, además del tensor, una MÁSCARA booleana por timestep: para
personas encuestadas en los primeros años de la serie (ej. 2007 con
ventana=3 -> necesitaría 2005 y 2006, que no existen), esos timesteps
quedan marcados como no disponibles (mask=False) en vez de rellenarse
con un valor inventado. Esa máscara se pasa a una capa Masking de
Keras/PyTorch para que el LSTM ignore esos timesteps -- así se
implementa la "ventana dinámica según disponibilidad real de años" que
pedía la propuesta de tesis.

DECISIÓN IMPORTANTE (evitar fuga de información): por defecto se
EXCLUYEN las columnas 'latino_*' de panel_macro_anual.csv (la serie
agregada de Latinobarómetro). Usarlas del mismo año T de la persona
sería fuga directa (esa serie se calcula con las respuestas de los
mismos encuestados que queremos predecir). Usarlas rezagadas (años
anteriores a T) sí sería válido, pero requeriría una ventana
DESALINEADA solo para esas columnas (distinta a la de V-Dem/ENEMDU), lo
cual complica la semántica de "mismo timestep = mismo año" para todas
las features del tensor. Se deja como extensión futura documentada, no
implementada aquí -- no se resuelve a medias ni se arriesga una fuga
por simplicidad.

BUG REAL ENCONTRADO Y CORREGIDO (corriendo contra panel_macro_anual.csv
real, no aparecía con datos sintéticos): V-Dem trae columnas
identificadoras de país ('country_name'='Ecuador', 'country_text_id'=
'ECU', 'country_id'=75) que sobreviven al merge de merge_final.py. No
empiezan con 'latino_', así que antes se colaban en columnas_macro por
defecto -- 'country_name'/'country_text_id' no son numéricas y rompían
la conversión a float del tensor; 'country_id' sí es numérica pero es
un identificador constante (Ecuador solamente), no un indicador
predictivo real. Ahora se excluyen explícitamente, además de filtrar
por dtype numérico como salvaguarda general ante cualquier otra columna
no numérica que pudiera aparecer en el panel en el futuro.
"""
import numpy as np
import pandas as pd

# Columnas de identificación/metadata de panel_macro_anual.csv que NO son
# indicadores macro y deben excluirse siempre del tensor, aunque no
# empiecen con 'latino_' (ver nota de bug real arriba).
COLUMNAS_METADATA_EXCLUIR = {"country_name", "country_text_id", "country_id", "n_personas_muestra"}


def construir_ventanas_temporales(
    df_personas: pd.DataFrame,
    panel_macro: pd.DataFrame,
    ventana: int = 3,
    columna_anio: str = "anio",
    columnas_macro: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Retorna:
        tensor: np.ndarray de forma [n_personas, ventana, n_features],
                con NaN en los timesteps sin dato disponible.
        mask:   np.ndarray booleano [n_personas, ventana], True si ese
                timestep tiene datos reales (no es padding).
        columnas_usadas: nombres de las columnas macro incluidas, en el
                mismo orden que la última dimensión del tensor.

    La ventana para una persona encuestada en el año T cubre
    [T-ventana+1, ..., T] (incluye el propio año T para V-Dem/ENEMDU,
    que no tienen riesgo de fuga al ser fuentes independientes de la
    percepción ciudadana).
    """
    panel = panel_macro.set_index(columna_anio).sort_index()

    if columnas_macro is None:
        columnas_macro = [
            c for c in panel.columns
            if not c.startswith("latino_")
            and c not in COLUMNAS_METADATA_EXCLUIR
            and pd.api.types.is_numeric_dtype(panel[c])
        ]

    n = len(df_personas)
    n_features = len(columnas_macro)
    tensor = np.full((n, ventana, n_features), np.nan, dtype=float)
    mask = np.zeros((n, ventana), dtype=bool)

    anios_persona = df_personas[columna_anio].astype(int).to_numpy()

    for fila_i in range(n):
        anio_t = anios_persona[fila_i]
        for t in range(ventana):
            anio_objetivo = anio_t - (ventana - 1 - t)
            if anio_objetivo in panel.index:
                tensor[fila_i, t, :] = panel.loc[anio_objetivo, columnas_macro].to_numpy(dtype=float)
                mask[fila_i, t] = True

    return tensor, mask, columnas_macro


def resumen_cobertura_ventanas(mask: np.ndarray) -> pd.DataFrame:
    """
    Resumen rápido de qué tan completas quedaron las ventanas: cuántas
    personas tienen 0, 1, 2, ... timesteps reales disponibles. Útil para
    reportar en la tesis cuántos encuestados de los primeros años de la
    serie tienen historia incompleta.
    """
    n_timesteps_disponibles = mask.sum(axis=1)
    conteo = pd.Series(n_timesteps_disponibles).value_counts().sort_index()
    conteo.index.name = "timesteps_disponibles"
    conteo.name = "n_personas"
    return conteo.to_frame()


if __name__ == "__main__":
    personas = pd.read_csv("data/processed/dataset_modelado_personas.csv")
    panel = pd.read_csv("data/processed/panel_macro_anual.csv")

    tensor, mask, columnas = construir_ventanas_temporales(personas, panel, ventana=3)
    print(f"Tensor: {tensor.shape}  (n_personas, ventana, n_features)")
    print(f"Columnas macro usadas: {columnas}")
    print(f"\nMáscara (True = timestep real disponible):\n{mask}")
    print(f"\nResumen de cobertura:\n{resumen_cobertura_ventanas(mask)}")
