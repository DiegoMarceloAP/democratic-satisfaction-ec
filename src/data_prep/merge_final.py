"""
merge_final.py
---------------------------------------------------------------------
Fase 2 (Preparación de datos) - CRISP-DM, paso final de integración.
Junta las tres fuentes usando 'anio' como llave, respetando la decisión
de merge nacional-año (ver sql/README_extraccion.md) y evitando fuga de
información (data leakage) del target.

Produce DOS artefactos con propósitos distintos:

1) dataset_modelado_personas.csv
   Nivel: persona (una fila por encuestado de Latinobarómetro Ecuador).
   Contiene el target 'satisfecho_democracia', las variables individuales
   ya limpias, y los indicadores CONTEMPORÁNEOS (mismo año) de ENEMDU y
   V-Dem. Listo para los modelos NO secuenciales (XGBoost, LightGBM,
   TabNet/MLP).

   Decisión de diseño para evitar data leakage: NO se incluye aquí la
   serie agregada nacional-anual de Latinobarómetro
   (latinobarometro_ecuador_serie_anual.csv). Para un año dado,
   'tasa_satisfaccion_democracia' de esa serie es literalmente el
   promedio del target de TODOS los encuestados de ese año, incluyendo
   la propia fila -- usarla como feature contemporánea sería fuga de
   información. V-Dem y ENEMDU no tienen ese problema: son fuentes
   independientes de la percepción ciudadana individual.

2) panel_macro_anual.csv
   Nivel: año (grilla completa 2007-2024). Junta V-Dem + ENEMDU + la
   serie agregada de Latinobarómetro (con su columna 'estado_anio').
   Este panel se deja SEPARADO (no se aplana dentro del dataset de
   personas): sirve para construir, en la fase de Modelado, las
   ventanas móviles [batch, timesteps, features] del CNN-LSTM, donde sí
   es válido usar la serie de Latinobarómetro pero SOLO como rezago
   (años anteriores al año de la encuesta de cada respondiente), nunca
   el valor del propio año objetivo.
"""
from pathlib import Path

import pandas as pd

PERSONAS_PATH = Path("data/processed/latinobarometro_ecuador_personas.csv")
SERIE_LATINO_PATH = Path("data/processed/latinobarometro_ecuador_serie_anual.csv")
ENEMDU_PATH = Path("data/processed/enemdu_indicadores_anuales.csv")
VDEM_PATH = Path("data/processed/vdem_ecuador_anual.csv")

OUT_DATASET_PERSONAS = Path("data/processed/dataset_modelado_personas.csv")
OUT_PANEL_MACRO = Path("data/processed/panel_macro_anual.csv")


def _cargar(path: Path, nombre: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path}' ({nombre}). Corre primero el módulo de "
            "procesamiento correspondiente (ver Fase 2)."
        )
    return pd.read_csv(path)


def construir_panel_macro_anual(
    vdem_path: Path = VDEM_PATH,
    enemdu_path: Path = ENEMDU_PATH,
    serie_latino_path: Path = SERIE_LATINO_PATH,
) -> pd.DataFrame:
    """Panel año x indicadores macro (V-Dem + ENEMDU + Latinobarómetro agregado)."""
    vdem = _cargar(vdem_path, "V-Dem")
    enemdu = _cargar(enemdu_path, "ENEMDU")
    serie_latino = _cargar(serie_latino_path, "serie anual Latinobarómetro")

    panel = vdem.merge(enemdu, on="anio", how="outer", suffixes=("_vdem", "_enemdu"))
    serie_latino_renombrada = serie_latino.rename(
        columns={c: f"latino_{c}" for c in serie_latino.columns if c != "anio"}
    )
    panel = panel.merge(serie_latino_renombrada, on="anio", how="outer")
    panel = panel.sort_values("anio").reset_index(drop=True)

    faltan = panel[panel[["v2x_polyarchy", "tasa_desempleo"]].isna().any(axis=1)]
    if len(faltan):
        print(
            f"[merge_final] AVISO: {len(faltan)} años sin dato de V-Dem o ENEMDU en el panel "
            "macro -- ambas fuentes deberían cubrir 2007-2024 sin huecos; revisar."
        )
    return panel


def construir_dataset_modelado_personas(
    personas_path: Path = PERSONAS_PATH,
    enemdu_path: Path = ENEMDU_PATH,
    vdem_path: Path = VDEM_PATH,
) -> pd.DataFrame:
    """
    Dataset a nivel persona con indicadores contemporáneos de ENEMDU y
    V-Dem. NO incluye la serie agregada de Latinobarómetro (ver
    docstring del módulo: riesgo de fuga de información del target).
    """
    personas = _cargar(personas_path, "Latinobarómetro personas")
    enemdu = _cargar(enemdu_path, "ENEMDU")
    vdem = _cargar(vdem_path, "V-Dem")

    n_antes = len(personas)
    dataset = personas.merge(enemdu, on="anio", how="left", suffixes=("", "_enemdu"))
    dataset = dataset.merge(vdem, on="anio", how="left", suffixes=("", "_vdem"))
    assert len(dataset) == n_antes, "El merge cambió el número de filas de personas -- revisar llaves duplicadas."

    sin_macro = dataset[dataset[["tasa_desempleo", "v2x_polyarchy"]].isna().any(axis=1)]
    if len(sin_macro):
        anios_afectados = sorted(sin_macro["anio"].unique().tolist())
        print(
            f"[merge_final] AVISO: {len(sin_macro)} encuestados sin indicadores macro "
            f"(años afectados: {anios_afectados}). Revisar cobertura de ENEMDU/V-Dem para esos años."
        )
    return dataset


def ejecutar_pipeline(
    personas_path: Path = PERSONAS_PATH,
    serie_latino_path: Path = SERIE_LATINO_PATH,
    enemdu_path: Path = ENEMDU_PATH,
    vdem_path: Path = VDEM_PATH,
    out_dataset_personas: Path = OUT_DATASET_PERSONAS,
    out_panel_macro: Path = OUT_PANEL_MACRO,
):
    panel_macro = construir_panel_macro_anual(vdem_path, enemdu_path, serie_latino_path)
    out_panel_macro.parent.mkdir(parents=True, exist_ok=True)
    panel_macro.to_csv(out_panel_macro, index=False)

    dataset_personas = construir_dataset_modelado_personas(personas_path, enemdu_path, vdem_path)
    out_dataset_personas.parent.mkdir(parents=True, exist_ok=True)
    dataset_personas.to_csv(out_dataset_personas, index=False)

    print(f"[merge_final] panel_macro_anual: {panel_macro.shape}")
    print(f"[merge_final] dataset_modelado_personas: {dataset_personas.shape}")
    return dataset_personas, panel_macro


if __name__ == "__main__":
    ejecutar_pipeline()
