"""
latinobarometro_loader_crudo.py
---------------------------------------------------------------------
Fase 2 (Preparación de datos) - CRISP-DM
Tesis: Deep Learning for Modeling Democratic Satisfaction and
Socioeconomic Inequality in Ecuador.

FUENTE (revisión con tutor, reemplaza a latinobarometro_loader_oficial.py):
descarga directa desde el sitio de Latinobarómetro de los microdatos de
Ecuador de cada ola (2007-2024), NO la extracción vía Superset (que tenía
inconsistencias y no pudo utilizarse -- ver limitaciones de la tesis) ni
el archivo pre-armonizado de la sesión anterior. Cada ola trae su propio
"Libro de Códigos" (PDF) con los nombres de columna que usó Latinobarómetro
ESE año en particular (no son constantes entre olas).

Este módulo combina DOS formatos de origen distintos, según lo que cada
ola tiene disponible y lo que resultó confiable tras la verificación:

  - 2011, 2013, 2015, 2016, 2017, 2018, 2020, 2023, 2024 (9 olas):
    archivo SPSS (.sav). Se lee con pyreadstat, que amarra cada valor a
    su variable mediante la metadata interna del archivo (no mediante
    texto de encabezado), y además trae las etiquetas de variable/valor
    de SPSS -- se usaron para verificar, ola por ola, que cada columna
    del crosswalk corresponde realmente a lo que dice medir (0 columnas
    encontradas con contenido distinto al esperado).

  - 2007, 2008, 2009, 2010 (4 olas): archivo CSV, porque el .sav de
    estos 4 años específicos no se puede abrir con la librería
    disponible (bug documentado de ReadStat/pyreadstat con archivos SPSS
    antiguos de ciertas versiones -- ver issue #287 en
    github.com/WizardMac/ReadStat; se resuelve abriendo y re-guardando
    el archivo en SPSS real, pero no se dispone de esa herramienta aquí).

    HALLAZGO IMPORTANTE (verificado comparando, columna por columna,
    los porcentajes que publica el Libro de Códigos oficial contra los
    datos reales del CSV): en estos 4 archivos, el CONTENIDO de cada
    columna está corrido UNA POSICIÓN antes de su propio nombre de
    encabezado. Es decir, el dato de la variable que el codebook llama
    "X" no está en la columna llamada "X" sino en la columna
    INMEDIATAMENTE ANTERIOR en el archivo. Esto se confirmó con
    coincidencias exactas de porcentaje en 3 secciones distintas del
    cuestionario (satisfacción con la democracia, sociodemográficas,
    bienes del hogar) en los 4 años, así que se aplica como una
    corrección uniforme (columna_real = columna_citada_por_codebook
    desplazada -1 posición), no una lista de excepciones caso por caso.
    Este corrimiento es la explicación técnica de la duda original sobre
    'democ_satis': para 2007-2010, la columna con ese nombre literal en
    el CSV NUNCA fue la variable correcta.

VARIABLES QUE QUEDAN FUERA (decisión explícita del tutor, no un
descuido):
  - elections_vote: solo tiene dato real en 1 de las 13 olas (2009);
    inutilizable como serie temporal, se descarta permanentemente.
  - resp_economic_perception: solo existe en 6 de las 13 olas
    (2008-2015); se conserva con NA explícito en el resto -- es un
    hueco estructural del cuestionario, no un error de esta extracción.

VARIABLES RECUPERADAS respecto a la sesión anterior (existen en las
descargas directas, aunque no estaban en el archivo pre-armonizado que
se usó antes): goods_wash_mach, goods_car, goods_sewage, goods_hot_water,
resp_chief -- cobertura completa en las 13 olas.
"""
import glob
from pathlib import Path

import pandas as pd
import pyreadstat

RAW_DIR = Path("data/raw/latinobarometro")
OUT_PATH = Path("data/raw/latinobarometro_ecuador_corregido.csv")

ANIOS_CSV = {2007, 2008, 2009, 2010}
ANIOS_SAV = {2011, 2013, 2015, 2016, 2017, 2018, 2020, 2023, 2024}

ARCHIVOS_CSV = {
    2007: "Latinobarometro_2007_Ecuador_Csv_esp_v1.csv",
    2008: "Latinobarometro_2008_Ecuador_Csv_esp_v1.csv",
    2009: "Latinobarometro_2009_Ecuador_Csv_esp_v1.csv",
    2010: "Latinobarometro_2010_Ecuador_Csv_esp_v12025.csv",
}
PATRON_SAV = "Latinobarometro_{anio}_Ecuador_Spss_esp_v1*.sav"

# ---------------------------------------------------------------------
# Crosswalk para los años CSV (2007-2010): columna tal como la CITA el
# Libro de Códigos oficial de cada ola. El loader le resta 1 posición
# (ver docstring) para llegar a la columna donde el dato realmente vive.
# ---------------------------------------------------------------------
CROSSWALK_CSV = {
    "democ_satis": {2007: "p12st", 2008: "p22st.a", 2009: "p12st.a", 2010: "P11ST.A"},
    "democ_supp": {2007: "p9st", 2008: "p13st", 2009: "p10st", 2010: "P10ST"},
    "left_right_scale": {2007: "p67st", 2008: "p56st", 2009: "p69st", 2010: "P60ST"},
    "job_concern": {2007: "s1", 2008: "s1", 2009: "s1", 2010: "S3"},
    "econ_situation": {2007: "p100st", 2008: "p4st", 2009: "p3st.a", 2010: "P3ST.A"},
    "resp_economic_perception": {2008: "p7st", 2009: "p6st", 2010: "P6ST"},
    "confidence_congress": {2007: "p24st.f", 2008: "p28st.a", 2009: "p26st.a", 2010: "P20ST.A"},
    "confidence_judiciary": {2007: "p24st.d", 2008: "p28st.b", 2009: "p26st.b", 2010: "P20ST.B"},
    "confidence_church": {2007: "p27st.c", 2008: "p28st.g", 2009: "p26st.g", 2010: "P20ST.G"},
    "confidence_police": {2007: "p27st.f", 2008: "p31st.c", 2009: "p24st.c", 2010: "P18ST.C"},
    "confidence_army": {2007: "p27st.d", 2008: "p28st.d", 2009: "p26st.d", 2010: "P20ST.D"},
    "confidence_political_parties": {2007: "p27st.e", 2008: "p28st.c", 2009: "p26st.c", 2010: "P20ST.C"},
    "resp_sex": {2007: "s10", 2008: "s8", 2009: "s5", 2010: "S7"},
    "resp_age": {2007: "s11", 2008: "s9", 2009: "s6", 2010: "S8"},
    "resp_education": {2007: "reeduc1", 2008: "reeduc1", 2009: "reeduc1", 2010: "REEDUC1"},
    "resp_employment": {2007: "s17a", 2008: "s17a", 2009: "s14a", 2010: "S16A"},
    "resp_religion": {2007: "s4", 2008: "s5", 2009: "s7", 2010: "S9"},
    "goods_wash_mach": {2007: "s19e", 2008: "s19e", 2009: "s19g", 2010: "S21G"},
    "goods_car": {2007: "s19h", 2008: "s19h", 2009: "s19j", 2010: "S21J"},
    "goods_sewage": {2007: "s19l", 2008: "s19l", 2009: "s19n", 2010: "S21N"},
    "goods_hot_water": {2007: "s19k", 2008: "s19k", 2009: "s19m", 2010: "S21M"},
    "resp_chief": {2007: "s12", 2008: "s12", 2009: "s9", 2010: "S11"},
}

# ---------------------------------------------------------------------
# Crosswalk para los años SPSS (2011-2024): columna real, verificada
# directamente contra las etiquetas de variable/valor embebidas en cada
# archivo .sav (sin corrimiento -- el formato SPSS amarra dato y
# metadata, no depende de una fila de texto de encabezado).
# ---------------------------------------------------------------------
CROSSWALK_SAV = {
    "democ_satis": {2011: "P14ST.A", 2013: "P13TGB.A", 2015: "P12TG.A", 2016: "P9STGBSA", 2017: "P9STGBSC.A", 2018: "P13STGBS.A", 2020: "P11STGBS.A", 2023: "P11STGBS.A", 2024: "P12STGBS.A"},
    "democ_supp": {2011: "P13ST", 2013: "P12STGBS", 2015: "P11STGBS", 2016: "P8STGBS", 2017: "P8STGBS", 2018: "P12STGBS", 2020: "P10STGBS", 2023: "P10STGBS", 2024: "P11STGBS"},
    "left_right_scale": {2011: "P76ST", 2013: "P41ST", 2015: "P27ST", 2016: "A_008_001", 2017: "P19STC", 2018: "P22ST", 2020: "P18ST", 2023: "P16ST", 2024: "P16ST"},
    "job_concern": {2011: "S9", 2013: "S5", 2015: "S3", 2016: "S3", 2017: "S4", 2018: "S3", 2020: "S3", 2023: "S4", 2024: "S4"},
    "econ_situation": {2011: "P3ST.A", 2013: "P3STGBS", 2015: "P3STGBS", 2016: "P4STGBS", 2017: "P4STGBSC", 2018: "P6STGBSC", 2020: "P4STGBS", 2023: "P5STGBS", 2024: "P6STGBS"},
    "resp_economic_perception": {2011: "P6ST", 2013: "P6STGBS", 2015: "P6STGBS"},
    "confidence_congress": {2011: "P22ST.A", 2013: "P26TGB.C", 2015: "P16ST.F", 2016: "P13STD", 2017: "P14ST.D", 2018: "P15STGBSC.D", 2020: "P13ST.D", 2023: "P13ST.D", 2024: "P14ST.D"},
    "confidence_judiciary": {2011: "P22ST.B", 2013: "P26TGB.E", 2015: "P16ST.H", 2016: "P13STF", 2017: "P14ST.F", 2018: "P15STGBSC.F", 2020: "P13ST.F", 2023: "P13ST.F", 2024: "P14ST.F"},
    "confidence_church": {2011: "P22ST.G", 2013: "P28ST.E", 2015: "P16ST.E", 2016: "P13STC", 2017: "P14ST.C", 2018: "P15STGBSC.C", 2020: "P13ST.C", 2023: "P13ST.C", 2024: "P14ST.C"},
    "confidence_police": {2011: "P20ST.C", 2013: "P28TGB.B", 2015: "P16TGB.B", 2016: "P13STGBSB", 2017: "P14STGBS.B", 2018: "P15STGBSC.B", 2020: "P13STGBS.B", 2023: "P13STGBS.B", 2024: "P14STGBS.B"},
    "confidence_army": {2011: "P22ST.D", 2013: "P28TGB.A", 2015: "P16TGB.A", 2016: "P13STGBSA", 2017: "P14STGBS.A", 2018: "P15STGBSC.A", 2020: "P13STGBS.A", 2023: "P13STGBS.A", 2024: "P14STGBS.A"},
    "confidence_political_parties": {2011: "P22ST.C", 2013: "P26TGB.G", 2015: "P19ST.C", 2016: "P13STG", 2017: "P14ST.G", 2018: "P15STGBSC.G", 2020: "P13ST.G", 2023: "P13ST.G", 2024: "P14ST.G"},
    "resp_sex": {2011: "S16", 2013: "S10", 2015: "S12", 2016: "SEXO", 2017: "SEXO", 2018: "SEXO", 2020: "SEXO", 2023: "SEXO", 2024: "SEXO"},
    "resp_age": {2011: "S17", 2013: "S11", 2015: "S13", 2016: "EDAD", 2017: "EDAD", 2018: "EDAD", 2020: "EDAD", 2023: "EDAD", 2024: "EDAD"},
    "resp_education": {2011: "REEDUC1", 2013: "REEDUC_1", 2015: "REEDUC_1", 2016: "REEDUC_1", 2017: "REEDUC.1", 2020: "REEDUC.1", 2023: "REEEDUC.1", 2024: "REEDUC.1"},
    "resp_employment": {2011: "S23A", 2013: "S19.A", 2015: "S21.A", 2016: "S18A", 2017: "S18.A", 2018: "S14A", 2020: "S24.A", 2023: "S18.A", 2024: "S18.A"},
    "resp_religion": {2011: "S18", 2013: "S14", 2015: "S16", 2016: "S8", 2017: "S9", 2018: "S5", 2020: "S10", 2023: "S1", 2024: "S1"},
    "goods_wash_mach": {2011: "S28E", 2013: "S22.E", 2015: "S24.E", 2016: "S16E", 2017: "S17.E", 2018: "S21.E", 2020: "S26.D", 2023: "S20.C", 2024: "S20C"},
    "goods_car": {2011: "S28H", 2013: "S22.H", 2015: "S24.I", 2016: "S16I", 2017: "S17.I", 2018: "S21.I", 2020: "S26.G", 2023: "S20.E", 2024: "S20E"},
    "goods_sewage": {2011: "S28K", 2013: "S22.K", 2015: "S24.K", 2016: "S16K", 2017: "S17.K", 2018: "S21.K", 2020: "S26.I", 2023: "S20.G", 2024: "S20G"},
    "goods_hot_water": {2011: "S28J", 2013: "S22.J", 2015: "S24.J", 2016: "S16J", 2017: "S17.J", 2018: "S21.J", 2020: "S26.H", 2023: "S20.F", 2024: "S20F"},
    "resp_chief": {2011: "S19", 2013: "S15", 2015: "S17", 2016: "S11", 2017: "S12", 2018: "S8", 2020: "S14", 2023: "S9", 2024: "S9"},
}


def _normalizar(nombre: str) -> str:
    """Normaliza un nombre de columna para comparar sin importar mayúsculas,
    puntos ni guiones bajos (las mismas variables aparecen escritas de forma
    ligeramente distinta entre olas: 'P14ST.A' vs 'p12st.a' vs 'S16E')."""
    return nombre.lower().replace(".", "").replace("_", "")


def _leer_ola_csv(anio: int) -> pd.DataFrame:
    """
    Lee una ola 2007-2010 desde el CSV crudo y aplica la corrección de
    corrimiento de -1 columna descubierta y verificada (ver docstring del
    módulo): el valor real de cada variable está en la columna
    INMEDIATAMENTE ANTERIOR a la que cita el Libro de Códigos.
    """
    ruta = RAW_DIR / ARCHIVOS_CSV[anio]
    df = pd.read_csv(ruta, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]
    columnas = list(df.columns)
    mapa_normalizado = {_normalizar(c): c for c in columnas}

    salida = pd.DataFrame(index=df.index)
    for variable, mapa_anios in CROSSWALK_CSV.items():
        citada = mapa_anios.get(anio)
        if citada is None:
            continue  # variable no disponible esta ola (ver resp_economic_perception)
        col_citada = mapa_normalizado.get(_normalizar(citada))
        if col_citada is None:
            raise ValueError(
                f"[{anio}] La columna citada por el codebook '{citada}' (variable "
                f"'{variable}') no existe en {ruta.name}. Revisa si cambió el crosswalk."
            )
        posicion = columnas.index(col_citada)
        if posicion == 0:
            raise ValueError(
                f"[{anio}] '{citada}' está en la primera columna del archivo; no hay "
                "columna anterior a la cual aplicar la corrección de corrimiento."
            )
        columna_real = columnas[posicion - 1]
        salida[variable] = df[columna_real]

    salida["research_year"] = anio
    return salida


def _leer_ola_sav(anio: int) -> pd.DataFrame:
    """Lee una ola 2011-2024 directamente del archivo SPSS (.sav), sin
    corrimiento -- el dato ya está correctamente amarrado a su variable
    por la metadata interna del formato."""
    coincidencias = glob.glob(str(RAW_DIR / PATRON_SAV.format(anio=anio)))
    if not coincidencias:
        raise FileNotFoundError(f"No se encontró el archivo .sav para la ola {anio}.")
    df, _meta = pyreadstat.read_sav(coincidencias[0])
    mapa_normalizado = {_normalizar(c): c for c in df.columns}

    salida = pd.DataFrame(index=df.index)
    for variable, mapa_anios in CROSSWALK_SAV.items():
        citada = mapa_anios.get(anio)
        if citada is None:
            continue
        col_real = mapa_normalizado.get(_normalizar(citada))
        if col_real is None:
            raise ValueError(
                f"[{anio}] La columna '{citada}' (variable '{variable}') no existe en "
                f"el archivo .sav de esa ola. Revisa si cambió el crosswalk."
            )
        salida[variable] = df[col_real].astype("string")

    salida["research_year"] = anio
    return salida


def construir_dataset_corregido() -> pd.DataFrame:
    partes = [_leer_ola_csv(anio) for anio in sorted(ANIOS_CSV)]
    partes += [_leer_ola_sav(anio) for anio in sorted(ANIOS_SAV)]
    df = pd.concat(partes, ignore_index=True, sort=False)
    df["country_name"] = "Ecuador"
    df["iso3"] = "ECU"
    return df


def ejecutar(out_path: Path = OUT_PATH) -> pd.DataFrame:
    df = construir_dataset_corregido()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[latinobarometro_loader_crudo] {len(df)} filas escritas en '{out_path}'.")
    print(df["research_year"].value_counts().sort_index())
    return df


if __name__ == "__main__":
    ejecutar()
