"""
latinobarometro_processing.py
---------------------------------------------------------------------
Fase 2 (Preparación de datos) - CRISP-DM
Tesis: Deep Learning for Modeling Democratic Satisfaction and
Socioeconomic Inequality in Ecuador.

Procesa los microdatos de encuestados de Latinobarómetro Ecuador y
construye:
  1) La variable objetivo binaria de la tesis: "Satisfecho" vs.
     "No Satisfecho" con la democracia, a partir de democ_satis.
  2) Un conjunto de variables auxiliares limpias (sin códigos de no
     respuesta) para usar como features sociodemográficas/actitudinales.

A diferencia de enemdu_processing.py, aquí NO se agrega a nivel
nacional-año: la unidad de análisis del problema de clasificación es el
encuestado individual. La columna 'anio' (= research_year) es la llave
que luego se usa para pegar los indicadores nacionales anuales de ENEMDU
y V-Dem (decisión de merge nacional-año, ver sql/README_extraccion.md).

FUENTE (revisión con tutor): los datos de Latinobarómetro para Ecuador
ya NO vienen de Superset (la extracción SQL en
sql/02_latinobarometro_ecuador.sql tenía inconsistencias y no se pudo
utilizar -- ver limitaciones de la tesis). RAW_PATH apunta al archivo
generado por latinobarometro_loader_crudo.py a partir de las descargas
directas de Latinobarómetro (CSV crudo 2007-2010, SPSS .sav 2011-2024),
que trae los 13 años reales correctamente etiquetados y verificados
variable por variable contra cada Libro de Códigos oficial. Ver el
docstring de ese módulo para el detalle del crosswalk.

Variables que quedan fuera de este pipeline (decisión explícita, no un
descuido): elections_vote (solo tiene dato real en 1 de las 13 olas).
resp_economic_perception se conserva pese a solo estar disponible en
6 de las 13 olas (hueco estructural del cuestionario, documentado más
abajo).

Códigos de no respuesta (verificados contra el archivo crudo completo
de Ecuador, 13 olas x 1200 encuestados):
    - La gran mayoría de variables usan códigos negativos (-1, -2, -3,
      -5, -6, -8) para "no sabe / no responde / no aplica".
    - left_right_scale y resp_religion además usan códigos altos de
      "cajón de sastre" (97 = no sabe/no contesta) que también se tratan
      como NA.
    - resp_education y resp_employment: el único código negativo
      observado es -2; se usa el set genérico completo por robustez.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/latinobarometro_ecuador_corregido.csv")  # salida de latinobarometro_loader_crudo.py
OUT_PATH = Path("data/processed/latinobarometro_ecuador_personas.csv")
OUT_SERIE_ANUAL_PATH = Path("data/processed/latinobarometro_ecuador_serie_anual.csv")

ANIO_MIN, ANIO_MAX = 2007, 2024  # rango de la tesis, no el rango real de olas disponibles

# Máximo de años consecutivos sin ola que se permite rellenar por
# interpolación lineal. Decisión metodológica: huecos de 1-2 años se
# interpolan; huecos más largos se dejan como faltante explícito
# en vez de forzar una tendencia sin respaldo empírico (así lo exige la
# propia propuesta de tesis). El manejo de huecos largos en la modelación
# secuencial queda a cargo de las ventanas dinámicas del CNN-LSTM, no de
# este script.
MAX_HUECO_INTERPOLABLE = 2

# Códigos de no respuesta por variable (verificados en la muestra Ecuador)
CODIGOS_NA_GENERICOS = {-1, -2, -3, -5, -6, -8}
CODIGOS_NA_POR_COLUMNA = {
    "left_right_scale": CODIGOS_NA_GENERICOS | {97},
    "resp_religion": CODIGOS_NA_GENERICOS | {96, 97},
}

# NOTA: elections_vote se descarta permanentemente (solo 1/13 olas tiene
# dato real). Las otras 5 -- goods_wash_mach, goods_car, goods_sewage,
# goods_hot_water, resp_chief -- SÍ están disponibles en la fuente cruda
# (ver latinobarometro_loader_crudo.py) y se incluyen aquí.
COLUMNAS_NUMERICAS = [
    "democ_satis", "democ_supp", "left_right_scale", "econ_situation",
    "job_concern", "resp_economic_perception",
    "confidence_congress", "confidence_judiciary", "confidence_church",
    "confidence_police", "confidence_army", "confidence_political_parties",
    "resp_sex", "resp_age", "resp_education",
    "resp_employment", "resp_religion",
    "goods_wash_mach", "goods_car", "goods_sewage", "goods_hot_water",
    "resp_chief",
]


def cargar_datos(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No se encontró '{raw_path}'. Ejecuta "
            "src/data_prep/latinobarometro_loader_crudo.py primero para generarlo "
            "a partir de las descargas directas de Latinobarómetro."
        )
    # sep=None + engine="python": tolera comas o tabuladores por si el
    # formato de exportación cambia -- ver el mismo problema encontrado en
    # enemdu_processing.py.
    df = pd.read_csv(raw_path, sep=None, engine="python", dtype=str)
    df.columns = df.columns.str.strip()
    return df


def tipificar_y_limpiar_na(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a numérico y reemplaza los códigos de no respuesta por NA."""
    df = df.copy()
    for c in COLUMNAS_NUMERICAS:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        codigos_na = CODIGOS_NA_POR_COLUMNA.get(c, CODIGOS_NA_GENERICOS)
        df.loc[df[c].isin(codigos_na), c] = np.nan
    return df


def construir_target_satisfaccion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variable objetivo de la tesis (clasificación binaria):
        democ_satis = 1 (muy satisfecho) o 2 (más bien satisfecho) -> Satisfecho (1)
        democ_satis = 3 (no muy satisfecho) o 4 (nada satisfecho)  -> No Satisfecho (0)
        cualquier otro valor / NA                                   -> se excluye (NA)
    Códigos verificados en la muestra Ecuador de la Ficha Metodológica
    Latinobarómetro (sección 2.3).
    """
    df = df.copy()
    satisfecho = df["democ_satis"].isin([1, 2])
    no_satisfecho = df["democ_satis"].isin([3, 4])
    df["satisfecho_democracia"] = pd.NA
    df.loc[satisfecho, "satisfecho_democracia"] = 1
    df.loc[no_satisfecho, "satisfecho_democracia"] = 0
    df["satisfecho_democracia"] = df["satisfecho_democracia"].astype("Int64")
    return df


def recodificar_auxiliares(df: pd.DataFrame) -> pd.DataFrame:
    """Recodificaciones de apoyo, siguiendo las definiciones de la Ficha Metodológica Latinobarómetro."""
    df = df.copy()

    # Apoyo a la democracia (2.1/2.2): 1=prefiere democracia, 2=prefiere autoritarismo, 3=indiferente
    mapa_democ_supp = {1: "prefiere_democracia", 2: "prefiere_autoritarismo", 3: "indiferente"}
    df["democ_supp_cat"] = df["democ_supp"].map(mapa_democ_supp)

    # Ubicación ideológica (2.4): izquierda 0-4, centro 5-6, derecha 7-10
    def bucket_ideologia(x):
        if pd.isna(x):
            return pd.NA
        if x <= 4:
            return "izquierda"
        if x <= 6:
            return "centro"
        return "derecha"
    df["ideologia_cat"] = df["left_right_scale"].apply(bucket_ideologia)

    # Evaluación económica del país (2.5): 1-2 positiva, 3 regular, 4-5 negativa
    def bucket_economia(x):
        if pd.isna(x):
            return pd.NA
        if x <= 2:
            return "positiva"
        if x == 3:
            return "regular"
        return "negativa"
    df["econ_situation_cat"] = df["econ_situation"].apply(bucket_economia)
    df["resp_economic_perception_cat"] = df["resp_economic_perception"].apply(bucket_economia)

    # Bienes del hogar (proxy de condición material, código 1=Sí/2=No en
    # las 13 olas): se recodifican a booleano igual que la confianza
    # institucional, para usarlas directamente como feature binaria.
    for c in ["goods_wash_mach", "goods_car", "goods_sewage", "goods_hot_water"]:
        df[f"{c}_bin"] = (df[c] == 1).where(df[c].notna())

    # Confianza institucional (2.9): 1-2 = alta, 3-4 = baja
    for c in ["confidence_congress", "confidence_judiciary", "confidence_church",
              "confidence_police", "confidence_army", "confidence_political_parties"]:
        df[f"{c}_alta"] = df[c].isin([1, 2]).where(df[c].notna())

    return df


def construir_serie_anual_nacional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a nivel nacional-año un conjunto de indicadores de Latinobarómetro,
    para usarlos como variables de CONTEXTO/rezago histórico en el
    componente temporal (CNN-LSTM), separado del dataset a nivel persona.

    Importante (Ficha Metodológica Latinobarómetro, sección de limitaciones):
    esta encuesta NO trae un factor de expansión/diseño en las columnas
    disponibles, así que los porcentajes son proporciones muestrales
    simples (no ponderadas), a diferencia de ENEMDU.
    """
    df = df.copy()
    filas = []
    for anio, g in df.groupby("anio"):
        def tasa(mask_valida, mask_positiva):
            n_validos = mask_valida.sum()
            return 100 * mask_positiva.sum() / n_validos if n_validos else np.nan

        filas.append({
            "anio": anio,
            "tasa_satisfaccion_democracia": tasa(
                df.loc[g.index, "satisfecho_democracia"].notna(), g["satisfecho_democracia"] == 1
            ),
            "tasa_apoyo_democracia": tasa(g["democ_supp_cat"].notna(), g["democ_supp_cat"] == "prefiere_democracia"),
            "tasa_apoyo_autoritarismo": tasa(g["democ_supp_cat"].notna(), g["democ_supp_cat"] == "prefiere_autoritarismo"),
            "tasa_econ_pais_positiva": tasa(g["econ_situation_cat"].notna(), g["econ_situation_cat"] == "positiva"),
            "confianza_congreso_alta": tasa(g["confidence_congress_alta"].notna(), g["confidence_congress_alta"] == True),  # noqa: E712
            "confianza_judicial_alta": tasa(g["confidence_judiciary_alta"].notna(), g["confidence_judiciary_alta"] == True),  # noqa: E712
            "confianza_partidos_alta": tasa(g["confidence_political_parties_alta"].notna(), g["confidence_political_parties_alta"] == True),  # noqa: E712
            "n_encuestados": len(g),
        })
    return pd.DataFrame(filas).sort_values("anio").reset_index(drop=True)


def completar_vacios_temporales(
    serie_anual: pd.DataFrame,
    anio_min: int = ANIO_MIN,
    anio_max: int = ANIO_MAX,
    max_hueco: int = MAX_HUECO_INTERPOLABLE,
) -> pd.DataFrame:
    """
    Reindexa la serie a TODOS los años del rango de la tesis (anio_min..anio_max)
    y trata los años sin ola de Latinobarómetro así:
      - Huecos CUYA LONGITUD TOTAL es <= 'max_hueco' años: se interpolan
        linealmente entre las dos olas reales que rodean el hueco.
      - Huecos más largos: se dejan COMPLETOS como NaN explícito -- no se
        rellena ni siquiera una parte del hueco. Esto es deliberado: un
        hueco largo no debe interpolarse solo en sus primeros años y
        dejar el resto sin dato, porque eso introduciría una tendencia
        parcial sin respaldo empírico. Por esto NO se usa pandas
        Series.interpolate(limit=...) directamente
        (rellena de forma unidireccional y puede rellenar solo una
        porción de un hueco largo); en su lugar se calcula la longitud
        real de cada hueco entre pares de años observados consecutivos y
        se decide todo-o-nada.
      - Nunca se extrapola antes de la primera ola ni después de la
        última (esos años, si existen en el rango de la tesis, quedan
        como 'faltante').
    Agrega columna 'estado_anio' en {'observado', 'imputado', 'faltante'}
    para trazabilidad -- fundamental para poder documentar en la tesis
    cuáles años son datos reales y cuáles no.
    """
    serie_anual = serie_anual.set_index("anio")
    anios_completos = pd.RangeIndex(anio_min, anio_max + 1, name="anio")
    serie_completa = serie_anual.reindex(anios_completos)

    columnas_indicador = [c for c in serie_completa.columns if c != "n_encuestados"]
    anios_observados = sorted(a for a in anios_completos if a in serie_anual.index)
    estado = {a: ("observado" if a in anios_observados else "faltante") for a in anios_completos}

    for a0, a1 in zip(anios_observados[:-1], anios_observados[1:]):
        longitud_hueco = a1 - a0 - 1
        if not (0 < longitud_hueco <= max_hueco):
            continue  # sin hueco, o hueco demasiado largo para interpolar
        for c in columnas_indicador:
            v0, v1 = serie_completa.loc[a0, c], serie_completa.loc[a1, c]
            if pd.isna(v0) or pd.isna(v1):
                continue
            for anio in range(a0 + 1, a1):
                paso = (anio - a0) / (a1 - a0)
                serie_completa.loc[anio, c] = v0 + (v1 - v0) * paso
        for anio in range(a0 + 1, a1):
            estado[anio] = "imputado"

    serie_completa["estado_anio"] = [estado[a] for a in anios_completos]

    n_obs = list(estado.values()).count("observado")
    n_imp = list(estado.values()).count("imputado")
    n_falt = list(estado.values()).count("faltante")
    print(
        f"[latinobarometro_processing] serie {anio_min}-{anio_max}: "
        f"{n_obs} años observados, {n_imp} imputados (hueco total <= {max_hueco}), "
        f"{n_falt} sin dato (hueco total > {max_hueco}, se deja NaN completo)."
    )

    return serie_completa.reset_index()


def ejecutar_pipeline(
    raw_path: Path = RAW_PATH,
    out_path: Path = OUT_PATH,
    out_serie_anual_path: Path = OUT_SERIE_ANUAL_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = cargar_datos(raw_path)
    df = tipificar_y_limpiar_na(df)
    df = construir_target_satisfaccion(df)
    df = recodificar_auxiliares(df)
    df = df.rename(columns={"research_year": "anio"})
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    if df["anio"].isna().any():
        raise ValueError(
            "Hay filas con 'research_year' no numérico o vacío; revisar la extracción antes de continuar."
        )
    df["anio"] = df["anio"].astype(int)

    n_total = len(df)
    n_validos = df["satisfecho_democracia"].notna().sum()
    print(
        f"[latinobarometro_processing] {n_validos}/{n_total} filas con target válido "
        f"({100 * n_validos / n_total:.1f}%). El resto se descarta al modelar (respuesta no válida)."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    serie_anual = construir_serie_anual_nacional(df)
    serie_completa = completar_vacios_temporales(serie_anual)
    out_serie_anual_path.parent.mkdir(parents=True, exist_ok=True)
    serie_completa.to_csv(out_serie_anual_path, index=False)

    return df, serie_completa


if __name__ == "__main__":
    resultado_personas, resultado_serie = ejecutar_pipeline()
    print(resultado_personas["anio"].value_counts().sort_index())
    print(resultado_personas["satisfecho_democracia"].value_counts(dropna=False))
    print(resultado_serie)
