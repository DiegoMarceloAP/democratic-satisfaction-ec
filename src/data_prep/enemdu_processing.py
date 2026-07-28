"""
enemdu_processing.py
---------------------------------------------------------------------
Fase 2 (Preparación de datos) - CRISP-DM.

Procesa los microdatos crudos de personas de ENEMDU extraídos desde
Superset (ver sql/04_enemdu_persona_por_periodo.sql, un CSV por periodo)
y calcula indicadores agregados ANUALES A NIVEL NACIONAL de mercado
laboral, ingreso y desigualdad.

Decisión de alcance (ver sql/README_extraccion.md):
el merge final con Latinobarómetro y V-Dem es por año a nivel nacional,
NO por provincia. Por eso este módulo sí deriva el código de provincia
(trazabilidad/QA y por si más adelante se decide una desagregación
subnacional), pero agrega los indicadores directamente a nivel
nacional-año.

Supuestos de codificación verificados contra la muestra adjunta
(35048dcd...csv), NO contra el 100% de los microdatos reales -- revisar
si algo choca al correr con la extracción completa:
    - pobreza / epobreza: 0 = no pobre, 1 = pobre (confirmado en muestra).
    - secemp: 1 = formal, 2 = informal (consistente con la muestra y con
      la Ficha Metodológica ENEMDU, que además define 3 = doméstico).
      El código 4 observado en la muestra no está documentado en la
      ficha; se excluye de formal/informal hasta confirmar su significado.
    - p03 = edad; código 99 = "No sabe" (Guía ENEMDU, Tabla 14) -> se
      trata como NA.

NOTA METODOLÓGICA -- CÁLCULO ANUAL:
ENEMDU es una encuesta con muestreo probabilístico complejo, NO un censo,
y su periodicidad cambió a lo largo de la serie histórica: rondas
TRIMESTRALES (marzo/junio/septiembre/diciembre) de 2007 a 2019, y encuesta
CONTINUA MENSUAL desde septiembre de 2020 en adelante (confirmado contra
los periodos realmente extraídos en data/raw/enemdu_persona/; también se
detectaron 2 meses genuinamente ausentes en la fuente -- noviembre 2021 y
octubre 2025 -- verificado contra la volumetría real pedida a Superset en
scripts/extraer_enemdu_superset.py, no es un descuido de la extracción).

Tres decisiones concretas sostienen `calcular_indicadores_anuales_nacional`:

1) PONDERACIÓN: ningún indicador es un promedio simple de microdatos.
   Todos usan 'fexp' (factor de expansión) como peso -- participación,
   desempleo, ingreso promedio, Gini, pobreza -- porque cada persona
   encuestada representa a un número distinto de personas en la
   población real; ignorar 'fexp' sesgaría los indicadores por el
   diseño muestral.

2) DEFINICIONES: PET/PEA/ocupado, y por tanto tasa de participación,
   tasa de desempleo y empleo formal/informal, replican textualmente la
   Ficha Metodológica ENEMDU (PET: edad>=15 y fexp>0; PEA: PET y
   condact en 1..8; ocupado: PET y condact en 1..6) -- no son fórmulas
   ad hoc. El Gini se calcula sobre 'ingpc' con curva de Lorenz
   ponderada (enfoque estándar para datos con pesos muestrales, ver
   Cowell, 2011, y _gini_ponderado() abajo).

3) "ANUALIZACIÓN" por agrupación de rondas (el punto metodológico más
   relevante de esta nota): `calcular_indicadores_anuales_nacional` agrupa
   TODAS las filas cuyo 'periodo' (YYYYMM) cae en el mismo año calendario
   y calcula un único indicador ponderado por 'fexp' sobre ese grupo. En
   la práctica esto significa que los años 2007-2019 agregan ~4 rondas
   trimestrales pooled, y los años 2020 en adelante agregan hasta 12
   rondas mensuales pooled -- cada ronda ya viene expandida correctamente
   a la población nacional por su propio 'fexp', así que agrupar todas
   las rondas de un año y ponderar es una forma válida de anualizar.
   LIMITACIÓN A DECLARAR EN LA TESIS: esto es una decisión metodológica
   explícita, NO necesariamente idéntica al indicador anual oficial que
   publica el INEC (que en ciertos reportes usa solo la ronda de
   diciembre, con muestra ampliada para representatividad provincial, en
   vez de agrupar todas las rondas disponibles del año). La columna
   'n_personas_muestra' del resultado queda como trazabilidad -- permite
   reportar cuántas filas/rondas aportó cada año.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/enemdu_persona")  # un CSV por periodo (salida de Superset)
DICCIONARIO_PROVINCIAS_PATH = Path("data/raw/diccionario_provincias.csv")
OUT_PATH = Path("data/processed/enemdu_indicadores_anuales.csv")

COLUMNAS_FLOAT = ["ingpc", "ingrl", "fexp"]
COLUMNAS_INT = [
    "estrato", "condact", "empleo", "desempleo", "secemp", "rama1",
    "pobreza", "epobreza", "p03", "nnivins",
]


def _leer_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    """
    Lee un CSV detectando automáticamente el separador (coma, punto y
    coma o tabulador). Necesario porque, según cómo se exporte el
    resultado desde Superset (botón de descarga vs. "Copy to Clipboard"
    de la grilla de SQL Lab), el archivo puede terminar separado por
    comas o por tabuladores -- sep=None + engine="python" hace que
    pandas lo detecte solo en vez de asumir coma a ciegas.
    """
    return pd.read_csv(path, sep=None, engine="python", **kwargs)


def cargar_periodos(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatena todos los CSV de periodo descargados desde Superset."""
    archivos = sorted(raw_dir.glob("enemdu_persona_*.csv"))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron CSV en '{raw_dir}'. Ejecuta primero cada consulta "
            "de sql/04_enemdu_persona_por_periodo.sql en Superset (una por periodo) "
            "y guarda los resultados como enemdu_persona_<periodo>.csv en esa carpeta."
        )
    dfs = [_leer_csv_flexible(f, dtype=str) for f in archivos]
    return pd.concat(dfs, ignore_index=True)


def tipificar(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte tipos explícitamente (equivalente a toInt32OrNull/toFloat64OrZero de ClickHouse)."""
    df = df.copy()
    for c in COLUMNAS_FLOAT:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in COLUMNAS_INT:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # p03 = 99 es código de "No sabe" para edad (Guía ENEMDU, Tabla 14) -> NA
    df.loc[df["p03"] == 99, "p03"] = pd.NA
    return df


def derivar_geografia(df: pd.DataFrame, diccionario_provincias: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva provincia/cantón/parroquia a partir de 'ciudad', replicando la
    lógica documentada en la Ficha Metodológica ENEMDU:
        geo_digits = solo dígitos de 'ciudad', normalizados a 6 caracteres
        provincia  = primeros 2 dígitos
        canton     = primeros 4 dígitos
        parroquia  = primeros 6 dígitos
    Se conserva como columna descriptiva/QA; el merge final es nacional-año.
    """
    df = df.copy()
    geo_digits = df["ciudad"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    df["cod_provincia"] = geo_digits.str.slice(0, 2)
    df["cod_canton"] = geo_digits.str.slice(0, 4)
    df["cod_parroquia"] = geo_digits.str.slice(0, 6)

    dicc = diccionario_provincias[["CodigoProvincia", "NombreProvincia"]].drop_duplicates()
    df = df.merge(dicc, left_on="cod_provincia", right_on="CodigoProvincia", how="left")
    return df.drop(columns=["CodigoProvincia"])


def derivar_condicion_actividad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Banderas PET / PEA / ocupado, según la Ficha Metodológica ENEMDU:
        is_pet = edad >= 15 y fexp > 0
        is_pea = is_pet y condact in [1..8]
        is_occ = is_pet y condact in [1..6]
    Se usa .fillna(False) porque las comparaciones con edad faltante (NA)
    producen valores nulos en dtype "boolean", y estas columnas se usan
    luego como máscaras booleanas puras en .loc[].
    """
    df = df.copy()
    edad = df["p03"]
    fexp_valido = df["fexp"] > 0
    df["is_pet"] = ((edad >= 15) & fexp_valido).fillna(False)
    df["is_pea"] = (df["is_pet"] & df["condact"].between(1, 8)).fillna(False)
    df["is_occ"] = (df["is_pet"] & df["condact"].between(1, 6)).fillna(False)
    return df


def _media_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    m = valores.notna() & pesos.notna() & (pesos > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.average(valores[m], weights=pesos[m]))


def _gini_ponderado(valores: pd.Series, pesos: pd.Series) -> float:
    """
    Aproximación de desigualdad: coeficiente de Gini ponderado por el
    factor de expansión, calculado sobre el ingreso per cápita del hogar
    (ingpc). Se construye la curva de Lorenz ponderada y se integra por
    trapecios (equivalente a Cowell, 2011, para datos con pesos muestrales).
    """
    m = valores.notna() & pesos.notna() & (pesos > 0) & (valores >= 0)
    v, w = valores[m].to_numpy(dtype=float), pesos[m].to_numpy(dtype=float)
    if len(v) < 2:
        return np.nan
    orden = np.argsort(v)
    v, w = v[orden], w[orden]
    w_acum = np.cumsum(w)
    w_total = w_acum[-1]
    vw_acum = np.cumsum(v * w)
    vw_total = vw_acum[-1]
    if w_total == 0 or vw_total == 0:
        return np.nan
    lorenz = np.concatenate([[0.0], vw_acum / vw_total])
    pesos_acum_norm = np.concatenate([[0.0], w_acum / w_total])
    area_bajo_lorenz = np.sum(
        (pesos_acum_norm[1:] - pesos_acum_norm[:-1]) * (lorenz[1:] + lorenz[:-1]) / 2
    )
    return 1 - 2 * area_bajo_lorenz


def calcular_indicadores_anuales_nacional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a nivel nacional-año los tres bloques de indicadores que pide
    la Fase 2 de la propuesta: mercado laboral, ingreso promedio y
    aproximación de desigualdad. Incluye además pobreza por ingresos como
    indicador complementario, ya disponible directamente en la tabla
    (columnas 'pobreza'/'epobreza').
    """
    df = df.copy()
    df["anio"] = df["periodo"].str.slice(0, 4).astype(int)

    filas = []
    for anio, g in df.groupby("anio"):
        pea = g.loc[g["is_pea"], "fexp"].sum()
        pet = g.loc[g["is_pet"], "fexp"].sum()
        occ = g.loc[g["is_occ"], "fexp"].sum()
        formal = g.loc[g["is_occ"] & (g["secemp"] == 1), "fexp"].sum()
        informal = g.loc[g["is_occ"] & (g["secemp"] == 2), "fexp"].sum()

        pobreza_valida = g["pobreza"].notna()
        epobreza_valida = g["epobreza"].notna()

        filas.append({
            "anio": anio,
            "tasa_participacion_global": 100 * pea / pet if pet else np.nan,
            "tasa_desempleo": 100 * (pea - occ) / pea if pea else np.nan,
            "empleo_formal": 100 * formal / occ if occ else np.nan,
            "empleo_informal": 100 * informal / occ if occ else np.nan,
            "ingreso_promedio_pc": _media_ponderada(g["ingpc"], g["fexp"]),
            "ingreso_promedio_laboral": _media_ponderada(
                g.loc[g["is_occ"], "ingrl"], g.loc[g["is_occ"], "fexp"]
            ),
            "gini_ingpc": _gini_ponderado(g["ingpc"], g["fexp"]),
            "pobreza_ingresos": (
                100 * g.loc[g["pobreza"] == 1, "fexp"].sum() / g.loc[pobreza_valida, "fexp"].sum()
                if pobreza_valida.any() else np.nan
            ),
            "pobreza_extrema_ingresos": (
                100 * g.loc[g["epobreza"] == 1, "fexp"].sum() / g.loc[epobreza_valida, "fexp"].sum()
                if epobreza_valida.any() else np.nan
            ),
            "n_personas_muestra": len(g),
        })
    return pd.DataFrame(filas).sort_values("anio").reset_index(drop=True)


def ejecutar_pipeline(
    raw_dir: Path = RAW_DIR,
    diccionario_provincias_path: Path = DICCIONARIO_PROVINCIAS_PATH,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    """Corre el pipeline completo y guarda el resultado en out_path."""
    df = cargar_periodos(raw_dir)
    df = tipificar(df)
    dicc = _leer_csv_flexible(diccionario_provincias_path, dtype=str)
    dicc.columns = dicc.columns.str.strip()
    df = derivar_geografia(df, dicc)
    df = derivar_condicion_actividad(df)
    indicadores = calcular_indicadores_anuales_nacional(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indicadores.to_csv(out_path, index=False)
    return indicadores


if __name__ == "__main__":
    resultado = ejecutar_pipeline()
    print(resultado.to_string())
