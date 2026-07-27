"""
vdem_processing.py
---------------------------------------------------------------------
Fase 2 (Preparación de datos) - CRISP-DM
Tesis: Deep Learning for Modeling Democratic Satisfaction and
Socioeconomic Inequality in Ecuador.

Procesa el extracto país-año de V-Dem para Ecuador (ver
sql/01_vdem_ecuador.sql) y lo deja listo para el merge por 'anio' con
Latinobarómetro (persona) y los indicadores nacionales anuales de ENEMDU.

Punto verificado explícitamente (no asumido): la Ficha Metodológica de
V-Dem indica que las variables de distribución del poder (v2pepwrses,
v2pepwrsoc, v2pepwrgen, v2pepwrort, v2pepwrgeo) se reescalan de su
escala original [-4, 4] a [0, 1] mediante x' = (x+4)/8 DENTRO de la
tabla indicadores.vdem en Superset. Pero la muestra V-Dem de referencia
usada para el desarrollo (ec15ec51...csv) SÍ trae estas columnas en su
escala original [-4, 4] (se observan valores negativos, p. ej. -0.658).
Es decir: no hay certeza de si lo que devuelve la consulta real de
Superset ya viene reescalado o no. Por eso esta función NO asume un
estado fijo: detecta el rango observado y reescala solo si hace falta,
dejando además una advertencia explícita en consola.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/vdem_ecuador.csv")  # salida de sql/01_vdem_ecuador.sql
OUT_PATH = Path("data/processed/vdem_ecuador_anual.csv")

COLUMNAS_INDICE_0_1 = [
    "v2x_polyarchy", "v2x_libdem", "v2x_partipdem", "v2x_delibdem", "v2x_egaldem",
    "v2x_freexp_altinf", "v2xel_frefair", "v2xcl_rol", "v2x_jucon", "v2xlg_legcon",
    "v2xeg_eqprotec", "v2xeg_eqaccess", "v2xeg_eqdr",
]
COLUMNAS_POSIBLEMENTE_SIN_RESCALAR = [
    "v2pepwrses", "v2pepwrsoc", "v2pepwrgen", "v2pepwrort", "v2pepwrgeo",
]


def cargar_datos(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No se encontró '{raw_path}'. Ejecuta sql/01_vdem_ecuador.sql en "
            "Superset y guarda el resultado en esa ruta."
        )
    # sep=None + engine="python": detecta solo si el archivo viene separado
    # por comas o por tabuladores (según cómo se haya exportado desde
    # Superset -- ver el mismo problema encontrado en enemdu_processing.py).
    df = pd.read_csv(raw_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df


def verificar_y_rescalar_distribucion_poder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica el rango observado de cada columna v2pepwr*. Si el mínimo
    observado es claramente menor que 0 (indicio de escala original
    [-4,4] sin reescalar), aplica x' = (x+4)/8. Si ya está en [0,1], no
    toca nada. En ambos casos imprime lo que decidió, para que quede
    trazable en el log de ejecución y sea verificable contra la
    documentación real de la extracción de Superset.
    """
    df = df.copy()
    for c in COLUMNAS_POSIBLEMENTE_SIN_RESCALAR:
        if c not in df.columns:
            continue
        minimo, maximo = df[c].min(), df[c].max()
        necesita_rescalar = minimo < -0.01 or maximo > 1.01
        if necesita_rescalar:
            print(
                f"[vdem_processing] '{c}': rango observado [{minimo:.3f}, {maximo:.3f}] "
                "fuera de [0,1] -> se aplica reescalamiento x' = (x+4)/8 (Ficha Metodológica V-Dem)."
            )
            df[c] = (df[c] + 4) / 8
        else:
            print(
                f"[vdem_processing] '{c}': rango observado [{minimo:.3f}, {maximo:.3f}] "
                "ya está en [0,1] -> no se reescala (asumido ya procesado por Superset)."
            )
    return df


def verificar_rango_indices_democracia(df: pd.DataFrame) -> None:
    """Solo verificación/advertencia; los índices v2x_* deben venir ya en [0,1] sin transformación."""
    for c in COLUMNAS_INDICE_0_1:
        if c not in df.columns:
            continue
        fuera_rango = df[(df[c] < -0.001) | (df[c] > 1.001)]
        if len(fuera_rango):
            print(
                f"[vdem_processing] AVISO: {len(fuera_rango)} valores de '{c}' fuera de [0,1]. "
                "Revisar antes de usar (posible error de extracción o de columna)."
            )


def ejecutar_pipeline(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    df = cargar_datos(raw_path)
    verificar_rango_indices_democracia(df)
    df = verificar_y_rescalar_distribucion_poder(df)
    df = df.rename(columns={"year": "anio"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    resultado = ejecutar_pipeline()
    print(resultado.describe().T)
