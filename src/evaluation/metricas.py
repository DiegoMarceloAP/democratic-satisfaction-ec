"""
metricas.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Tabla comparativa final de los 4
modelos del plan de modelado (XGBoost, LightGBM, CNN-LSTM, TabNet).

Cada modelo ya guarda sus resultados POR FOLD de Time Series Split en
reports/tablas/resultados_<modelo>.csv (ver guardar_resultados() en
cada src/models/*.py). Este módulo los carga, promedia entre folds
-- reportando también la desviación estándar, para mostrar qué tan
ESTABLE es cada modelo entre años (relevante porque el balance de
clases cambia mucho por año, ver notebooks/01_eda_balance_clases.ipynb:
un modelo con buen promedio pero alta desviación es menos confiable que
uno más parejo) -- y arma la tabla final con las métricas pedidas por
el plan de modelado: Accuracy, F1 (Macro/Weighted), F1 de la clase
"Satisfecho" (métrica secundaria de la propuesta), PR-AUC (métrica
reina de selección de modelos) y tiempo de entrenamiento.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUTA_TABLAS = Path("reports/tablas")
RUTA_FIGURES = Path("reports/figures")

# Nombre a mostrar -> archivo de resultados por fold (guardado por cada
# src/models/*.py). Editar aquí si se agrega un modelo nuevo.
RESULTADOS_POR_MODELO = {
    "XGBoost": "resultados_baseline_xgboost.csv",
    "LightGBM": "resultados_baseline_lightgbm.csv",
    "CNN-LSTM": "resultados_cnn_lstm.csv",
    "TabNet": "resultados_tabnet.csv",
}

METRICAS_A_PROMEDIAR = [
    "accuracy", "f1_macro", "f1_weighted", "f1_satisfecho", "pr_auc", "tiempo_entrenamiento_seg",
]


def cargar_resultados_modelo(nombre_modelo: str, archivo: str, ruta_tablas: Path = RUTA_TABLAS) -> pd.DataFrame:
    path = ruta_tablas / archivo
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path}' -- corre primero src/models/ correspondiente "
            f"para generar los resultados por fold de {nombre_modelo}."
        )
    df = pd.read_csv(path)
    df.insert(0, "modelo", nombre_modelo)
    return df


def construir_tabla_comparativa(
    modelos: dict = RESULTADOS_POR_MODELO,
    ruta_tablas: Path = RUTA_TABLAS,
    ordenar_por: str = "pr_auc",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna (tabla_comparativa, resultados_detallados):
        tabla_comparativa: una fila por modelo, con la media y desviación
            estándar (entre folds de Time Series Split) de cada métrica.
            Ordenada de mayor a menor 'ordenar_por' (por defecto PR-AUC,
            la métrica reina definida en el plan de modelado).
        resultados_detallados: la concatenación de los resultados POR
            FOLD de todos los modelos disponibles (una fila por modelo
            x fold/año de prueba), útil para inspeccionar la volatilidad
            año a año de cada uno.

    Si a algún modelo le falta su CSV de resultados, se avisa por
    consola y se arma la tabla solo con los modelos disponibles (no
    bloquea comparar los que sí terminaron de correr).
    """
    detalles = []
    faltantes = []
    for nombre_modelo, archivo in modelos.items():
        try:
            detalles.append(cargar_resultados_modelo(nombre_modelo, archivo, ruta_tablas))
        except FileNotFoundError:
            faltantes.append(nombre_modelo)

    if faltantes:
        print(f"[metricas] AVISO: no se encontraron resultados de {faltantes} -- se arma la tabla solo con los modelos disponibles.")
    if not detalles:
        raise RuntimeError("No hay resultados de NINGÚN modelo todavía -- corre al menos uno primero (ver src/models/).")

    resultados_detallados = pd.concat(detalles, ignore_index=True)

    resumen = resultados_detallados.groupby("modelo")[METRICAS_A_PROMEDIAR].agg(["mean", "std"])
    resumen.columns = [f"{metrica}_{stat}" for metrica, stat in resumen.columns]
    resumen = resumen.round(4).reset_index()

    n_folds = resultados_detallados.groupby("modelo").size().rename("n_folds").reset_index()
    resumen = resumen.merge(n_folds, on="modelo")

    resumen = resumen.sort_values(f"{ordenar_por}_mean", ascending=False).reset_index(drop=True)

    columnas_orden = [
        "modelo", "n_folds",
        "accuracy_mean", "accuracy_std",
        "f1_macro_mean", "f1_macro_std",
        "f1_weighted_mean", "f1_weighted_std",
        "f1_satisfecho_mean", "f1_satisfecho_std",
        "pr_auc_mean", "pr_auc_std",
        "tiempo_entrenamiento_seg_mean", "tiempo_entrenamiento_seg_std",
    ]
    resumen = resumen[columnas_orden]

    print("\n[metricas] Tabla comparativa final (ordenada por PR-AUC promedio -- métrica reina del plan de modelado):")
    print(resumen.to_string(index=False))

    return resumen, resultados_detallados


def graficar_comparacion(
    tabla_comparativa: pd.DataFrame, metrica: str = "pr_auc", ruta_figures: Path = RUTA_FIGURES
) -> Path:
    """Barra comparativa de 'metrica' (por defecto PR-AUC) entre modelos, con barras de error = desviación estándar entre folds."""
    ruta_figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        tabla_comparativa["modelo"], tabla_comparativa[f"{metrica}_mean"],
        yerr=tabla_comparativa[f"{metrica}_std"], capsize=4, color="#2980b9",
    )
    ax.set_ylabel(metrica.upper().replace("_", "-"))
    ax.set_title(f"Comparación de modelos -- {metrica} promedio (Time Series Split)")
    fig.tight_layout()
    out_path = ruta_figures / f"comparacion_modelos_{metrica}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[metricas] Gráfico guardado en '{out_path}'.")
    return out_path


def guardar_tabla_comparativa(tabla_comparativa: pd.DataFrame, ruta_tablas: Path = RUTA_TABLAS) -> Path:
    ruta_tablas.mkdir(parents=True, exist_ok=True)
    out_path = ruta_tablas / "tabla_comparativa_final.csv"
    tabla_comparativa.to_csv(out_path, index=False)
    print(f"[metricas] Tabla comparativa guardada en '{out_path}'.")
    return out_path


if __name__ == "__main__":
    tabla, detalles = construir_tabla_comparativa()
    guardar_tabla_comparativa(tabla)
    graficar_comparacion(tabla, metrica="pr_auc")
    graficar_comparacion(tabla, metrica="f1_satisfecho")
