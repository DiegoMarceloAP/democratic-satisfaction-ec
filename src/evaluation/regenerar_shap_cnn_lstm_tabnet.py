"""
regenerar_shap_cnn_lstm_tabnet.py
---------------------------------------------------------------------
Fase de Evaluación (CRISP-DM) - Script de un solo uso para C5
(observación de revisores sobre la neutralización de SHAP > 10):
entrena CNN-LSTM y TabNet sobre el dataset COMPLETO (SAMPLING_MODE=False,
igual que la corrida oficial de la Tabla 5.1) y vuelve a calcular SHAP
para sus 8 folds cada uno.

No cambia ningún resultado: usa exactamente los mismos hiperparámetros,
random_state=42 y max_test_explicar=300 que ya se usaron para los CSV/
PNG actuales de reports/tablas y reports/figures. La única diferencia
es que shap_explicabilidad._neutralizar_columnas_implausibles ahora
imprime el modelo Y el año del fold en el AVISO (antes solo decía "este
fold", sin identificarlo) -- así el aviso en la consola de esta corrida
dirá exactamente algo como:

    [cnn_lstm, fold 2017] AVISO: 2 columna(s) con SHAP implausible ...

que es el dato que falta para confirmar/corregir la Sección 5.4.4 de la
tesis (el texto actual dice "fold 2013", el docstring del código y
GUIA_EJECUCION.md dicen "fold 2017" -- esta corrida decide cuál es).

Requiere GPU/PyTorch -- pensado para correr en el servidor, no en un
entorno sin GPU. Ejecutar desde la raíz del proyecto:

    python src/evaluation/regenerar_shap_cnn_lstm_tabnet.py

Revisa la consola completa al terminar: cualquier línea que empiece con
"[cnn_lstm, fold ####] AVISO" o "[tabnet, fold ####] AVISO" identifica
el/los fold(s) realmente afectados. Si no aparece ningún AVISO para
alguno de los dos modelos, es que en esta corrida (con esta versión de
librerías) el problema no se reprodujo -- reportarlo también, es
información relevante para la Sección 5.4.4.
"""
import sys
from pathlib import Path

import pandas as pd

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))
sys.path.insert(0, str(_RAIZ_SRC / "features"))
sys.path.insert(0, str(_RAIZ_SRC / "models"))

from preparacion_modelado import cargar_dataset  # noqa: E402
from cnn_lstm import entrenar_evaluar_cnn_lstm, PANEL_MACRO_PATH  # noqa: E402
from tabnet_model import entrenar_evaluar_tabnet  # noqa: E402
from shap_explicabilidad import (  # noqa: E402
    explicar_cnn_lstm, explicar_tabnet, guardar_resumen_shap, graficar_resumen_shap,
)

RUTA_TABLAS = Path("reports/tablas")
RUTA_FIGURES = Path("reports/figures")


def regenerar_cnn_lstm(df_personas: pd.DataFrame, panel_macro: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Entrenando CNN-LSTM (8 folds, dataset completo)...")
    print("=" * 70)
    _, modelos = entrenar_evaluar_cnn_lstm(df_personas, panel_macro, min_anios_train=5)

    print("\n" + "=" * 70)
    print("Calculando SHAP (KernelSHAP) para CNN-LSTM -- revisa los AVISO abajo")
    print("=" * 70)
    explicaciones = explicar_cnn_lstm(df_personas, panel_macro, modelos, min_anios_train=5)

    for anio, resultado in explicaciones.items():
        resumen = resultado["resumen_global"]
        guardar_resumen_shap(resumen, RUTA_TABLAS / f"shap_global_cnn_lstm_{anio}.csv")
        graficar_resumen_shap(
            resumen, f"CNN-LSTM -- importancia SHAP global ({anio})",
            RUTA_FIGURES / f"shap_global_cnn_lstm_{anio}.png",
        )


def regenerar_tabnet(df_personas: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Entrenando TabNet (8 folds, dataset completo)...")
    print("=" * 70)
    _, modelos = entrenar_evaluar_tabnet(df_personas, min_anios_train=5)

    print("\n" + "=" * 70)
    print("Calculando SHAP (KernelSHAP) para TabNet -- revisa los AVISO abajo")
    print("=" * 70)
    explicaciones = explicar_tabnet(df_personas, modelos, min_anios_train=5)

    for anio, resultado in explicaciones.items():
        resumen = resultado["resumen_global"]
        guardar_resumen_shap(resumen, RUTA_TABLAS / f"shap_global_tabnet_{anio}.csv")
        graficar_resumen_shap(
            resumen, f"TabNet -- importancia SHAP global ({anio})",
            RUTA_FIGURES / f"shap_global_tabnet_{anio}.png",
        )


if __name__ == "__main__":
    df_personas = cargar_dataset()
    panel_macro = pd.read_csv(PANEL_MACRO_PATH)

    regenerar_cnn_lstm(df_personas, panel_macro)
    regenerar_tabnet(df_personas)

    print("\n" + "=" * 70)
    print("Listo. Busca en la consola de arriba líneas 'AVISO' -- el año que")
    print("aparece junto al nombre del modelo es el fold que hay que citar en")
    print("la Sección 5.4.4 de la tesis.")
    print("=" * 70)
