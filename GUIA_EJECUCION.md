# Guía de ejecución — extracción, preparación de datos y modelado

Para instalación, estructura de carpetas y descripción general del proyecto, ver `README.md`. Esta guía cubre solo los pasos de ejecución, en orden, con las notas de troubleshooting relevantes.

## Paso 1 — Extracción

**V-Dem, ENEMDU y el diccionario de provincias** vienen de Superset. Ejecutar las consultas de `sql/` en Superset SQL Lab, en este orden:

| # | Archivo SQL | Cuántas veces correrlo | Dónde guardar el CSV |
|---|---|---|---|
| 1 | `00_verificacion_volumetria.sql` | Una vez (solo para revisar tamaños) | no se guarda |
| 2 | `01_vdem_ecuador.sql` | Una vez | `data/raw/vdem_ecuador.csv` |
| 3 | `03_diccionario_provincias.sql` | Una vez | `data/raw/diccionario_provincias.csv` |
| 4 | `04_enemdu_persona_por_periodo.sql` | Una vez por periodo (`${PERIODO}`) | `data/raw/enemdu_persona/enemdu_persona_<periodo>.csv` |
| 5 | `04b_enemdu_persona_periodo_grande_paginado.sql` | Solo periodos que superan 100k filas | `data/raw/enemdu_persona/enemdu_persona_<periodo>_pagina<N>.csv` |

Latinobarómetro no tiene consulta SQL en esta carpeta (ver `sql/README_extraccion.md`).

**Latinobarómetro:** descargar directamente del sitio oficial los microdatos de Ecuador de cada ola (2007-2024) y colocarlos en `data/raw/latinobarometro/` sin renombrar (el loader busca los nombres tal como los entrega Latinobarómetro). CSV para 2007-2010, SPSS `.sav` para el resto.

## Paso 2 — Procesamiento (Python)

Con los CSV de Superset en `data/raw/` y las descargas de Latinobarómetro en `data/raw/latinobarometro/`, correr en este orden exacto (`latinobarometro_processing.py` depende de `latinobarometro_loader_crudo.py`; `merge_final.py` depende de los otros tres):

```bash
cd <raíz-del-proyecto>
python src/data_prep/enemdu_processing.py
python src/data_prep/vdem_processing.py
python src/data_prep/latinobarometro_loader_crudo.py
python src/data_prep/latinobarometro_processing.py
python src/data_prep/merge_final.py
```

Si aparece un `AVISO` en consola, revisar la causa antes de continuar — no ignorarlo.

Al terminar, `data/processed/` tiene `dataset_modelado_personas.csv` (input de XGBoost/LightGBM/TabNet) y `panel_macro_anual.csv` (input de las ventanas del CNN-LSTM), además de los indicadores intermedios por fuente.

## Paso 3 — Entrenamiento

`notebooks/02_entrenamiento_modelos.ipynb` entrena los 5 modelos uno a la vez:

1. Correr `00_parametros_globales.ipynb` primero (rutas, semilla, GPU, `SAMPLING_MODE`).
2. Con `SAMPLING_MODE=True`, correr la celda de la muestra estratificada y luego cada modelo por separado (Regresión Logística → XGBoost → LightGBM → CNN-LSTM → TabNet), revisando resultados y tiempo antes de pasar al siguiente. La Regresión Logística no tiene hiperparámetros que ajustar según `SAMPLING_MODE` (configuración estándar fija); las celdas de CNN-LSTM/TabNet sí ajustan solas `n_epochs`/`max_epochs` según `SAMPLING_MODE` (5/10 en modo prueba, 15/100 en la corrida real).
3. Sección opcional: variables macro rezagadas (`rezagos_macro.py`) — compara XGBoost con y sin rezagos.
4. Cuando los 5 corran bien con la muestra: cambiar `SAMPLING_MODE=False` y volver a correr el notebook completo desde el inicio. Cada `resultados_*.csv` se guarda automáticamente en `reports/tablas/`.

## Paso 4 — Evaluación final: tabla comparativa y SHAP

```bash
python src/evaluation/metricas.py
```

Genera `reports/tablas/tabla_comparativa_final.csv` (Accuracy, F1, PR-AUC, tiempo de entrenamiento) y los gráficos de comparación en `reports/figures/`.

Para SHAP, `notebooks/02_entrenamiento_modelos.ipynb` ya incluye las celdas de `src/evaluation/shap_explicabilidad.py` justo después de cada uno de los 4 modelos de caja negra — no hace falta re-entrenar ni agregarlas a mano: `explicar_arbol` (XGBoost/LightGBM, TreeSHAP), `explicar_tabnet`/`explicar_cnn_lstm` (KernelSHAP), y `explicar_perfil_local` (auditoría de un encuestado puntual, justo después del TreeSHAP de XGBoost, para el marco XAI dual global+local). Corre por fold, no como resumen global, para poder comparar qué variables cambian entre años. La Regresión Logística no pasa por este módulo: su celda de entrenamiento en el notebook ya extrae e imprime directamente los coeficientes de mayor magnitud del último fold, que cumplen el mismo rol de explicabilidad sin necesidad de SHAP.

**Notas de validación:** `explicar_arbol` corrigió un bug real de compatibilidad `xgboost`/`shap`; `explicar_tabnet`/`explicar_cnn_lstm` corrigieron una inestabilidad numérica puntual en el fold 2017 (ver `src/evaluation/README.md`).

**Costo de KernelSHAP:** `explicar_cnn_lstm`/`explicar_tabnet` submuestrean por defecto el fold de prueba a `max_test_explicar=300` encuestados (estratificado por clase) antes de calcular SHAP — es el paso más caro de esta fase. Usar `max_test_explicar=None` para explicar el fold completo.

## Si algo falla

Los scripts fallan con un mensaje claro (`FileNotFoundError`) si falta algún CSV de entrada, en vez de fallar silenciosamente. Ese error indica que falta ejecutar el paso anterior de esta guía.
