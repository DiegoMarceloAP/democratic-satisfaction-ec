# Guía de ejecución — Fase 1, 2 y 3 (extracción, preparación de datos y modelado)

Todo lo construido hasta ahora vive en **esta misma carpeta** (la raíz del proyecto). No necesitas crear otro proyecto ni otra carpeta.

```
<raíz del proyecto>/
├── sql/                    consultas para Superset (ya listas)
├── scripts/                automatización de extracción (extraer_enemdu_superset.py)
├── src/
│   ├── data_prep/          scripts de Fase 2 (ya listos)
│   ├── features/           ventanas temporales, SMOTE, config y preparación de features (listos)
│   ├── models/             XGBoost, LightGBM, CNN-LSTM, TabNet (listos)
│   └── evaluation/         Time Series Split, tabla comparativa (metricas.py) y SHAP (todo listo)
├── notebooks/
│   ├── 00_parametros_globales.ipynb   semillas, GPU, SAMPLING_MODE, rutas
│   ├── 01_eda_balance_clases.ipynb    EDA de balance de clases
│   └── 02_entrenamiento_modelos.ipynb entrenamiento local paso a paso + rezagos macro (Paso 4 de esta guía)
├── reports/
│   ├── figures/            gráficos (EDA, SHAP, curvas PR)
│   └── tablas/             tabla comparativa final de modelos
├── data/
│   ├── raw/
│   │   ├── vdem_ecuador.csv                    <- de Superset
│   │   ├── diccionario_provincias.csv          <- de Superset
│   │   ├── enemdu_persona/                     <- de Superset (un CSV por periodo)
│   │   │   └── (un CSV por periodo)
│   │   ├── latinobarometro_codebook_oficial.xlsx   <- crosswalk oficial (ya provisto)
│   │   └── latinobarometro/                    <- descargas DIRECTAS de Latinobarómetro (NO Superset)
│   │       ├── Latinobarometro_2007_Ecuador_Csv_esp_v1.csv   (... 2008, 2009, 2010: CSV crudo)
│   │       └── Latinobarometro_2011_Ecuador_Spss_esp_v1.sav  (... 2013...2024: SPSS .sav)
│   ├── processed/            <- los scripts de Python escriben aquí solos (no lo llenas a mano)
│   └── models/               modelos entrenados serializados (.pkl, .pt, .zip)
└── requirements.txt
```

**Latinobarómetro ya NO viene de Superset** (revisión con tutor: la extracción SQL tenía inconsistencias y no se pudo usar -- ver `sql/02_latinobarometro_ecuador.sql`, deprecado, y `src/data_prep/latinobarometro_loader_crudo.py` para el detalle completo). Los datos se obtienen directamente del sitio de Latinobarómetro: CSV crudo para 2007-2010, SPSS `.sav` para 2011-2024 (el `.sav` de 2007-2010 no se puede leer por una limitación de la librería `pyreadstat`/ReadStat con esos 4 archivos específicos -- por eso esos 4 años usan CSV). V-Dem y ENEMDU siguen igual, sin cambios.

## Paso 1 — Superset (SQL) + descarga directa de Latinobarómetro

**V-Dem, ENEMDU y el diccionario de provincias** siguen viniendo de Superset. Ejecuta las consultas de `sql/` en Superset SQL Lab, en este orden, y descarga cada resultado como CSV con el nombre indicado:

| # | Archivo SQL | Cuántas veces correrlo | Dónde guardar el CSV |
|---|---|---|---|
| 1 | `00_verificacion_volumetria.sql` | Una vez (solo para revisar tamaños, no genera insumo) | no se guarda |
| 2 | `01_vdem_ecuador.sql` | Una vez | `data/raw/vdem_ecuador.csv` |
| 3 | `03_diccionario_provincias.sql` | Una vez | `data/raw/diccionario_provincias.csv` |
| 4 | `04_enemdu_persona_por_periodo.sql` | Una vez POR periodo (reemplazando `${PERIODO}`) | `data/raw/enemdu_persona/enemdu_persona_<periodo>.csv` |
| 5 | `04b_enemdu_persona_periodo_grande_paginado.sql` | Solo para los periodos que superan 100k filas (varias páginas cada uno) | `data/raw/enemdu_persona/enemdu_persona_<periodo>_pagina<N>.csv` |

`02_latinobarometro_ecuador.sql` está **deprecado** -- no lo ejecutes.

**Latinobarómetro:** descarga directamente del sitio de Latinobarómetro los microdatos de Ecuador de cada ola (2007-2024) y colócalos en `data/raw/latinobarometro/` sin renombrarlos (el loader busca los nombres de archivo tal como los entrega Latinobarómetro). Necesitas el CSV para 2007-2010 y el SPSS `.sav` para el resto.

## Paso 2 — Python (Fase 2, preparación de datos)

Con los CSV de Superset en `data/raw/` y las descargas de Latinobarómetro en `data/raw/latinobarometro/`, corres los scripts **en este orden** (`latinobarometro_processing.py` depende de que `latinobarometro_loader_crudo.py` ya haya corrido; `merge_final.py` depende de que los otros tres procesamientos ya hayan corrido):

```bash
cd <raíz-del-proyecto>
python src/data_prep/enemdu_processing.py
python src/data_prep/vdem_processing.py
python src/data_prep/latinobarometro_loader_crudo.py
python src/data_prep/latinobarometro_processing.py
python src/data_prep/merge_final.py
```

Cada script imprime en consola avisos si algo no cuadra (columnas fuera de rango, años sin dato, etc.) — si aparece un `AVISO`, no ignorarlo: revisar la causa antes de continuar con el siguiente paso.

Al terminar, `data/processed/` va a tener:
- `enemdu_indicadores_anuales.csv`
- `vdem_ecuador_anual.csv`
- `latinobarometro_ecuador_personas.csv`
- `latinobarometro_ecuador_serie_anual.csv`
- `dataset_modelado_personas.csv` ← este es el que se usa para los modelos baseline (XGBoost/LightGBM/TabNet/MLP)
- `panel_macro_anual.csv` ← este es el que se usa para las ventanas del CNN-LSTM

## Paso 3 — Modelado

Los 4 modelos del plan (`src/models/baseline_xgboost.py`, `baseline_lightgbm.py`, `cnn_lstm.py`, `tabnet_model.py`) y toda la preparación de datos que comparten (`src/features/config_features.py`, `preparacion_modelado.py`, `smote_train.py`, `ventanas_temporales.py`, `rezagos_macro.py`, `src/evaluation/time_series_split.py`) ya están escritos. Ver los `README.md` de cada carpeta para el detalle de qué hace cada archivo.

## Paso 4 — Entrenamiento local, paso a paso (prueba con muestra)

Antes de la corrida final, `notebooks/02_entrenamiento_modelos.ipynb` deja correr los 4 modelos **uno a la vez** con una muestra reducida y estratificada (por año y clase), para probar que todo funciona sin saturar tu equipo:

1. Corre `pip install -r requirements.txt` si no lo has hecho.
2. Abre `notebooks/02_entrenamiento_modelos.ipynb`.
3. Corre la celda de `%run 00_parametros_globales.ipynb` y la de verificación del entorno (confirma que `xgboost`, `lightgbm`, `torch` y `pytorch_tabnet` están instalados).
4. Corre la celda de la muestra estratificada.
5. Corre **una sección de modelo a la vez** (XGBoost → LightGBM → CNN-LSTM → TabNet), revisando resultados y tiempo antes de pasar a la siguiente. Si tu equipo se satura, interrumpe esa celda (no hace falta reiniciar todo). Las celdas de CNN-LSTM/TabNet ajustan solas `n_epochs`/`max_epochs` según `SAMPLING_MODE` (5/10 en modo prueba, 15/100 en la corrida real) -- no hay que editarlas a mano.
6. Sección 6 (opcional, a pedido del tutor): variables macro rezagadas (`rezagos_macro.py`) -- compara XGBoost con y sin rezagos sobre la misma muestra.
7. Cuando los 4 corran bien con la muestra: cambia `SAMPLING_MODE = False` en `00_parametros_globales.ipynb` y vuelve a correr el notebook completo desde el inicio — esa es la corrida que alimenta la tabla comparativa de la tesis. Guarda cada `resultados_*.csv` que imprime el notebook (ya se guardan solos en `reports/tablas/`).

**Nota sobre CNN-LSTM y TabNet:** dependen de `torch` (y TabNet además de `pytorch-tabnet`), que no se pudieron instalar en el entorno usado para escribir el código inicialmente (paquete demasiado grande para ese entorno) -- su lógica de datos se validó primero con datos sintéticos, y el entrenamiento real se ejecutó y validó posteriormente en un equipo con esas dependencias instaladas (ver resultados finales en `reports/tablas/` y el detalle en `src/models/README.md`).

**Bugs encontrados y corregidos durante la validación con el dataset real** (no deberían reaparecer al correr el notebook, quedan documentados por transparencia): (a) las columnas `confidence_*_alta` se leían como texto en vez de número desde el CSV — se fuerza a `float` en `preparar_features`; (b) XGBoost rechazaba predicciones cuando una categoría (ej. `__faltante__`) aparecía en el año de prueba pero nunca en los años de entrenamiento de ese fold — ahora `castear_categoricas` declara las mismas categorías en ambos.

## Paso 5 — Evaluación final: tabla comparativa y SHAP

Con los 4 `resultados_*.csv` de la corrida completa (`SAMPLING_MODE=False`) ya en `reports/tablas/`:

```bash
cd <raíz-del-proyecto>
python src/evaluation/metricas.py
```

Esto arma `reports/tablas/tabla_comparativa_final.csv` (Accuracy, F1, PR-AUC y tiempo de entrenamiento, ordenada por PR-AUC) y los gráficos de barras en `reports/figures/`.

Para SHAP (IA Explicable), no hace falta re-entrenar los 4 modelos por separado: `src/evaluation/shap_explicabilidad.py` expone `explicar_arbol` (XGBoost/LightGBM, TreeSHAP), `explicar_tabnet` y `explicar_cnn_lstm` (KernelSHAP), que reciben el `df`/`panel_macro` y la lista de `modelos` que ya devolvió cada `entrenar_evaluar_*` del Paso 4. El notebook `02_entrenamiento_modelos.ipynb` YA incluye estas celdas justo después de entrenar cada modelo (XGBoost, LightGBM, CNN-LSTM, TabNet) -- no hay que agregarlas a mano. También incluye, justo después del TreeSHAP de XGBoost, un ejemplo de explicación **LOCAL** (`explicar_perfil_local`): auditoría de un encuestado puntual del fold más reciente, para cumplir el marco XAI dual (global + local) que pide la propuesta.

Corre por fold (año), no un solo resumen global -- así ves si las variables importantes cambian entre años.

**Nota:** `explicar_arbol` (XGBoost/LightGBM) está validado contra el dataset completo real -- corre sin error; en el camino se encontró y corrigió un bug real de compatibilidad `xgboost`/`shap` (ver `src/evaluation/README.md`). `explicar_tabnet`/`explicar_cnn_lstm` (KernelSHAP) también están validadas contra los modelos reales, incluyendo la corrección de una inestabilidad numérica puntual en el fold 2017 del CNN-LSTM (ver `src/evaluation/README.md`, sección de KernelSHAP).

**Optimización de costo (KernelSHAP, sobre todo CNN-LSTM):** `explicar_cnn_lstm`/`explicar_tabnet` submuestrean por defecto el fold de prueba a `max_test_explicar=300` encuestados (estratificado por la clase real) antes de calcular SHAP -- es, con diferencia, el paso más caro de esta fase (KernelSHAP evalúa el modelo `nsamples` veces por cada encuestado explicado). No afecta `nsamples` (ya validado para evitar la inestabilidad numérica del fold 2017) ni el entrenamiento; usa `max_test_explicar=None` si en algún momento quieres explicar el fold completo.

## Requisitos previos

Instala las dependencias del proyecto (incluye lo necesario para Fase 2 y Fase 3):

```bash
cd <raíz-del-proyecto>
pip install -r requirements.txt
```

Si tu equipo tiene GPU NVIDIA y quieres que PyTorch la use, instala la variante CUDA de `torch` siguiendo las instrucciones de https://pytorch.org/get-started/locally/ en vez de la versión CPU que instala `requirements.txt` por defecto.

## Si algo falla

Los scripts están escritos para fallar con un mensaje claro (`FileNotFoundError`) si falta algún CSV de entrada, en vez de fallar silenciosamente. Si ves ese error, es que falta ejecutar/guardar el paso anterior de esta guía.
