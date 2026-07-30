# Deep Learning for Modeling Democratic Satisfaction and Socioeconomic Inequality in Ecuador using Latinobarómetro, V-Dem, and ENEMDU Data

Tesis de Maestría en Inteligencia Artificial (Universidad Yachay Tech): *Deep Learning for Modeling Democratic Satisfaction and Socioeconomic Inequality in Ecuador using Latinobarómetro, V-Dem, and ENEMDU Data*.

## Descripción

El proyecto modela la satisfacción ciudadana con la democracia en Ecuador (2007-2024) como un problema de clasificación binaria, integrando —por primera vez a nivel de encuestado individual— tres fuentes de naturaleza distinta: encuestas de opinión política (Latinobarómetro), indicadores de calidad democrática (V-Dem) y condiciones socioeconómicas oficiales (ENEMDU/INEC). Sigue la metodología CRISP-DM: extracción y preparación de datos, modelado comparativo (Regresión Logística + 2 modelos de machine learning + 2 arquitecturas de deep learning) y evaluación bajo un marco de IA explicable (SHAP) dual — complementado, en el caso de la Regresión Logística, con sus propios coeficientes, ya que no requiere SHAP para ser interpretable.

**Variable objetivo:** `satisfecho_democracia` — binaria, construida a partir de `democ_satis` de Latinobarómetro (1 = Satisfecho, 0 = No Satisfecho; casos sin respuesta válida se descartan, no se imputan).

**Métrica principal:** PR-AUC (área bajo la curva Precision-Recall) de la clase minoritaria "Satisfecho", complementada con F1 de esa misma clase — elegidas por el desbalance de clases inherente al problema, que varía año a año. La validación usa un esquema de Time Series Split de ventana expansiva (8 folds de prueba reales: 2013, 2015, 2016, 2017, 2018, 2020, 2023 y 2024), nunca K-Fold clásico, para no filtrar información del futuro hacia el entrenamiento.

## Fuentes de datos

| Fuente | Descripción | Cobertura |
|---|---|---|
| **Latinobarómetro** | Encuesta de opinión pública anual (Corporación Latinobarómetro); percepción ciudadana, actitudes políticas, variable objetivo. Se obtiene **directamente de las descargas oficiales por ola** (la extracción SQL original presentó inconsistencias y fue retirada del proyecto, ver `sql/README_extraccion.md`). | Ecuador, 13 olas reales entre 2007-2024 (2007-2011, 2013, 2015-2018, 2020, 2023-2024); huecos de máximo 2 años, interpolados. |
| **V-Dem** (Varieties of Democracy) | Índices de calidad democrática institucional (poliarquía electoral, democracia liberal, democracia igualitaria), codificados por expertos país-específicos. Se extrae vía SQL/Superset. | Ecuador, panel anual continuo 2007-2024, sin huecos. |
| **ENEMDU** | Encuesta Nacional de Empleo, Desempleo y Subempleo (INEC); indicadores socioeconómicos agregados (desempleo, ingreso, informalidad, desigualdad). Se extrae vía SQL/Superset, ponderada por el factor de expansión `fexp`. | Ecuador, agregada anualmente a nivel nacional, 2007-2024. |

Latinobarómetro (nivel individuo) y V-Dem/ENEMDU (nivel país-año) se integran mediante un *contextual data merge*: el vector de contexto país-año se difunde hacia cada encuestado de ese año (ver `src/data_prep/merge_final.py`).

## Estructura del proyecto

```
<raíz del proyecto>/
├── sql/                     consultas SQL para Superset (V-Dem y ENEMDU; Latinobarómetro no usa esta vía)
├── scripts/                 automatización de extracción (ENEMDU vía Superset)
├── src/
│   ├── data_prep/           Fase 2 -- procesamiento de las 3 fuentes + merge final
│   ├── features/            ventanas temporales, rezagos macro, SMOTENC, config de features
│   ├── models/               Regresión Logística, XGBoost, LightGBM, CNN-LSTM, TabNet, línea base trivial
│   └── evaluation/           Time Series Split, tabla comparativa, pruebas estadísticas, SHAP
├── notebooks/
│   ├── 00_parametros_globales.ipynb   semillas, GPU, SAMPLING_MODE, rutas (correr primero)
│   ├── 01_eda_balance_clases.ipynb    análisis exploratorio de datos
│   └── 02_entrenamiento_modelos.ipynb entrenamiento y evaluación paso a paso
├── reports/
│   ├── figures/              gráficos (EDA, SHAP, curvas PR)
│   └── tablas/                tabla comparativa final y resultados por fold/modelo
├── data/                     NO versionado (raw >500MB; processed se regenera desde raw)
├── GUIA_EJECUCION.md         guía paso a paso de ejecución (detalle ampliado de este README)
└── requirements.txt
```

## Descripción de carpetas

| Carpeta | Contenido |
|---|---|
| `sql/` | Consultas SQL para extraer V-Dem y ENEMDU de Superset (Fase 1 de CRISP-DM). Latinobarómetro no tiene consulta aquí (ver `src/data_prep/latinobarometro_loader_crudo.py`). |
| `scripts/` | `extraer_enemdu_superset.py` — automatiza la paginación de las consultas de ENEMDU contra Superset. |
| `src/data_prep/` | Procesamiento de cada fuente (`enemdu_processing.py`, `vdem_processing.py`, `latinobarometro_loader_crudo.py`, `latinobarometro_processing.py`) y el merge final (`merge_final.py`) que produce los datasets de modelado. |
| `src/features/` | Selección centralizada de variables (`config_features.py`), preparación común para los modelos no secuenciales (`preparacion_modelado.py`), ventanas temporales del CNN-LSTM (`ventanas_temporales.py`), rezagos macro para modelos tabulares (`rezagos_macro.py`) y SMOTENC por fold (`smote_train.py`). |
| `src/models/` | Los 5 modelos del plan de modelado (`baseline_logistic_regression.py`, `baseline_xgboost.py`, `baseline_lightgbm.py`, `cnn_lstm.py`, `tabnet_model.py`) y la línea base trivial de referencia (`baseline_trivial.py`). |
| `src/evaluation/` | Partición temporal (`time_series_split.py`), tabla comparativa final (`metricas.py`), prueba estadística formal Friedman/Wilcoxon (`pruebas_estadisticas.py`) y el marco SHAP dual (`shap_explicabilidad.py`). |
| `notebooks/` | Los 3 notebooks de ejecución, ver detalle abajo. |
| `reports/figures/` | Gráficos generados (EDA, comparación de modelos, SHAP global por fold). |
| `reports/tablas/` | Resultados por fold de cada modelo, línea base trivial, pruebas estadísticas y la tabla comparativa final consolidada. |
| `data/` | `raw/` (descargas originales de las 3 fuentes) y `processed/` (datasets ya integrados: `dataset_modelado_personas.csv`, `panel_macro_anual.csv`, entre otros) — ninguna de las dos se versiona en git. |

Cada subcarpeta de `src/` tiene su propio `README.md` con el detalle de qué hace cada archivo, decisiones de diseño y hallazgos de la validación.

## Notebooks

| Notebook | Propósito |
|---|---|
| `00_parametros_globales.ipynb` | Configura rutas, semilla de reproducibilidad (`RANDOM_STATE=42`), detecta automáticamente GPU (`torch.cuda.is_available()`) y define el flag `SAMPLING_MODE` (`True` para iterar rápido con una muestra estratificada, `False` para la corrida final con el dataset completo). Se corre primero, antes que cualquier otro notebook. |
| `01_eda_balance_clases.ipynb` | Análisis exploratorio del dataset de personas ya unificado: balance de la variable objetivo por año, proporción de valores faltantes por columna, correlaciones entre variables numéricas, variables categóricas frente al target, distribución demográfica y evolución anual de los indicadores macro. |
| `02_entrenamiento_modelos.ipynb` | Entrena y evalúa los 5 modelos (Regresión Logística, XGBoost, LightGBM, CNN-LSTM, TabNet) uno a la vez sobre los 8 folds de Time Series Split, con SMOTENC por fold. Los 4 modelos de caja negra (XGBoost, LightGBM, CNN-LSTM, TabNet) incluyen SHAP (global y local) inmediatamente después de cada uno; la Regresión Logística incluye en su lugar sus propios coeficientes, ya interpretables sin necesidad de SHAP. Incluye una sección opcional de variables macro rezagadas para los modelos tabulares. |

## Módulos Python

**`src/data_prep/`** — `enemdu_processing.py` (indicadores agregados anuales ponderados por `fexp`), `vdem_processing.py` (índices V-Dem a nivel país-año), `latinobarometro_loader_crudo.py` (descarga oficial por ola, combina `.sav` y `.csv` corregido), `latinobarometro_processing.py` (variable objetivo y variables individuales), `merge_final.py` (integración multinivel de las 3 fuentes).

**`src/features/`** — `config_features.py` (lista única de features compartida por los 5 modelos), `preparacion_modelado.py` (imputación y codificación comunes, después del split), `ventanas_temporales.py` (ventanas `[batch, timesteps, features]` del CNN-LSTM), `rezagos_macro.py` (variables macro rezagadas para modelos tabulares), `smote_train.py` (SMOTENC únicamente sobre `X_train` de cada fold; sin ruta alternativa a SMOTE genérico, por diseño).

**`src/models/`** — `baseline_logistic_regression.py` (baseline estadístico adicional, incorporado a pedido del tutor; codifica categóricas *one-hot* y estandariza numéricas, a diferencia de los modelos de árboles), `baseline_xgboost.py`, `baseline_lightgbm.py` (machine learning clásico), `cnn_lstm.py` (arquitectura mixta CNN + LSTM), `tabnet_model.py` (arquitectura nativa para datos tabulares), `baseline_trivial.py` (clasificador de clase mayoritaria, punto de referencia sin información).

**`src/evaluation/`** — `time_series_split.py` (partición de ventana expansiva), `metricas.py` (tabla comparativa final), `pruebas_estadisticas.py` (Friedman + Wilcoxon pareado con corrección de Holm-Bonferroni), `shap_explicabilidad.py` (TreeSHAP para árboles, KernelSHAP para redes, global y local).

## Instalación y ejecución

Requiere Python 3.10+.

**1. Crear entorno e instalar dependencias:**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```
Si el equipo tiene GPU NVIDIA, instalar la variante CUDA de `torch` según [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) en vez de la versión CPU por defecto — `00_parametros_globales.ipynb` detecta la GPU automáticamente sin tocar el resto del código.

**2. Obtener los datos crudos** (`data/raw/`, no versionado):
- V-Dem, ENEMDU y el diccionario de provincias: ejecutar las consultas de `sql/` en Superset SQL Lab (o `scripts/extraer_enemdu_superset.py` para automatizar ENEMDU) y guardar los CSV según `data/raw/LEEME.md`.
- Latinobarómetro: descargar directamente del sitio oficial de Latinobarómetro (CSV para 2007-2010, SPSS `.sav` para 2011-2024) y guardar en `data/raw/latinobarometro/`.

**3. Procesar y unificar los datos** (`data/processed/`), en este orden exacto:
```bash
cd <raíz-del-proyecto>
python src/data_prep/enemdu_processing.py
python src/data_prep/vdem_processing.py
python src/data_prep/latinobarometro_loader_crudo.py
python src/data_prep/latinobarometro_processing.py
python src/data_prep/merge_final.py
```
Esto produce `dataset_modelado_personas.csv` (input de XGBoost/LightGBM/TabNet) y `panel_macro_anual.csv` (input de las ventanas del CNN-LSTM).

**4. Entrenar y evaluar los modelos:**
1. Abrir `notebooks/02_entrenamiento_modelos.ipynb` y correr `00_parametros_globales.ipynb` primero.
2. Con `SAMPLING_MODE=True`, correr cada modelo (Regresión Logística → XGBoost → LightGBM → CNN-LSTM → TabNet) sobre una muestra reducida, para validar que todo funciona.
3. Cambiar a `SAMPLING_MODE=False` y volver a correr el notebook completo desde el inicio — esta es la corrida que alimenta los resultados finales. Cada `resultados_*.csv` se guarda automáticamente en `reports/tablas/`.

**5. Consolidar la tabla comparativa final:**
```bash
python src/evaluation/metricas.py
```
Genera `reports/tablas/tabla_comparativa_final.csv` y los gráficos de comparación en `reports/figures/`.

Ver `GUIA_EJECUCION.md` para el detalle ampliado de cada paso, incluyendo notas sobre bugs encontrados y corregidos durante la validación.

## Reproducibilidad

- **Semillas fijas** (`RANDOM_STATE=42`) en NumPy, `random` y PyTorch, definidas una sola vez en `00_parametros_globales.ipynb`.
- **Rutas relativas**: sin rutas absolutas hardcodeadas; el proyecto puede moverse de carpeta sin romper nada (recrear `.venv` en ese caso, no copiarlo).
- **`SAMPLING_MODE`**: permite iterar rápido con una muestra estratificada antes de la corrida final con el dataset completo, sin cambiar código.
