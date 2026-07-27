# Democratic Satisfaction & Socioeconomic Inequality in Ecuador — Deep Learning

Tesis de Maestría en Inteligencia Artificial (Universidad Yachay Tech): *Deep Learning for Modeling Democratic Satisfaction and Socioeconomic Inequality in Ecuador using Latinobarómetro, V-Dem, and ENEMDU Data*.

Clasificación binaria ("Satisfecho" vs. "No Satisfecho" con la democracia) a nivel de encuestado, integrando encuestas de opinión (Latinobarómetro), indicadores de calidad democrática (V-Dem) e indicadores socioeconómicos anuales (ENEMDU/INEC), bajo la metodología CRISP-DM.

## Estado del proyecto

Las 3 fases técnicas del alcance están completas y validadas contra datos reales (dataset completo, no muestra):

| Fase (CRISP-DM) | Contenido | Estado |
|---|---|---|
| Preparación de datos | Extracción (Superset), procesamiento ENEMDU/Latinobarómetro/V-Dem, merge final | ✅ Completo |
| Modelado | XGBoost, LightGBM, CNN-LSTM, TabNet -- Time Series Split (5 folds reales) + SMOTE por fold | ✅ Completo |
| Evaluación / XAI | Tabla comparativa (PR-AUC, F1, Accuracy), SHAP dual (TreeSHAP + KernelSHAP, global + local) | ✅ Completo |

Resultado agregado (PR-AUC, métrica de optimización, promedio de 5 folds): XGBoost 0.489, LightGBM 0.489, CNN-LSTM 0.479, TabNet 0.460 -- desempeño estadísticamente comparable entre los 4 modelos (ver `seccion_resultados.md` para el detalle completo y las limitaciones metodológicas).

## Estructura del proyecto

```
<raíz del proyecto>/
├── sql/                    consultas para Superset (extracción, Fase 1)
├── scripts/                automatización de extracción (extraer_enemdu_superset.py)
├── src/
│   ├── data_prep/          Fase 2 -- procesamiento ENEMDU/Latinobarómetro/V-Dem + merge
│   ├── features/            ventanas temporales, rezagos, SMOTE, config de features
│   ├── models/              XGBoost, LightGBM, CNN-LSTM, TabNet
│   └── evaluation/          Time Series Split, tabla comparativa, SHAP
├── notebooks/
│   ├── 00_parametros_globales.ipynb   semillas, GPU, SAMPLING_MODE, rutas (correr primero)
│   ├── 01_eda_balance_clases.ipynb    análisis exploratorio
│   └── 02_entrenamiento_modelos.ipynb entrenamiento y evaluación paso a paso
├── reports/
│   ├── figures/             gráficos (EDA, SHAP, curvas PR)
│   └── tablas/              tabla comparativa final y resultados por fold
├── data/                    NO versionado -- ver sección "Datos" abajo
├── seccion_resultados.md    borrador de la sección de resultados de la tesis
├── GUIA_EJECUCION.md        guía paso a paso de ejecución
└── requirements.txt
```

Cada subcarpeta de `src/` tiene su propio `README.md` con el detalle de qué hace cada archivo, decisiones de diseño y hallazgos de la validación.

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

Si el equipo tiene GPU NVIDIA y se quiere que PyTorch la use, instalar la variante CUDA de `torch` según [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) en vez de la versión CPU que instala `requirements.txt` por defecto -- `notebooks/00_parametros_globales.ipynb` detecta automáticamente si hay GPU disponible (`torch.cuda.is_available()`) y ajusta el dispositivo (`DEVICE`) sin necesidad de tocar el resto del código.

## Datos

`data/raw/` y `data/processed/` **no se versionan** (raw pesa >500MB de microdatos ENEMDU; processed se regenera determinísticamente a partir de raw). Para reconstruirlos:

1. Ejecutar las consultas de `sql/` en Superset SQL Lab (o `scripts/extraer_enemdu_superset.py` para automatizar ENEMDU) y guardar los CSV en `data/raw/` según `data/raw/LEEME.md`.
2. Correr los scripts de `src/data_prep/` en orden (ver `GUIA_EJECUCION.md`, Paso 2) para producir `data/processed/dataset_modelado_personas.csv` y `data/processed/panel_macro_anual.csv`.

Ver `GUIA_EJECUCION.md` para la guía completa paso a paso (extracción → preparación → modelado → evaluación/SHAP).

## Reproducibilidad y portabilidad entre equipos

- **Semillas fijas** (`RANDOM_STATE=42`) en NumPy, `random` y PyTorch, definidas una sola vez en `notebooks/00_parametros_globales.ipynb`.
- **Rutas relativas**: todas las rutas del proyecto se resuelven a partir de `Path.cwd()` (raíz del proyecto o `notebooks/`), no hay rutas absolutas hardcodeadas en el código -- el proyecto puede moverse o renombrarse de carpeta sin romper nada. Si se mueve el proyecto, recrear `.venv` (los entornos virtuales de Python no son portables entre rutas) en vez de copiarlo.
- **`SAMPLING_MODE`**: flag en `00_parametros_globales.ipynb` para iterar rápido con una muestra estratificada (20%) en un equipo de capacidad media, y correr con el dataset completo (`SAMPLING_MODE=False`) para los resultados finales -- sin cambiar ninguna otra parte del código.
- **Detección automática de hardware**: `torch.cuda.is_available()` decide CPU vs. GPU sin configuración manual.

## Resultados

`seccion_resultados.md` contiene la sección de resultados completa (tabla comparativa, hallazgos SHAP global/local, limitaciones metodológicas), redactada a partir de las corridas reales en `reports/tablas/`.
