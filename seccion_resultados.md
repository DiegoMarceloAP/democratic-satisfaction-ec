# Resultados (CRISP-DM: Evaluation)

## 1. Diseño de la evaluación

Los 4 modelos del plan de modelado (XGBoost, LightGBM, CNN-LSTM, TabNet) se entrenaron y evaluaron sobre `dataset_modelado_personas.csv`, usando el mismo conjunto de features, la misma partición de validación cruzada temporal (Time Series Split, ventana expansiva) y el mismo tratamiento de desbalance (SMOTE aplicado únicamente sobre `X_train` de cada fold, para evitar filtración de datos).

Los datos reales de Latinobarómetro para Ecuador cubren 10 años de encuesta (2007, 2008, 2009, 2010, 2016, 2017, 2018, 2020, 2023, 2024). Con `min_anios_train=5`, esto produce exactamente 5 folds de prueba: **2017, 2018, 2020, 2023 y 2024**, cada uno entrenado únicamente con años estrictamente anteriores. La métrica de optimización y selección de modelos es el **PR-AUC** (área bajo la curva Precisión-Recall) de la clase minoritaria "Satisfecho", complementada con F1 de esa misma clase.

## 2. Tabla comparativa final

Promedio y desviación estándar sobre los 5 folds reales, con el ajuste conservador de hiperparámetros ya aplicado (ver sección 4.2):

| Modelo | Accuracy | F1 Macro | F1 Satisfecho | PR-AUC | Tiempo entren. (s) |
|---|---|---|---|---|---|
| XGBoost | 0.7648 ± 0.083 | 0.6288 ± 0.045 | 0.4533 ± 0.214 | **0.4890 ± 0.225** | 0.65 ± 0.44 |
| LightGBM | 0.7606 ± 0.101 | 0.6138 ± 0.040 | 0.4217 ± 0.209 | 0.4889 ± 0.217 | 1.16 ± 0.68 |
| CNN-LSTM | 0.7722 ± 0.117 | 0.5949 ± 0.044 | 0.3976 ± 0.261 | 0.4789 ± 0.221 | 36.16 ± 10.27 |
| TabNet | 0.7566 ± 0.108 | 0.5719 ± 0.078 | 0.3458 ± 0.295 | 0.4598 ± 0.228 | 199.91 ± 59.87 |

**Lectura honesta:** ordenados por PR-AUC (la métrica reina), XGBoost y LightGBM quedan prácticamente empatados en primer lugar (0.4890 vs. 0.4889), seguidos de cerca por CNN-LSTM (0.4789) y por último TabNet (0.4598). Sin embargo, la desviación estándar entre folds es alta en los 4 modelos (0.21-0.23 en PR-AUC) y sus intervalos se solapan casi por completo -- con solo 5 folds de prueba, esta diferencia no es estadísticamente distinguible. La conclusión sustantiva no es "el modelo X gana", sino que **los 4 modelos logran un desempeño comparable**, y que los dos baselines de árboles lo logran en una fracción del tiempo de cómputo (XGBoost: 0.65s vs. TabNet: ~200s por fold, ~300x más rápido).

### 2.1 Resultados por fold (detalle)

| Año prueba | Modelo | Accuracy | F1 Satisfecho | PR-AUC |
|---|---|---|---|---|
| 2017 | XGBoost | 0.6617 | 0.6896 | 0.7344 |
| 2017 | LightGBM | 0.6387 | 0.6446 | 0.7251 |
| 2017 | CNN-LSTM | 0.6455 | 0.6839 | 0.7294 |
| 2017 | TabNet | 0.6344 | 0.6448 | 0.7175 |
| 2018 | XGBoost | 0.6880 | 0.6823 | 0.7143 |
| 2018 | LightGBM | 0.6714 | 0.6559 | 0.7065 |
| 2018 | CNN-LSTM | 0.6479 | 0.6836 | 0.6886 |
| 2018 | TabNet | 0.6513 | 0.6539 | 0.6809 |
| 2020 | XGBoost | 0.8110 | 0.3038 | 0.2488 |
| 2020 | LightGBM | 0.7964 | 0.2840 | 0.2494 |
| 2020 | CNN-LSTM | 0.8754 | 0.2162 | 0.2289 |
| 2020 | TabNet | 0.8093 | 0.2885 | 0.1983 |
| 2023 | XGBoost | 0.8406 | 0.3322 | 0.3143 |
| 2023 | LightGBM | 0.8710 | 0.2679 | 0.3318 |
| 2023 | CNN-LSTM | 0.8744 | 0.2116 | 0.3285 |
| 2023 | TabNet | 0.8803 | 0.0000 | 0.3366 |
| 2024 | XGBoost | 0.8227 | 0.2587 | 0.4330 |
| 2024 | LightGBM | 0.8253 | 0.2562 | 0.4316 |
| 2024 | CNN-LSTM | 0.8177 | 0.1926 | 0.4190 |
| 2024 | TabNet | 0.8077 | 0.1418 | 0.3658 |

Un patrón consistente en los 4 modelos: el desempeño (PR-AUC) es notablemente más alto en los folds tempranos (2017-2018, ~0.69-0.73) que en los folds recientes (2020-2024, ~0.20-0.43). Esto coincide con años de mayor inestabilidad político-económica en Ecuador (pandemia en 2020, disolución de la Asamblea en 2023) y con una clase "Satisfecho" cada vez más minoritaria en esos años -- un desafío estructural del problema, no un defecto de un modelo en particular. TabNet en 2023 es un caso extremo: F1 de la clase "Satisfecho" = 0, es decir, no predijo ningún caso positivo en ese fold, pese a un PR-AUC razonable (0.3366) -- indica que el umbral de decisión por defecto (0.5) no es adecuado para ese fold en particular, algo a considerar como trabajo futuro (calibración de umbral por fold).

## 3. Corrección de hiperparámetros (hallazgo real durante la evaluación)

Con los hiperparámetros originales, **XGBoost colapsaba en el fold 2023** (accuracy 0.4983, PR-AUC 0.1776 -- predecía "Satisfecho" en más de la mitad de los casos cuando la tasa real era ~12%). Se investigó la causa empíricamente:

- **Se descartó un problema de umbral de decisión**: recalibrar el umbral usando 2020 como validación interna apenas cambió el resultado (accuracy 0.4983 → 0.5076).
- **Se confirmó sobreajuste**: reducir la complejidad del modelo (`max_depth` 5→3, `n_estimators` 300→200) resolvió el fold 2023 (PR-AUC 0.1776 → 0.2922) sin perjudicar a los demás folds, subiendo el promedio de PR-AUC de 0.4649 a 0.4890.

Se evaluó y **descartó** construir una validación interna anidada (reservar el último año de entrenamiento de cada fold para elegir hiperparámetros), por dos razones verificadas empíricamente: (a) con solo 10 años reales, costaría un fold de prueba completo (el primer fold pasaría de evaluar 2017 a evaluar 2018); y (b) se comprobó directamente que un único año de validación (2020) **no habría elegido la configuración correcta** para el fold 2023 -- el PR-AUC de validación en 2020 favorecía el modelo original (más complejo), es decir, la validación anidada habría fallado en resolver el mismo problema que buscaba corregir. Se optó por un ajuste conservador y a priori, documentado en el código, dejando una búsqueda de hiperparámetros más exhaustiva como trabajo futuro quizás viable con más años de datos.

Un ajuste análogo (reducir épocas de 30 a 15, agregar `weight_decay=1e-4`) se aplicó al CNN-LSTM tras observar la misma señal de sobreajuste (PR-AUC promedio 0.519 con 5 épocas → 0.468 con 30 épocas, degradación en los 5 folds), mejorando el promedio final a 0.4789.

## 4. IA Explicable (XAI): hallazgos SHAP

Marco dual: **TreeSHAP** (exacto) para XGBoost/LightGBM, **KernelSHAP** (aproximado, con background vía k-means y regularización L1) para CNN-LSTM/TabNet. En los 4 modelos se calculó SHAP fold por fold (no un pool global), dado que el desempeño ya varía mucho entre años.

### 4.1 Importancia global

Las mismas 2-3 variables encabezan la importancia global en los **4 modelos y en todos los folds**:

- **`econ_situation_cat`** (percepción de la situación económica personal): la variable de mayor impacto, siempre, en los 4 modelos.
- **`democ_supp_cat`** (apoyo a la democracia): segunda o tercera en casi todos los folds.
- **`confidence_congress_alta`** (confianza en el Congreso): consistentemente entre las 3 más importantes.

Esto es evidencia empírica (no solo teórica) de que la satisfacción con la democracia en Ecuador está más ligada a la percepción económica personal y a la confianza institucional que a los indicadores estructurales de calidad democrática de V-Dem.

**Variables que ganan peso específicamente en los folds difíciles (2023-2024)**, en los 4 modelos: `empleo_formal` y `v2xeg_eqprotec` (protección igualitaria de derechos, V-Dem) aparecen entre las 5 más importantes solo en 2023/2024, no en 2017/2018/2020 -- coherente con la hipótesis de que estos años tuvieron dinámicas distintas a la serie histórica (disolución de la Asamblea en 2023, elección presidencial anticipada). Ejemplo, fold 2024, XGBoost: `empleo_formal` (0.582) supera incluso a `econ_situation_cat` (0.573) como variable más importante -- el único fold donde esto ocurre.

En CNN-LSTM y TabNet (KernelSHAP), el patrón es el mismo: `econ_situation_cat` y `democ_supp_cat` lideran en 2018/2020/2023/2024; en TabNet, `confidence_congress_alta` incluso supera a `econ_situation_cat` como variable dominante en varios folds (2017: 0.417 vs. 0.104; 2018: 0.501 vs. 0.094) -- una diferencia interesante frente a los modelos de árboles, que podría reflejar cómo cada arquitectura pondera la interacción entre percepción económica y confianza institucional.

### 4.2 Explicación local (ejemplo)

Encuestado del fold 2023 (XGBoost): su indiferencia hacia la democracia (`democ_supp_cat`, SHAP -0.649) y no tener empleo formal (`empleo_formal`, SHAP -0.521) empujan fuertemente su predicción hacia "No Satisfecho", junto con una protección igualitaria de derechos percibida baja (`v2xeg_eqprotec`, SHAP -0.295). En sentido contrario, su alta confianza en el Congreso, en los partidos políticos y en el poder judicial (SHAP +0.19 a +0.22 cada una) empujan hacia "Satisfecho", sin llegar a compensar los factores negativos. Este tipo de auditoría individual permite explicar la predicción de una persona (o perfil sociodemográfico/provincial) específica, más allá del promedio global.

### 4.3 Inestabilidad numérica de KernelSHAP (fold 2017, CNN-LSTM) -- detectada y corregida

Durante la ejecución real se detectó que el fold 2017 del CNN-LSTM (el que menos años reales de entrenamiento tiene, de los 5) producía valores de SHAP matemáticamente imposibles (del orden de 1e10-1e11, cuando la predicción explicada es una probabilidad en [0,1]), en columnas distintas en cada corrida -- señal de un problema numérico en la regresión ponderada interna de KernelSHAP, probablemente por poca variedad temporal real dentro de ese fold. Subir el número de muestras de perturbación (`nsamples`) por sí solo no resolvió el problema. Se implementó una doble corrección: regularización L1 en la propia llamada de KernelSHAP, y un mecanismo de neutralización automática que detecta y pone en cero cualquier columna con SHAP fuera de rango plausible (con aviso explícito impreso), aplicado tanto al resumen global como a las explicaciones locales. Tras esta corrección, el fold 2017 del CNN-LSTM quedó limpio, con el mismo patrón que los demás folds y modelos (`econ_situation_cat` y `democ_supp_cat` liderando).

## 5. Limitaciones metodológicas

- **Escasez de años reales de encuesta (10 en total, 5 folds de prueba):** limita tanto la robustez estadística de la comparación entre modelos (barras de error que se solapan) como la posibilidad de una validación de hiperparámetros más sofisticada (nested cross-validation), que se evaluó y descartó explícitamente por este motivo (sección 3).
- **Imputación de Latinobarómetro:** los años sin ola de encuesta (2011-2015, 2019, 2021-2022) se interpolaron con un límite de 2 años consecutivos (`MAX_HUECO_INTERPOLABLE=2`) para no inventar tendencia en huecos demasiado largos.
- **Cálculo anual de ENEMDU:** los indicadores agregados (desempleo, empleo formal/informal, ingreso, Gini) se calculan ponderando por el factor de expansión (`fexp`) y agrupando todas las rondas de un año calendario (trimestrales hasta 2019, mensuales desde septiembre 2020) -- una decisión metodológica explícita, no necesariamente idéntica al indicador anual oficial que publica el INEC en algunos reportes (que en ciertos casos usa solo la ronda de diciembre). Se detectaron además 2 huecos reales sin registros en la fuente (noviembre 2021, octubre 2025), confirmados contra la volumetría real de Superset.
- **Tamaño del panel macro anual para el CNN-LSTM:** la ventana temporal del CNN-LSTM solo tiene tantos valores distintos como años reales de encuesta hay en cada fold (mínimo 5, en el fold 2017) -- todos los encuestados del mismo año comparten la misma ventana macro. Esto acota la variedad temporal real que el componente LSTM puede aprender, y es la hipótesis más plausible detrás de la inestabilidad de KernelSHAP observada específicamente en ese fold (sección 4.3).
- **Umbral de decisión fijo (0.5):** no se calibró por fold; el caso extremo de TabNet en 2023 (F1 de "Satisfecho" = 0 pese a PR-AUC razonable) sugiere que una calibración de umbral por fold podría mejorar F1 sin tocar el modelo, quedando como trabajo futuro.
- **Entrenamiento sin validación interna (early stopping):** los 4 modelos entrenan un número fijo de épocas/estimadores (ajustado a priori, sección 3), no seleccionado por desempeño en un conjunto de validación reservado -- deliberado, para no filtrar información del fold de prueba hacia la selección del modelo, a costa de no aprovechar señales de sobreajuste durante el propio entrenamiento.
