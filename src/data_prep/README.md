# src/data_prep/

Preparación de datos (CRISP-DM: Data Preparation). Limpia, tipifica y calcula los indicadores intermedios de las tres fuentes (ENEMDU, V-Dem, Latinobarómetro) antes del merge final en `merge_final.py`.

ENEMDU y V-Dem parten de los CSV extraídos por SQL/Superset en `data/raw/` (sin cambios). Latinobarómetro parte de las descargas oficiales directas por ola en `data/raw/latinobarometro/`: la consulta original contra Superset presentó inconsistencias durante la revisón de extración y fue reemplazada por la fuente directa (ver `sql/README_extraccion.md`).

-  `enemdu_processing.py`: procesa los microdatos de personas de ENEMDU y calcula indicadores agregados ANUALES a nivel nacional (tasa de participación, tasa de desempleo, empleo formal/informal, ingreso promedio, Gini ponderado, pobreza por ingresos). Deriva también el código de provincia (trazabilidad/QA), aunque el merge final es nacional-año, no provincial.

  **Nota metodológica sobre el cálculo anual:** todos los indicadores se calculan ponderados por `fexp` (factor de expansión), nunca como promedio simple de microdatos -- necesario porque ENEMDU es una muestra probabilística compleja. Las definiciones de PET/PEA/ocupado replican textualmente la Ficha Metodológica ENEMDU. La "anualización" agrupa todas las rondas cuyo `periodo` (YYYYMM) cae en el mismo año calendario y pondera por `fexp` sobre ese grupo -- en la práctica, 2007-2019 agregan ~4 rondas trimestrales (marzo/junio/septiembre/diciembre) y 2020 en adelante agregan hasta 12 rondas mensuales (ENEMDU pasó a encuesta continua mensual desde septiembre 2020). Esto es una decisión metodológica explícita, no necesariamente idéntica al indicador anual oficial que publica el INEC (que en ciertos reportes usa solo la ronda de diciembre). Se detectaron además 2 huecos reales en la fuente -- noviembre 2021 y octubre 2025, sin registros en absoluto -- verificados contra la volumetría real pedida a Superset, no un descuido de la extracción. 

-  `latinobarometro_loader_crudo.py`: construye `data/raw/latinobarometro_ecuador_corregido.csv` a partir de las 13 olas descargadas directamente de Latinobarómetro (CSV crudo 2007-2010, SPSS .sav 2011-2024 -- ver docstring del módulo para por qué se mezclan dos formatos y el corrimiento de columnas que se detectó y corrigió en los 4 años en CSV).

-  `latinobarometro_processing.py`: construye la variable objetivo binaria ("Satisfecho" vs. "No Satisfecho" con la democracia, a partir de `democ_satis`) y las variables sociodemográficas/actitudinales, a nivel de encuestado individual (no se agrega, a diferencia de ENEMDU). Incluye imputación/interpolación de los años sin ola de Latinobarómetro (2012, 2014, 2019, 2021-2022), con un límite `MAX_HUECO_INTERPOLABLE=2` para no interpolar huecos demasiado largos sin evidencia.

-  `vdem_processing.py`: procesa los indicadores de calidad democrática de V-Dem para Ecuador (una fila por año, sin necesidad de agregación).

-  `merge_final.py`: une las 3 fuentes por llave `anio` (Latinobarómetro `research_year` = ENEMDU agregado nacional = V-Dem `year`) -- decisión de merge nacional-año confirmada y documentada en `sql/README_extraccion.md`, no por provincia.

Salida: `data/processed/dataset_modelado_personas.csv` (nivel encuestado, para los 4 modelos) y `data/processed/panel_macro_anual.csv` (panel anual nacional, para `ventanas_temporales.py` y `rezagos_macro.py` en `src/features/`).
