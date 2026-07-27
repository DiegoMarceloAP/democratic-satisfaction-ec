# Extracción SQL (Fase 1 – Superset)

Consultas contra el esquema ClickHouse `indicadores`, diseñadas a partir de las tres fichas metodológicas adjuntas (ENEMDU, Latinobarómetro, V-Dem) y los esquemas de columnas (`DESCRIBE TABLE`) provistos.

**Latinobarómetro (revisión con tutor):** la extracción de `02_latinobarometro_ecuador.sql` presentó inconsistencias y no pudo utilizarse para la tesis; se conserva solo como registro histórico. Los datos de Latinobarómetro para el proyecto se obtienen ahora directamente desde las descargas oficiales por ola (ver `src/data_prep/latinobarometro_loader_crudo.py`). V-Dem y ENEMDU sí siguen viniendo de Superset/SQL sin cambios.

## Periodos de ENEMDU que superan 100k filas: RESUELTO

La verificación de volumetría confirmó que unos pocos periodos de `enemdu_persona` superan ampliamente el límite (del orden de 150k-300k), consistente con las rondas de diciembre con muestra ampliada para representatividad provincial (Guía ENEMDU). Solución adoptada: **paginación con `LIMIT 100000 OFFSET n*100000`** sobre `ORDER BY id_persona` (clave única y estable), ver `04b_enemdu_persona_periodo_grande_paginado.sql`.

Por qué paginación y no otra partición (p. ej. por provincia): no requiere conocer de antemano el tamaño exacto de cada periodo grande ni inventar un segundo eje de corte — simplemente se piden páginas de 100k hasta que una página devuelva menos de 100k filas (esa es la señal de parada). Cada página se guarda como un archivo `enemdu_persona_<periodo>_pagina<N>.csv`; `enemdu_processing.py` ya concatena todos los archivos que calcen con `enemdu_persona_*.csv`, así que no importa si un periodo quedó en 1 archivo o en 3.

## Orden de ejecución

1. `00_verificacion_volumetria.sql` — cuenta filas por tabla/periodo. Ejecutar primero siempre; de aquí sale la evidencia para decidir la partición real de ENEMDU.
2. `01_vdem_ecuador.sql` — una sola consulta (~18 filas).
3. ~~`02_latinobarometro_ecuador.sql`~~ — deprecado, no ejecutar (ver nota arriba). Latinobarómetro se obtiene con `src/data_prep/latinobarometro_loader_crudo.py`.
4. `03_diccionario_provincias.sql` — una sola consulta (tabla de dimensión).
5. `04_enemdu_persona_por_periodo.sql` — plantilla parametrizada; se ejecuta una vez por cada valor de `periodo` (YYYYMM) que arroje el paso 1.

## Supuestos de la extracción (a confirmar contra el Superset de destino)

- **Nombre del esquema**: `indicadores`, tal como lo documentan las fichas metodológicas. Si el Superset de destino apunta a otro esquema/base, es un cambio mecánico de prefijo en los 6 archivos.
- **Nombres de tabla**: `indicadores.vdem`, `indicadores.latinobarometro`, `indicadores.enemdu_persona`, `indicadores.enemdu_vivienda`, `indicadores.diccionario_provincias`. Los cuatro primeros están explícitamente nombrados en las fichas; `diccionario_provincias` solo aparece mencionada por nombre en las instrucciones del proyecto, no en las fichas -- verificar que ese es el nombre real de la tabla/vista en Superset antes de ejecutar.
- **Filtro de país**: en `indicadores.latinobarometro` y `indicadores.vdem` se filtra por `country_name = 'Ecuador'`. Se verificó contra la muestra de referencia que el código numérico crudo de Ecuador es `218` (coincide con ISO 3166-1 numérico), como alternativa si hiciera falta filtrar por `resp_country`/código en vez de por nombre.
- **Columnas de ENEMDU**: no se seleccionan las ~94 (persona) / ~107 (vivienda) columnas completas, sino el subconjunto que sustenta los indicadores de la propuesta (empleo, ingreso, pobreza, materiales de vivienda, hacinamiento). Para preguntas más específicas (p. ej. un NBI más detallado), la consulta puede ampliarse.

## Decisión sobre el cruce geográfico Latinobarómetro–ENEMDU: RESUELTA

**Decisión: Camino 1 — unión a nivel nacional-año.**

Cada encuestado de Latinobarómetro (Ecuador) se une con los indicadores de ENEMDU **agregados a nivel país por año**, no por provincia. `research_region`/`research_city` de Latinobarómetro se conservan en la extracción solo como variables de control/desagregación descriptiva (no como llave de unión geográfica), dado que la propia Ficha Metodológica de Latinobarómetro advierte que esos códigos no son equivalentes a las provincias INEC y pueden variar entre olas.

Implicaciones para la Fase 2 (Pandas), a tener en cuenta en el pipeline de preparación:

- Las consultas ENEMDU (`04`/`05`) siguen extrayendo microdatos con desagregación provincial (vía `ciudad`), porque igual sirven para calcular los indicadores agregados nacionales por año (se agregan de provincia → nación) y quedan disponibles si más adelante se decide una desagregación subnacional para otro análisis.
- El merge final será por llave `anio` (Latinobarómetro `research_year` = ENEMDU `anio` agregado nacional = V-Dem `year`), no por provincia+año.
- La variable de provincia de cada encuestado de Latinobarómetro (si se conserva) pasa a ser descriptiva, no funcional para el merge.
