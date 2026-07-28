# Extracción SQL (Fase 1 – Superset)

Consultas contra el esquema ClickHouse `indicadores`, diseñadas a partir de las tres fichas metodológicas adjuntas (ENEMDU, Latinobarómetro, V-Dem) y los esquemas de columnas (`DESCRIBE TABLE`) provistos.

**Latinobarómetro** no se extrae de Superset (la consulta original presentó inconsistencias durante la revisión con el tutor y fue retirada del proyecto). Los datos se obtienen directamente desde las descargas oficiales por ola (ver `src/data_prep/latinobarometro_loader_crudo.py`). V-Dem y ENEMDU sí siguen viniendo de Superset/SQL sin cambios.

## Archivos SQL

**`00_verificacion_volumetria.sql`** — cuenta filas por tabla y, para ENEMDU, por periodo. Superset limita cada consulta a 100,000 filas, así que este archivo se corre primero para saber qué tablas caben en una sola consulta y cuáles necesitan extraerse por partes. También revisa el volumen de Latinobarómetro por año en Superset, como referencia histórica, aunque esa fuente ya no se usa para la extracción.

**`01_vdem_ecuador.sql`** — extrae los indicadores de V-Dem para Ecuador, un año por fila (2007-2024). Selecciona los índices de democracia de alto nivel y de distribución del poder documentados en la Ficha Metodológica de V-Dem. Volumen pequeño (~18 filas): una sola consulta es suficiente.

**`03_diccionario_provincias.sql`** — extrae el catálogo de provincia/cantón/parroquia bajo el estándar político-administrativo del INEC, usado para traducir los códigos geográficos crudos de ENEMDU. Tabla de dimensión pequeña: una sola consulta.

**`04_enemdu_persona_por_periodo.sql`** — extrae los microdatos de personas de ENEMDU, un periodo (`YYYYMM`) a la vez, porque el histórico completo excede el límite de filas por consulta. Selecciona solo las columnas necesarias para los indicadores de empleo, ingreso, pobreza y las variables sociodemográficas base que usa la tesis (no las ~94 columnas completas de la tabla). Se ejecuta una vez por cada periodo detectado en `00_verificacion_volumetria.sql`.

**`04b_enemdu_persona_periodo_grande_paginado.sql`** — mismas columnas que `04`, para los pocos periodos (rondas de diciembre, con muestra ampliada para representatividad provincial) que superan el límite de filas por consulta incluso filtrando por periodo. Se ejecuta por partes sucesivas hasta cubrir el periodo completo; cada parte se guarda como un archivo separado y `enemdu_processing.py` los concatena automáticamente.

## Orden de ejecución

1. `00_verificacion_volumetria.sql` — cuenta filas por tabla/periodo. Ejecutar primero siempre; de aquí sale la evidencia para decidir cómo extraer ENEMDU.
2. `01_vdem_ecuador.sql` — una sola consulta.
3. `03_diccionario_provincias.sql` — una sola consulta.
4. `04_enemdu_persona_por_periodo.sql` — una vez por cada periodo detectado en el paso 1; si algún periodo supera el límite de filas, usar `04b_enemdu_persona_periodo_grande_paginado.sql` en su lugar para ese periodo.

Latinobarómetro no tiene consulta SQL en esta carpeta; se obtiene con `src/data_prep/latinobarometro_loader_crudo.py` (ver nota arriba).

## Supuestos de la extracción (a confirmar contra el Superset de destino)

- **Nombre del esquema**: `indicadores`, tal como lo documentan las fichas metodológicas. Si el Superset de destino apunta a otro esquema/base, es un cambio mecánico de prefijo en estos archivos.
- **Nombres de tabla**: `indicadores.vdem`, `indicadores.enemdu_persona`, `indicadores.enemdu_vivienda`, `indicadores.diccionario_provincias`. Las tres primeras están explícitamente nombradas en las fichas; `diccionario_provincias` solo aparece mencionada por nombre en las instrucciones del proyecto, no en las fichas -- verificar que ese es el nombre real de la tabla/vista en Superset antes de ejecutar.
- **Filtro de país**: en `indicadores.vdem` se filtra por `country_name = 'Ecuador'`. Se verificó contra la muestra de referencia que el código numérico crudo de Ecuador es `218` (coincide con ISO 3166-1 numérico), como alternativa si hiciera falta filtrar por código en vez de por nombre.
- **Columnas de ENEMDU**: no se seleccionan las ~94 (persona) / ~107 (vivienda) columnas completas, sino el subconjunto que sustenta los indicadores de la propuesta (empleo, ingreso, pobreza, materiales de vivienda, hacinamiento). Para preguntas más específicas (p. ej. un NBI más detallado), la consulta puede ampliarse.

## Decisión sobre el cruce geográfico Latinobarómetro–ENEMDU

**Decisión: unión a nivel nacional-año.**

Cada encuestado de Latinobarómetro (Ecuador) se une con los indicadores de ENEMDU **agregados a nivel país por año**, no por provincia. `research_region`/`research_city` de Latinobarómetro se conservan en la extracción solo como variables de control/desagregación descriptiva (no como llave de unión geográfica), dado que la propia Ficha Metodológica de Latinobarómetro advierte que esos códigos no son equivalentes a las provincias INEC y pueden variar entre olas.

Implicaciones para la Fase 2 (Pandas), a tener en cuenta en el pipeline de preparación:

- Las consultas ENEMDU (`04`/`04b`) siguen extrayendo microdatos con desagregación provincial (vía `ciudad`), porque igual sirven para calcular los indicadores agregados nacionales por año (se agregan de provincia → nación) y quedan disponibles si más adelante se decide una desagregación subnacional para otro análisis.
- El merge final será por llave `anio` (Latinobarómetro `research_year` = ENEMDU `anio` agregado nacional = V-Dem `year`), no por provincia+año.
- La variable de provincia de cada encuestado de Latinobarómetro (si se conserva) pasa a ser descriptiva, no funcional para el merge.
