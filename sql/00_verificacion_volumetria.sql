-- =====================================================================
-- 00_verificacion_volumetria.sql
-- -----------------------------------------------------------------------
-- Propósito: correr ANTES de cualquier extracción. Superset/SQL Lab
-- limita la descarga a 100,000 filas por consulta, así que necesitamos
-- saber de antemano qué tablas requieren partición (chunking) y con qué
-- granularidad (año vs. periodo mensual/trimestral).
--
-- Supuesto a confirmar por Diego: el esquema analítico se llama
-- 'indicadores' (ClickHouse), tal como lo documentan las tres fichas
-- metodológicas adjuntas (ENEMDU, Latinobarómetro, V-Dem). Si en tu
-- Superset el esquema tiene otro nombre, solo reemplaza el prefijo.
-- =====================================================================

-- 1) V-Dem Ecuador: tabla país-año, volumen trivial (~18 filas si es
--    2007-2024). No requiere partición.
SELECT count(*) AS filas_vdem_ecuador
FROM indicadores.vdem
WHERE country_name = 'Ecuador';

-- 2) Latinobarómetro Ecuador: una fila por encuestado y ola. Con ~1,200
--    encuestados por ola y ~15-18 olas entre 1995-2024, el total esperado
--    es de decenas de miles: probablemente cabe en una sola consulta,
--    pero lo confirmamos aquí en vez de asumirlo.
SELECT count(*) AS filas_latinobarometro_ecuador
FROM indicadores.latinobarometro
WHERE country_name = 'Ecuador';

-- 2b) Detalle por año, para decidir si hace falta partir por research_year
SELECT research_year, count(*) AS n
FROM indicadores.latinobarometro
WHERE country_name = 'Ecuador'
GROUP BY research_year
ORDER BY research_year;

-- 3) diccionario_provincias: tabla de dimensión geográfica (provincia/
--    cantón/parroquia), tamaño fijo y pequeño. No requiere partición.
SELECT count(*) AS filas_diccionario_provincias
FROM indicadores.diccionario_provincias;

-- 4) ENEMDU persona: aquí SÍ es crítico medir antes de extraer. Un pull
--    histórico completo (2007-2024) casi seguro supera 100k filas; lo
--    que no sabemos de antemano es si basta con partir por periodo
--    (YYYYMM) o si algún periodo puntual (p. ej. diciembre, ronda
--    "ampliada" para representatividad provincial) ya excede el límite
--    por sí solo.
SELECT periodo, count(*) AS n
FROM indicadores.enemdu_persona
GROUP BY periodo
ORDER BY periodo;

-- -----------------------------------------------------------------------
-- Regla de decisión al revisar los resultados de 4):
--   - Si TODOS los periodos tienen < 100,000 filas -> usar
--     04_enemdu_persona_por_periodo.sql, ejecutando una consulta por periodo
--     (patrón ya incluido).
--   - Si ALGÚN periodo individual supera 100,000 filas -> ese periodo
--     necesita además paginación con LIMIT/OFFSET (o partición adicional
--     por rango de 'ciudad'/provincia). Aviso: no lo asumas, revisa el
--     resultado real antes de decidir.
-- =====================================================================
