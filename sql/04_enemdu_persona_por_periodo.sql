-- =====================================================================
-- 04_enemdu_persona_por_periodo.sql
-- -----------------------------------------------------------------------
-- Fuente: indicadores.enemdu_persona (microdatos de personas, ENEMDU-INEC).
-- Por qué se particiona por periodo: el histórico completo 2007-2024
-- excede casi con certeza el límite de 100,000 filas de Superset. La
-- muestra adjunta ya viene truncada exactamente en 100,000 filas, lo que
-- confirma que en una sola consulta sin filtrar por periodo se llega al
-- tope. La partición por 'periodo' (formato YYYYMM) es la granularidad
-- natural de la tabla y, salvo que 00_verificacion_volumetria.sql
-- muestre lo contrario, cada periodo individual debería quedar muy por
-- debajo de 100k filas (ENEMDU es una encuesta por muestreo, no un censo).
--
-- Cómo usar esta plantilla:
--   1) Corre primero la query 4) de 00_verificacion_volumetria.sql para
--      obtener la lista real de periodos disponibles y sus tamaños.
--   2) Reemplaza '${PERIODO}' por cada valor de periodo (p. ej. '200712')
--      y ejecuta/descarga una vez por periodo en Superset SQL Lab.
--   3) CONFIRMADO por Diego (verificación real de volumetría): unos pocos
--      periodos superan ampliamente 100k filas (del orden de 150k-300k),
--      consistente con las rondas de diciembre con muestra ampliada para
--      representatividad provincial que documenta la Guía ENEMDU. Para
--      esos periodos puntuales usar la plantilla paginada en
--      04b_enemdu_persona_periodo_grande_paginado.sql en vez de esta.
--
-- Columnas seleccionadas: solo las necesarias para (a) identificación y
-- llave de unión con vivienda, (b) indicadores de empleo/ingreso/pobreza
-- que pide la Fase 2 (tasa de desempleo provincial, ingreso promedio,
-- aproximación de desigualdad), y (c) variables sociodemográficas base.
-- No se seleccionan todas las p01-p77 (preguntas específicas de módulos
-- que no son insumo directo de los indicadores agregados de la tesis).
-- =====================================================================

SELECT
    periodo,                  -- YYYYMM
    id_persona,
    id_hogar,
    id_vivienda,
    ciudad,                   -- código geográfico crudo; en Pandas se
                               -- deriva provincia = substr(digits, 1, 2),
                               -- canton = substr(digits, 1, 4) según la
                               -- Ficha Metodológica ENEMDU
    area,                      -- urbano/rural
    zona,
    sector,
    estrato,                   -- estrato socioeconómico

    -- Empleo / actividad
    condact,                   -- condición de actividad (PEA/PET/ocupado)
    empleo,
    desempleo,
    secemp,                    -- formal/informal/doméstico
    rama1,                     -- rama de actividad CIIU rev.4 (1 dígito)
    grupo1,                    -- grupo ocupacional CIUO-08 (1 dígito)

    -- Ingreso y pobreza
    ingpc,                     -- ingreso per cápita del hogar
    ingrl,                     -- ingreso laboral
    pobreza,
    epobreza,                  -- pobreza extrema

    -- Factor de expansión (obligatorio para cualquier agregación)
    fexp,

    -- Sociodemográficas base
    p03,                       -- edad (código 99 = No sabe, tratar como NA)
    nnivins                    -- nivel de instrucción

FROM indicadores.enemdu_persona
WHERE periodo = '${PERIODO}'   -- ej: '200712'  (reemplazar en cada corrida)
ORDER BY id_persona
LIMIT 100000;
