-- =====================================================================
-- ⚠ DEPRECADO (revisión con tutor) -- NO USAR PARA REPROCESAR EL PROYECTO.
-- La extracción de Latinobarómetro vía Superset/SQL presentó
-- inconsistencias y no pudo utilizarse para la tesis. Los datos de
-- Latinobarómetro se obtienen ahora directamente desde las descargas
-- oficiales por ola (ver src/data_prep/latinobarometro_loader_crudo.py).
-- Esta query se conserva únicamente como registro histórico de la
-- extracción original. V-Dem y ENEMDU SÍ siguen usando Superset/SQL sin
-- cambios (ver 01_vdem_ecuador.sql y 04_enemdu_persona_por_periodo.sql).
-- =====================================================================
-- 02_latinobarometro_ecuador.sql
-- -----------------------------------------------------------------------
-- Fuente: indicadores.latinobarometro (microdatos armonizados a nivel
--         de encuestado, ya con nombres de variable normalizados según
--         el diccionario latinobarometro_glossary_candidates_BM y con
--         alias country_name/iso3 calculados, según Ficha Metodológica
--         Latinobarómetro).
-- Filtro: solo Ecuador (country_name = 'Ecuador'; alternativa robusta:
--         iso3 = 'ECU', o resp_country = 218 si trabajas contra la
--         tabla cruda -- confirmado con la muestra adjunta, donde el
--         código 218 corresponde a Ecuador en las 10 olas de ejemplo).
-- Volumen esperado: correr primero 00_verificacion_volumetria.sql (query 2).
-- Con ~1,200 encuestados/ola y como máximo ~18 olas (1995-2024, y ojo:
-- Latinobarómetro NO se levantó todos los años), se espera muy por
-- debajo de 100k filas. Si la verificación confirma esto, una sola
-- consulta basta; si no, particionar por research_year igual que ENEMDU.
--
-- IMPORTANTE (pendiente de decisión, no asumido): research_region /
-- research_city son códigos propios de Latinobarómetro, no equivalen
-- directamente a los códigos de provincia del INEC (2 dígitos) que usa
-- ENEMDU. La propia Ficha de Latinobarómetro lo señala como limitación
-- ("La geocodificación a nivel de región/ciudad depende de la calidad y
-- consistencia de... research_region y research_city, que pueden variar
-- entre oleadas"). Por eso se incluyen research_region/research_city
-- aquí (para inspección), pero el cruce persona-provincia con ENEMDU se
-- decide en la fase de preparación de datos (Pandas), no en esta query.
-- =====================================================================

SELECT
    research_year,
    country_name,
    iso3,
    research_region,
    research_city,
    research_city_size,

    -- Variable objetivo (a recodificar como binaria Satisfecho/No Satisfecho
    -- en la fase de preparación de datos)
    democ_satis,

    -- Variables de percepción política y económica
    democ_supp,
    left_right_scale,
    elections_vote,
    job_concern,
    econ_situation,
    resp_economic_perception,

    -- Bienes del hogar (proxy de condición material)
    goods_wash_mach,
    goods_car,
    goods_sewage,
    goods_hot_water,

    -- Confianza institucional
    confidence_congress,
    confidence_judiciary,
    confidence_church,
    confidence_police,
    confidence_army,
    confidence_political_parties,

    -- Sociodemográficas
    resp_sex,
    resp_age,
    resp_chief,
    resp_education,
    resp_employment,
    resp_religion

FROM indicadores.latinobarometro
WHERE country_name = 'Ecuador'
  AND research_year BETWEEN 2007 AND 2024
ORDER BY research_year
LIMIT 100000;
