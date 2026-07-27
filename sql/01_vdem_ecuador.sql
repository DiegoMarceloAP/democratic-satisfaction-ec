-- =====================================================================
-- 01_vdem_ecuador.sql
-- -----------------------------------------------------------------------
-- Fuente: indicadores.vdem (1 fila por país-año, cobertura 1789-2024
--         según Ficha Metodológica V-Dem).
-- Filtro: solo Ecuador, y solo años relevantes para la tesis (2007-2024).
-- Volumen esperado: ~18 filas -> muy por debajo del límite de 100k,
--                    una sola consulta es suficiente.
-- Columnas: se seleccionan únicamente los índices de alto nivel y de
--           distribución del poder documentados en la ficha (sección 2),
--           más los identificadores necesarios para el merge posterior
--           por año con Latinobarómetro/ENEMDU.
-- =====================================================================

SELECT
    country_name,
    country_text_id,          -- 'ECU'
    country_id,
    year,

    -- Índices de democracia de alto nivel (escala 0-1)
    v2x_polyarchy,             -- democracia electoral
    v2x_libdem,                -- democracia liberal
    v2x_partipdem,             -- democracia participativa
    v2x_delibdem,              -- democracia deliberativa
    v2x_egaldem,               -- democracia igualitaria

    -- Libertades, estado de derecho y calidad electoral (0-1)
    v2x_freexp_altinf,
    v2xel_frefair,
    v2xcl_rol,
    v2x_jucon,
    v2xlg_legcon,

    -- Igualdad sustantiva y distribución del poder
    -- (v2peer* vienen originalmente en escala [-4,4]; la Ficha V-Dem
    --  indica que ya se reescalan a [0,1] dentro de indicadores.vdem
    --  con x' = (x+4)/8. Verificar en Superset que el rango de los
    --  valores devueltos sea [0,1] antes de asumirlo en Pandas.)
    v2xeg_eqprotec,
    v2xeg_eqaccess,
    v2xeg_eqdr,
    v2pepwrses,
    v2pepwrsoc,
    v2pepwrgen,
    v2pepwrort,
    v2pepwrgeo

FROM indicadores.vdem
WHERE country_name = 'Ecuador'
  AND year BETWEEN 2007 AND 2024
ORDER BY year
LIMIT 100000;
