-- =====================================================================
-- 04b_enemdu_persona_periodo_grande_paginado.sql
-- -----------------------------------------------------------------------
-- Uso: SOLO para los periodos que 00_verificacion_volumetria.sql marcó
-- por encima de 100,000 filas (confirmado: unos pocos periodos, del
-- orden de 150k-300k -- probablemente rondas de diciembre con muestra
-- ampliada para representatividad provincial).
--
-- Estrategia: paginación con LIMIT/OFFSET sobre un ORDER BY determinístico
-- (id_persona, que es único por fila). Es la solución más robusta porque
-- no depende de conocer el tamaño exacto de antemano ni de inventar una
-- segunda dimensión de partición (provincia, upm, etc.): simplemente se
-- piden páginas de 100,000 filas hasta que una página vuelva con menos
-- de 100,000 filas (esa es la señal de que ya no queda nada más).
--
-- Importante: id_persona debe ser único y el ORDER BY debe ser el MISMO
-- en todas las páginas de un mismo periodo, para garantizar que no se
-- pierdan ni se dupliquen filas entre una página y otra.
--
-- Cómo usar:
--   1) Identifica el/los periodo(s) grande(s) con la query 4) de
--      00_verificacion_volumetria.sql (columna 'n').
--   2) Para cada periodo grande, calcula cuántas páginas necesitas:
--         n_paginas = ceil(n_filas_del_periodo / 100000)
--      (p. ej. si el periodo tiene 230,000 filas, son 3 páginas: offsets
--      0, 100000 y 200000).
--   3) Corre esta consulta una vez por página, cambiando '${PERIODO}' y
--      '${OFFSET}' (0, 100000, 200000, ...), y guarda cada resultado
--      como un archivo DISTINTO:
--          enemdu_persona_<periodo>_pagina1.csv  (OFFSET 0)
--          enemdu_persona_<periodo>_pagina2.csv  (OFFSET 100000)
--          enemdu_persona_<periodo>_pagina3.csv  (OFFSET 200000)
--          ...
--      No hace falta unirlos manualmente: enemdu_processing.py ya
--      concatena TODOS los CSV que calcen con el patrón
--      'enemdu_persona_*.csv' en data/raw/enemdu_persona/, así que da
--      igual si un periodo quedó en 1 archivo o en 3.
--   4) Regla de parada: cuando una página devuelva menos de 100,000
--      filas, esa es la última página de ese periodo -- no hace falta
--      pedir una página adicional vacía.
--
-- Columnas: idénticas a 04_enemdu_persona_por_periodo.sql (ver ese
-- archivo para el detalle de por qué se eligió cada una).
-- =====================================================================

SELECT
    periodo,
    id_persona,
    id_hogar,
    id_vivienda,
    ciudad,
    area,
    zona,
    sector,
    estrato,
    condact,
    empleo,
    desempleo,
    secemp,
    rama1,
    grupo1,
    ingpc,
    ingrl,
    pobreza,
    epobreza,
    fexp,
    p03,
    nnivins

FROM indicadores.enemdu_persona
WHERE periodo = '${PERIODO}'    -- ej: '201712'
ORDER BY id_persona              -- clave estable: NO cambiar entre páginas del mismo periodo
LIMIT 100000
OFFSET ${OFFSET};                -- 0, luego 100000, luego 200000, ... hasta que una página devuelva < 100000 filas
